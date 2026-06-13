"""Promote HAR activity logs to Timeline events and dispatch live action notifications."""

from __future__ import annotations

import json
from dataclasses import dataclass
import logging
from datetime import datetime, timedelta, timezone

from sqlalchemy.orm import Session

from vision_ops_alerting.config import settings
from vision_ops_alerting.db.models import AlertRule, EmailTemplateRecord, Event, HarActivityLog, new_id
from vision_ops_alerting.email_layout import build_template_vars
from vision_ops_alerting.schemas import Actor, Evidence, IndustrialContext, Links
from vision_ops_alerting.services.alert_snapshot import enrich_context
from vision_ops_alerting.services.email_templates import ensure_builtin_templates
from vision_ops_alerting.services.events import (
    _looks_like_image_url,
    log_email_delivery,
    log_telegram_delivery,
)
from vision_ops_alerting.services.har_activity_store import HAR_CRITICAL_CONFIDENCE, PRIMARY_ACTION
from vision_ops_alerting.services.rule_config import (
    find_har_rule_for_label,
    rule_confidence_threshold,
    rule_event_severity,
    rule_has_active_channel,
)

logger = logging.getLogger(__name__)


def _resolve_har_snapshot_thumb(snapshot_url: str | None) -> str | None:
    if not _looks_like_image_url(snapshot_url):
        return None
    assert snapshot_url is not None
    thumb = snapshot_url.strip()
    if thumb.startswith("/api/vision/"):
        return f"{settings.public_api_base.rstrip('/')}{thumb}"
    if thumb.startswith("/api/alerting/") or thumb.startswith("/api/har/"):
        return f"{settings.public_base_url.rstrip('/')}{thumb}"
    return thumb

HAR_CASE_TYPE = "har_action_deviation"
HAR_TEMPLATE_ID = "har.action_deviation"

HAR_DETECT_CASE_TYPE = "har_action_detected"
HAR_DETECT_TEMPLATE_ID = "har.action_detected"


def _ensure_har_rule(db: Session) -> AlertRule | None:
    rule = (
        db.query(AlertRule)
        .filter(AlertRule.case_type == HAR_CASE_TYPE)
        .first()
    )
    if rule:
        return rule
    rule = AlertRule(
        id=new_id("rule"),
        icon="precision_manufacturing",
        title="HAR Action Deviation",
        description="Non-assembly or low-confidence HAR activity on a live camera feed.",
        zone="HAR LIVE",
        case_type=HAR_CASE_TYPE,
        severity="WARNING",
        enabled=True,
        notify_email=False,
        email_template_id=None,
    )
    db.add(rule)
    db.flush()
    return rule


def _recent_event(
    db: Session,
    *,
    case_type: str,
    camera_id: str,
    label: str,
    cooldown_sec: int,
) -> bool:
    since = datetime.now(timezone.utc) - timedelta(seconds=cooldown_sec)
    recent = (
        db.query(Event)
        .filter(
            Event.camera_id == camera_id,
            Event.case_type == case_type,
            Event.occurred_at >= since,
            Event.title.contains(label),
        )
        .first()
    )
    return recent is not None


@dataclass(frozen=True)
class HarSessionSummary:
    action_label: str
    threshold_pct: int
    trigger_pct: int
    mean_pct: int
    peak_pct: int
    qualifying_count: int
    har_logs: str
    telegram_text: str


def _summarize_session_for_action(
    db: Session,
    log: HarActivityLog,
    *,
    action_label: str,
    threshold: float,
    max_recent: int = 5,
) -> HarSessionSummary:
    """Build a condensed log: only readings for this action at or above the alert threshold."""
    q = db.query(HarActivityLog).filter(HarActivityLog.camera_id == log.camera_id)
    if log.session_id:
        q = q.filter(HarActivityLog.session_id == log.session_id)
    rows = q.order_by(HarActivityLog.occurred_at.asc()).limit(120).all()

    threshold_pct = int(round(threshold * 100))
    trigger_pct = int(round(float(log.confidence or 0) * 100))
    qualifying = [
        r
        for r in rows
        if (r.predicted_label or "") == action_label
        and float(r.confidence or 0) >= threshold
    ]
    confs = [float(r.confidence or 0) for r in qualifying]
    mean_pct = int(round(sum(confs) / len(confs) * 100)) if confs else trigger_pct
    peak_pct = int(round(max(confs) * 100)) if confs else trigger_pct
    count = len(qualifying)

    recent_pcts: list[str] = []
    for row in qualifying[-max_recent:]:
        recent_pcts.append(f"{int(round((row.confidence or 0) * 100))}%")

    recent_str = ", ".join(recent_pcts) if recent_pcts else f"{trigger_pct}%"
    person_label = log.actor_name or (f"Track #{log.actor_track_id}" if log.actor_track_id else None)
    person_part = f" · {person_label}" if person_label else ""
    har_logs = (
        f"{action_label}{person_part} · mean {mean_pct}% · peak {peak_pct}% · "
        f"{count} read(s) ≥{threshold_pct}% · recent: {recent_str}"
    )
    person_line = f"Person: {person_label}\n" if person_label else ""
    telegram_text = (
        f"ℹ️ {action_label}\n"
        f"{log.camera_id} · {log.model_id or '—'}\n"
        f"{person_line}"
        f"Mean {mean_pct}% · peak {peak_pct}% · {count}× ≥{threshold_pct}%\n"
        f"Recent: {recent_str}"
    )
    return HarSessionSummary(
        action_label=action_label,
        threshold_pct=threshold_pct,
        trigger_pct=trigger_pct,
        mean_pct=mean_pct,
        peak_pct=peak_pct,
        qualifying_count=count,
        har_logs=har_logs,
        telegram_text=telegram_text,
    )


def _build_detect_context(log: HarActivityLog, har_logs: str, *, severity: str = "info") -> IndustrialContext:
    occurred = log.occurred_at or datetime.now(timezone.utc)
    if occurred.tzinfo is None:
        occurred = occurred.replace(tzinfo=timezone.utc)
    snap = log.snapshot_url or ""
    return IndustrialContext(
        site_id="site-01",
        line_id=log.camera_id,
        camera_id=log.camera_id,
        timestamp=occurred.isoformat(),
        actor=Actor(
            type=log.actor_type or "operator",
            track_id=str(log.actor_track_id) if log.actor_track_id else None,
            name=log.actor_name,
        ),
        evidence=Evidence(snapshot_path=snap if snap else None),
        links=Links(
            live_url="http://localhost:3000/live",
            timeline_url="http://localhost:3000/timeline",
        ),
        severity=severity,
    )


def _template_values(
    ctx: IndustrialContext,
    log: HarActivityLog,
    summary: HarSessionSummary,
) -> dict[str, str]:
    values = build_template_vars(ctx, footer_reason="HAR live action notifications", snapshot_url="")
    values.update(
        {
            "action_label": summary.action_label,
            "confidence_pct": str(summary.trigger_pct),
            "mean_confidence_pct": str(summary.mean_pct),
            "peak_confidence_pct": str(summary.peak_pct),
            "qualifying_count": str(summary.qualifying_count),
            "threshold_pct": str(summary.threshold_pct),
            "model_id": log.model_id or "—",
            "har_logs": summary.har_logs,
        }
    )
    return values


def _dispatch_har_detect_notifications(
    db: Session,
    *,
    ctx: IndustrialContext,
    log: HarActivityLog,
    summary: HarSessionSummary,
    rule: AlertRule,
    event_id: str,
    event_severity: str = "info",
) -> None:
    ensure_builtin_templates(db)
    template_row = db.get(EmailTemplateRecord, HAR_DETECT_TEMPLATE_ID)
    if template_row is None:
        logger.warning("HAR detect template missing; skipping notifications")
        return

    ctx_enriched, snapshot_asset = enrich_context(ctx)
    values = _template_values(ctx_enriched, log, summary)
    if snapshot_asset is not None:
        from vision_ops_alerting.services.alert_snapshot import snapshot_url_for_email

        values["snapshot_url"] = snapshot_url_for_email(snapshot_asset)

    from vision_ops_alerting.email_layout import build_email_html, build_email_text, substitute

    subject = substitute(template_row.subject, values)
    text = build_email_text(
        category=template_row.category,
        headline=template_row.headline,
        body=template_row.body,
        values=values,
    )
    text += f"\n\n--- Summary ---\n{summary.har_logs}\n"
    html = build_email_html(
        slug=template_row.slug,
        category=template_row.category,
        headline=template_row.headline,
        body=template_row.body,
        severity_level=event_severity if event_severity in ("warning", "critical", "info") else template_row.severity_level,
        values=values,
        layout=template_row.layout,
    )

    email_ids: list[str] = []
    email_dry = settings.dry_run
    email_err: str | None = None
    if rule.notify_email:
        try:
            if not settings.dry_run:
                if not settings.mailersend_api_token:
                    raise RuntimeError("Missing VISIONOPS_MAILERSEND_API_TOKEN")
                if not settings.to_emails:
                    raise RuntimeError("Missing VISIONOPS_TO_EMAIL")
                from mailersend import EmailBuilder, MailerSendClient
                from vision_ops_alerting.agent import _extract_message_id

                ms = MailerSendClient(api_key=settings.mailersend_api_token)
                for recipient in settings.to_recipients():
                    builder = (
                        EmailBuilder()
                        .from_email(settings.from_email, settings.from_name)
                        .to_many([recipient])
                        .subject(subject)
                        .html(html)
                        .text(text)
                    )
                    if snapshot_asset is not None and snapshot_asset.local_path.is_file():
                        builder = builder.attach_file(
                            snapshot_asset.local_path,
                            filename="visionops-har-action.jpg",
                            disposition="attachment",
                        )
                    response = ms.emails.send(builder.build())
                    mid = _extract_message_id(response)
                    if mid:
                        email_ids.append(mid)
                email_dry = False
        except Exception as exc:
            email_err = str(exc)[:500]
            logger.warning("HAR detect email failed: %s", exc)

        log_email_delivery(
            db,
            event_id=event_id,
            template_id=HAR_DETECT_TEMPLATE_ID,
            to_emails=settings.to_emails or ["dry-run@visionops.local"],
            message_ids=email_ids,
            dry_run=email_dry,
            error_message=email_err,
        )

    tg_ids: list[str] = []
    tg_dry = settings.dry_run
    tg_err: str | None = None
    if rule.notify_telegram:
        try:
            caption = summary.telegram_text[:1024]
            if not settings.dry_run:
                if not settings.telegram_bot_token:
                    raise RuntimeError("Missing VISIONOPS_TELEGRAM_BOT_TOKEN")
                if not settings.telegram_chat_id_list:
                    raise RuntimeError("Missing VISIONOPS_TELEGRAM_CHAT_IDS")
                import httpx
                from pathlib import Path
                from vision_ops_alerting.services.telegram_notify import _api_base, _extract_message_id

                with httpx.Client(timeout=60) as client:
                    for chat_id in settings.telegram_chat_id_list:
                        has_photo = snapshot_asset and snapshot_asset.local_path.is_file()
                        if has_photo:
                            with Path(snapshot_asset.local_path).open("rb") as photo:
                                resp = client.post(
                                    f"{_api_base()}/sendPhoto",
                                    data={"chat_id": chat_id, "caption": caption},
                                    files={"photo": photo},
                                )
                                mid = _extract_message_id(resp.json())
                                if mid:
                                    tg_ids.append(mid)
                        else:
                            resp = client.post(
                                f"{_api_base()}/sendMessage",
                                json={"chat_id": chat_id, "text": caption[:4000]},
                                timeout=30,
                            )
                            mid = _extract_message_id(resp.json())
                            if mid:
                                tg_ids.append(mid)
                tg_dry = False
        except Exception as exc:
            tg_err = str(exc)[:500]
            logger.warning("HAR detect telegram failed: %s", exc)

        log_telegram_delivery(
            db,
            event_id=event_id,
            template_id=HAR_DETECT_TEMPLATE_ID,
            chat_ids=settings.telegram_chat_id_list,
            message_ids=tg_ids,
            dry_run=tg_dry,
            error_message=tg_err,
        )


def maybe_notify_high_confidence(db: Session, log: HarActivityLog) -> Event | None:
    """Info email + Telegram when live HAR confidence meets threshold."""
    if log.promoted_to_event_id:
        return None
    if not settings.har_notify_enabled:
        return None

    label = log.predicted_label or ""
    if not label:
        return None

    rule = find_har_rule_for_label(db, label)
    if rule is None:
        return None

    if not rule_has_active_channel(rule):
        return None

    conf = float(log.confidence or 0)
    threshold = rule_confidence_threshold(rule, settings.har_notify_confidence_threshold)
    if conf < threshold:
        return None

    event_severity = rule_event_severity(rule)

    if _recent_event(
        db,
        case_type=HAR_DETECT_CASE_TYPE,
        camera_id=log.camera_id,
        label=label,
        cooldown_sec=settings.har_notify_cooldown_sec,
    ):
        return None

    pct = int(round(conf * 100))
    summary = _summarize_session_for_action(
        db, log, action_label=label, threshold=threshold,
    )
    ctx = _build_detect_context(log, summary.har_logs, severity=event_severity)

    top_k = []
    if log.top_k_json:
        try:
            top_k = json.loads(log.top_k_json)
        except json.JSONDecodeError:
            pass

    title = f"{log.camera_id}: {label} ({pct}%)"
    description = (
        f"Live HAR «{label}» on {log.camera_id} — trigger {pct}%, "
        f"mean {summary.mean_pct}% ({summary.qualifying_count} reads ≥{summary.threshold_pct}%)."
    )

    meta = [
        {"icon": "smart_toy", "text": "HAR live"},
        {"icon": "videocam", "text": log.camera_id},
        {"icon": "info", "text": f"{event_severity} notification"},
    ]
    if log.actor_name:
        meta.insert(0, {"icon": "person", "text": log.actor_name})

    thumb = _resolve_har_snapshot_thumb(log.snapshot_url)

    event = Event(
        id=new_id("evt"),
        rule_id=rule.id,
        site_id="site-01",
        line_id=log.camera_id,
        camera_id=log.camera_id,
        case_type=HAR_DETECT_CASE_TYPE,
        severity=event_severity,
        title=title,
        description=description,
        actor_type=log.actor_type,
        actor_track_id=log.actor_track_id,
        actor_name=log.actor_name,
        evidence_json=json.dumps(
            {
                "label": label,
                "confidence": conf,
                "top_k": top_k,
                "harLogs": summary.har_logs,
                "meanConfidence": summary.mean_pct,
                "peakConfidence": summary.peak_pct,
                "qualifyingCount": summary.qualifying_count,
            },
            ensure_ascii=False,
        ),
        context_json=json.dumps(
            {"harLogId": log.id, "modelId": log.model_id, "sessionId": log.session_id},
            ensure_ascii=False,
        ),
        meta_json=json.dumps(meta, ensure_ascii=False),
        thumbnail_url=thumb,
        clip_url=log.clip_url,
        occurred_at=log.occurred_at or datetime.now(timezone.utc),
        resolution_status="OPEN",
    )
    db.add(event)
    db.flush()

    log.promoted_to_event_id = event.id
    db.flush()

    _dispatch_har_detect_notifications(
        db,
        ctx=ctx,
        log=log,
        summary=summary,
        rule=rule,
        event_id=event.id,
        event_severity=event_severity,
    )
    return event


def maybe_promote_log(db: Session, log: HarActivityLog) -> Event | None:
    """Create Timeline event for notable deviation logs (non-assembly / very low confidence)."""
    if log.promoted_to_event_id:
        return None

    conf = float(log.confidence or 0)
    label = log.predicted_label or "unknown"
    har_rule = find_har_rule_for_label(db, label) if label != "unknown" else None
    notify_threshold = (
        rule_confidence_threshold(har_rule, settings.har_notify_confidence_threshold)
        if har_rule
        else settings.har_notify_confidence_threshold
    )
    if conf >= notify_threshold and settings.har_notify_enabled and har_rule is not None:
        return None
    pct = int(round(conf * 100))

    promote = False
    severity = "info"

    if settings.har_promote_non_assembly and not log.is_primary_action:
        promote = True
        if conf < HAR_CRITICAL_CONFIDENCE:
            severity = "critical"
        else:
            severity = "warning"

    if conf < settings.har_low_confidence_threshold and label:
        promote = True
        if severity != "warning":
            severity = "info"

    if not promote:
        return None

    if _recent_event(
        db,
        case_type=HAR_CASE_TYPE,
        camera_id=log.camera_id,
        label=label,
        cooldown_sec=settings.har_promote_cooldown_sec,
    ):
        return None

    rule = _ensure_har_rule(db)
    camera_name = log.camera_id
    title = f"{camera_name}: {label} ({pct}%)"
    description = (
        f"HAR detected «{label}» at {pct}% confidence on {log.camera_id} "
        f"(model {log.model_id}, source {log.source})."
    )
    if not log.is_primary_action:
        description += f" Expected primary action: {PRIMARY_ACTION}."

    top_k = []
    if log.top_k_json:
        try:
            top_k = json.loads(log.top_k_json)
        except json.JSONDecodeError:
            pass
    detections = []
    if log.detections_json:
        try:
            detections = json.loads(log.detections_json)
        except json.JSONDecodeError:
            pass

    meta = [
        {"icon": "smart_toy", "text": "HAR activity"},
        {"icon": "videocam", "text": log.camera_id},
    ]
    if log.actor_name:
        meta.insert(0, {"icon": "person", "text": log.actor_name})

    clip_url = None
    if log.session_id:
        from vision_ops_alerting.db.models import HarWatchSession

        sess = db.query(HarWatchSession).filter(HarWatchSession.id == log.session_id).first()
        if sess:
            clip_url = sess.clip_url

    event = Event(
        id=new_id("evt"),
        rule_id=rule.id if rule else None,
        site_id="site-01",
        line_id=log.camera_id,
        camera_id=log.camera_id,
        case_type=HAR_CASE_TYPE,
        severity=severity,
        title=title,
        description=description,
        actor_type=log.actor_type,
        actor_track_id=log.actor_track_id,
        actor_name=log.actor_name,
        evidence_json=json.dumps(
            {"top_k": top_k, "detections": detections, "confidence": conf, "label": label},
            ensure_ascii=False,
        ),
        context_json=json.dumps({"harLogId": log.id, "modelId": log.model_id}, ensure_ascii=False),
        meta_json=json.dumps(
            meta + [{"icon": "source", "text": "har"}],
            ensure_ascii=False,
        ),
        clip_url=clip_url,
        thumbnail_url=_resolve_har_snapshot_thumb(log.snapshot_url),
        occurred_at=log.occurred_at or datetime.now(timezone.utc),
        resolution_status="OPEN",
    )
    db.add(event)
    db.flush()

    log.promoted_to_event_id = event.id
    db.flush()

    dry_run = True
    if not settings.har_email_enabled:
        dry_run = True
    elif settings.dry_run:
        dry_run = True

    log_email_delivery(
        db,
        event_id=event.id,
        template_id=HAR_TEMPLATE_ID,
        to_emails=settings.to_emails or ["dry-run@visionops.local"],
        message_ids=[],
        dry_run=dry_run,
        error_message=None if dry_run else "HAR email disabled",
    )
    return event


def dispatch_after_ingest(db: Session, logs: list[HarActivityLog]) -> list[str]:
    """Promote ingested logs and dispatch live-action notifications."""
    event_ids: list[str] = []
    for log in logs:
        ev = maybe_notify_high_confidence(db, log)
        if ev:
            event_ids.append(ev.id)
            continue
        ev = maybe_promote_log(db, log)
        if ev:
            event_ids.append(ev.id)
    if event_ids:
        db.commit()
    return event_ids


@dataclass
class _TestHarLog:
    predicted_label: str
    confidence: float
    camera_id: str = "cam-har-01"
    model_id: str = "har-test"
    actor_type: str | None = "operator"
    actor_track_id: int | None = 1
    actor_name: str | None = "Test Operator"
    snapshot_url: str | None = None
    clip_url: str | None = None
    session_id: str | None = None
    top_k_json: str | None = None
    occurred_at: datetime | None = None


def dispatch_har_action_test(
    db: Session,
    *,
    har_action_label: str,
    channel: str = "both",
) -> Event:
    """Send a test email/Telegram using the per-action alert rule configuration."""
    from vision_ops_alerting.services.rule_dispatch import require_enabled_har_rule

    rule = require_enabled_har_rule(db, har_action_label)
    if channel == "email" and not rule.notify_email:
        raise ValueError(f"Email is off for «{har_action_label}». Enable it on the alert rule.")
    if channel == "telegram" and not rule.notify_telegram:
        raise ValueError(f"Telegram is off for «{har_action_label}». Enable it on the alert rule.")

    conf = rule_confidence_threshold(rule, settings.har_notify_confidence_threshold)
    pct = int(round(conf * 100))
    event_severity = rule_event_severity(rule)
    log = _TestHarLog(
        predicted_label=har_action_label,
        confidence=conf,
        occurred_at=datetime.now(timezone.utc),
    )
    summary = _summarize_session_for_action(
        db, log, action_label=har_action_label, threshold=conf,
    )
    ctx = _build_detect_context(log, summary.har_logs, severity=event_severity)

    title = f"TEST: {log.camera_id}: {har_action_label} ({pct}%)"
    description = f"Test notification for HAR action «{har_action_label}» (threshold {pct}%)."

    event = Event(
        id=new_id("evt"),
        rule_id=rule.id,
        site_id="site-01",
        line_id=log.camera_id,
        camera_id=log.camera_id,
        case_type=HAR_DETECT_CASE_TYPE,
        severity=event_severity,
        title=title,
        description=description,
        actor_type=log.actor_type,
        actor_track_id=log.actor_track_id,
        actor_name=log.actor_name,
        evidence_json=json.dumps({"label": har_action_label, "confidence": conf, "harLogs": summary.har_logs}),
        context_json=json.dumps({"test": True, "harActionLabel": har_action_label}),
        meta_json=json.dumps([
            {"icon": "smart_toy", "text": "HAR test"},
            {"icon": "info", "text": f"{event_severity} notification"},
        ]),
        occurred_at=datetime.now(timezone.utc),
        resolution_status="OPEN",
    )
    db.add(event)
    db.flush()

    email_on = rule.notify_email and channel in ("both", "email")
    telegram_on = rule.notify_telegram and channel in ("both", "telegram")
    saved_email, saved_telegram = rule.notify_email, rule.notify_telegram
    rule.notify_email = email_on
    rule.notify_telegram = telegram_on
    try:
        _dispatch_har_detect_notifications(
            db,
            ctx=ctx,
            log=log,
            summary=summary,
            rule=rule,
            event_id=event.id,
            event_severity=event_severity,
        )
    finally:
        rule.notify_email = saved_email
        rule.notify_telegram = saved_telegram

    return event
