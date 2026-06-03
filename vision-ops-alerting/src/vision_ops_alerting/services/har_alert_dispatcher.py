"""Promote notable HAR activity logs to Timeline events (email always dry-run)."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

from sqlalchemy.orm import Session

from vision_ops_alerting.config import settings
from vision_ops_alerting.db.models import AlertRule, Event, HarActivityLog, new_id
from vision_ops_alerting.services.events import log_email_delivery
from vision_ops_alerting.services.har_activity_store import PRIMARY_ACTION, HAR_CRITICAL_CONFIDENCE

HAR_CASE_TYPE = "har_action_deviation"
HAR_TEMPLATE_ID = "har.action_deviation"


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


def _recent_promoted(
    db: Session,
    *,
    camera_id: str,
    label: str,
    cooldown_sec: int,
) -> bool:
    since = datetime.now(timezone.utc) - timedelta(seconds=cooldown_sec)
    recent = (
        db.query(Event)
        .filter(
            Event.camera_id == camera_id,
            Event.case_type == HAR_CASE_TYPE,
            Event.occurred_at >= since,
            Event.title.contains(label),
        )
        .first()
    )
    return recent is not None


def maybe_promote_log(db: Session, log: HarActivityLog) -> Event | None:
    """Create Timeline event for notable HAR logs; never send real email."""
    if log.promoted_to_event_id:
        return None

    label = log.predicted_label or "unknown"
    conf = float(log.confidence or 0)
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

    if _recent_promoted(
        db,
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
    """Promote ingested logs; returns created event ids."""
    event_ids: list[str] = []
    for log in logs:
        ev = maybe_promote_log(db, log)
        if ev:
            event_ids.append(ev.id)
    if event_ids:
        db.commit()
    return event_ids
