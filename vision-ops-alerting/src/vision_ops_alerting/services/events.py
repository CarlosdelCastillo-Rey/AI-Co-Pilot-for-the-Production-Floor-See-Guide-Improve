from __future__ import annotations

import json
from datetime import datetime, timezone

from sqlalchemy.orm import Session

from vision_ops_alerting.db.models import AlertDelivery, AlertRule, Event, new_id
from vision_ops_alerting.schemas import CaseType, IndustrialContext, Severity
from vision_ops_alerting.templates import TEMPLATES


def _format_duration(seconds: int | None) -> str:
    if not seconds:
        return "00:00"
    return f"{seconds // 60:02d}:{seconds % 60:02d}"


def _case_title(case_type: CaseType) -> str:
    titles = {
        "user_not_working": "Operator Idle Detected",
        "user_left_position": "Operator Left Position",
        "forklift_in_zone": "Forklift in Restricted Zone",
        "unknown": "VisionOps Alert",
    }
    return titles.get(case_type, "VisionOps Alert")


def _case_description(ctx: IndustrialContext, case_type: CaseType) -> str:
    actor = ctx.actor.name or ctx.actor.track_id or "unknown"
    if case_type == "user_not_working":
        idle = ctx.evidence.idle_seconds or 0
        return f"Operator {actor} idle for {idle}s on {ctx.line_id} / {ctx.camera_id}."
    if case_type == "user_left_position":
        return f"Operator {actor} left assigned position on {ctx.line_id} / {ctx.camera_id}."
    if case_type == "forklift_in_zone":
        roi = ctx.evidence.roi or "restricted zone"
        return f"Forklift detected in {roi} on {ctx.line_id} / {ctx.camera_id}."
    return f"Alert on {ctx.line_id} / {ctx.camera_id} at {ctx.timestamp}."


def _build_meta(ctx: IndustrialContext) -> list[dict[str, str]]:
    meta: list[dict[str, str]] = []
    if ctx.actor.name or ctx.actor.track_id:
        meta.append({"icon": "person", "text": ctx.actor.name or ctx.actor.track_id or "Unknown"})
    if ctx.line_id or ctx.camera_id:
        meta.append({"icon": "location_on", "text": f"{ctx.line_id} / {ctx.camera_id}"})
    if ctx.camera_id:
        meta.append({"icon": "videocam", "text": ctx.camera_id})
    return meta


def _looks_like_image_url(url: str | None) -> bool:
    if not url or not url.strip():
        return False
    lower = url.lower().split("?")[0]
    if lower.endswith((".jpg", ".jpeg", ".png", ".gif", ".webp", ".avif")):
        return True
    if "googleusercontent.com" in lower or "/artifacts/" in lower or "/api/vision/" in lower:
        return True
    return False


def _thumbnail_from_context(ctx: IndustrialContext) -> str | None:
    snap = ctx.evidence.snapshot_path
    if _looks_like_image_url(snap):
        return snap
    return None


def create_event_from_context(
    db: Session,
    ctx: IndustrialContext,
    *,
    case_type: CaseType,
    severity: Severity,
    rule_id: str | None = None,
) -> Event:
    occurred = datetime.now(timezone.utc)
    try:
        occurred = datetime.fromisoformat(ctx.timestamp.replace("Z", "+00:00"))
    except ValueError:
        pass

    event = Event(
        id=new_id("evt"),
        rule_id=rule_id,
        site_id=ctx.site_id,
        line_id=ctx.line_id,
        camera_id=ctx.camera_id,
        case_type=case_type,
        severity=severity,
        title=_case_title(case_type),
        description=_case_description(ctx, case_type),
        actor_type=ctx.actor.type,
        actor_track_id=ctx.actor.track_id,
        actor_name=ctx.actor.name,
        evidence_json=json.dumps(ctx.evidence.model_dump(mode="json")),
        context_json=json.dumps(ctx.model_dump(mode="json")),
        meta_json=json.dumps(_build_meta(ctx)),
        thumbnail_url=_thumbnail_from_context(ctx),
        clip_url=ctx.links.timeline_url,
        clip_duration_sec=ctx.evidence.idle_seconds,
        occurred_at=occurred,
        resolution_status="OPEN",
    )
    db.add(event)
    db.flush()
    return event


def log_email_delivery(
    db: Session,
    *,
    event_id: str,
    template_id: str,
    to_emails: list[str],
    message_ids: list[str],
    dry_run: bool,
    error_message: str | None = None,
) -> AlertDelivery:
    status = "dry_run" if dry_run else ("failed" if error_message else "sent")
    delivery = AlertDelivery(
        id=new_id("dlv"),
        event_id=event_id,
        channel="email",
        template_id=template_id,
        to_emails_json=json.dumps(to_emails),
        message_ids_json=json.dumps(message_ids) if message_ids else None,
        dry_run=dry_run,
        status=status,
        error_message=error_message,
    )
    db.add(delivery)
    db.flush()
    return delivery


def find_matching_rule(db: Session, case_type: CaseType) -> AlertRule | None:
    return (
        db.query(AlertRule)
        .filter(AlertRule.case_type == case_type, AlertRule.enabled.is_(True))
        .order_by(AlertRule.updated_at.desc())
        .first()
    )


def event_to_timeline_dict(event: Event) -> dict:
    meta = json.loads(event.meta_json) if event.meta_json else []
    har_source = event.case_type == "har_action_deviation" or any(
        isinstance(m, dict) and m.get("text") == "har" for m in meta
    )
    thumbnail = event.thumbnail_url or ""
    if not _looks_like_image_url(thumbnail):
        thumbnail = ""
    return {
        "id": event.id,
        "time": event.occurred_at.strftime("%H:%M:%S"),
        "severity": event.severity,
        "title": event.title,
        "description": event.description,
        "meta": meta,
        "thumbnail": thumbnail,
        "clipDuration": _format_duration(event.clip_duration_sec),
        "clipUrl": event.clip_url or "",
        "cameraId": event.camera_id,
        "caseType": event.case_type,
        "occurredAt": event.occurred_at.isoformat(),
        "resolutionStatus": event.resolution_status or "OPEN",
        "acknowledgedAt": event.acknowledged_at.isoformat() if event.acknowledged_at else None,
        "acknowledgedBy": event.acknowledged_by,
        "resolvedAt": event.resolved_at.isoformat() if event.resolved_at else None,
        "resolvedBy": event.resolved_by,
        "industrialReasonCode": event.industrial_reason_code,
        "downtimeCausedSeconds": event.downtime_caused_seconds or 0,
        "scrapCausedUnits": event.scrap_caused_units or 0,
        "closureNotes": event.closure_notes,
        "hiddenFromPanel": bool(event.hidden_from_panel),
        "harSource": har_source,
    }


def rule_to_dict(rule: AlertRule) -> dict:
    return {
        "id": rule.id,
        "icon": rule.icon,
        "title": rule.title,
        "description": rule.description,
        "zone": rule.zone,
        "caseType": rule.case_type,
        "severity": rule.severity if rule.enabled else "DISABLED",
        "enabled": rule.enabled,
        "notifyEmail": rule.notify_email,
        "emailTemplateId": rule.email_template_id,
        "updatedAt": rule.updated_at.isoformat() if rule.updated_at else None,
        "updatedBy": rule.updated_by,
    }


def delivery_to_dict(delivery: AlertDelivery) -> dict:
    return {
        "id": delivery.id,
        "eventId": delivery.event_id,
        "channel": delivery.channel,
        "templateId": delivery.template_id,
        "toEmails": json.loads(delivery.to_emails_json),
        "messageIds": json.loads(delivery.message_ids_json) if delivery.message_ids_json else [],
        "dryRun": delivery.dry_run,
        "status": delivery.status,
        "errorMessage": delivery.error_message,
        "sentAt": delivery.sent_at.isoformat(),
    }
