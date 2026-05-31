from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from vision_ops_alerting.agent import send_email_for_case_type
from vision_ops_alerting.db.session import get_db
from vision_ops_alerting.schemas import CaseType, EmailSendResponse, EmailSent, IndustrialContext, Actor, Evidence, Links
from vision_ops_alerting.config import settings
from vision_ops_alerting.services.events import create_event_from_context, log_email_delivery
from vision_ops_alerting.services.email_templates import resolve_template
from vision_ops_alerting.services.rule_dispatch import RuleNotEnabledError, require_enabled_rule
from vision_ops_alerting.services.telemetry import collect_and_build, email_notification_status

router = APIRouter(tags=["notifications"])


TEST_CONTEXTS: dict[CaseType, IndustrialContext] = {
    "user_not_working": IndustrialContext(
        site_id="site-01",
        line_id="line-a",
        camera_id="cam-01",
        timestamp=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        actor=Actor(type="operator", track_id="12", name="Operator 12"),
        evidence=Evidence(idle_seconds=900),
        links=Links(live_url="http://localhost:3000/live", timeline_url="http://localhost:3000/timeline"),
        case_type="user_not_working",
        severity="warning",
    ),
    "user_left_position": IndustrialContext(
        site_id="site-01",
        line_id="line-b",
        camera_id="cam-02",
        timestamp=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        actor=Actor(type="operator", track_id="7", name="Operator 7"),
        evidence=Evidence(roi="station-3"),
        links=Links(live_url="http://localhost:3000/live", timeline_url="http://localhost:3000/timeline"),
        case_type="user_left_position",
        severity="critical",
    ),
    "forklift_in_zone": IndustrialContext(
        site_id="site-01",
        line_id="line-c",
        camera_id="cam-02",
        timestamp=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        actor=Actor(type="forklift", track_id="FL-02", name="Forklift FL-02"),
        evidence=Evidence(roi="restricted-zone-b"),
        links=Links(live_url="http://localhost:3000/live", timeline_url="http://localhost:3000/timeline"),
        case_type="forklift_in_zone",
        severity="critical",
    ),
    "unknown": IndustrialContext(
        site_id="site-01",
        line_id="line-a",
        camera_id="cam-01",
        timestamp=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        actor=Actor(type="unknown"),
        evidence=Evidence(),
        links=Links(live_url="http://localhost:3000/live", timeline_url="http://localhost:3000/timeline"),
        case_type="unknown",
        severity="info",
    ),
}


@router.get("/api/notifications/email/status")
def email_status():
    return email_notification_status()


@router.post("/api/alerting/email/test/{case_type}", response_model=EmailSendResponse)
def send_test_email(case_type: CaseType, db: Session = Depends(get_db)):
    if case_type not in TEST_CONTEXTS:
        raise HTTPException(status_code=400, detail="Unknown case type")
    ctx = TEST_CONTEXTS[case_type]
    try:
        rule = require_enabled_rule(db, case_type)
        db_template = resolve_template(db, case_type, rule.email_template_id)
        classified, message_ids, dry_run = send_email_for_case_type(
            ctx, case_type, db_template=db_template
        )
        if not rule.notify_email:
            dry_run = True
            message_ids = []
        event = create_event_from_context(
            db,
            ctx,
            case_type=classified.case_type,
            severity=classified.severity,
            rule_id=rule.id,
        )
        delivery = log_email_delivery(
            db,
            event_id=event.id,
            template_id=classified.template_id,
            to_emails=settings.to_emails,
            message_ids=message_ids,
            dry_run=dry_run,
        )
        db.commit()
        return EmailSendResponse(
            ok=True,
            event=EmailSent(
                event_id=event.id,
                delivery_id=delivery.id,
                template_id=classified.template_id,
                case_type=classified.case_type,
                severity=classified.severity,
                to_emails=settings.to_emails,
                message_ids=message_ids,
                dry_run=dry_run,
            ),
        )
    except RuleNotEnabledError as e:
        db.rollback()
        raise HTTPException(status_code=403, detail=str(e)) from e
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=400, detail=str(e)) from e


@router.get("/api/telemetry")
async def get_telemetry(db: Session = Depends(get_db)):
    return await collect_and_build(db)
