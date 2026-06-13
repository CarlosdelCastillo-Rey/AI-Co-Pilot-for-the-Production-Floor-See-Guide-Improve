from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from vision_ops_alerting.db.models import Event
from vision_ops_alerting.db.session import get_db
from vision_ops_alerting.schemas import CaseType, EmailSendResponse, IndustrialContext, Actor, Evidence, Links
from vision_ops_alerting.services.email_templates import resolve_template
from vision_ops_alerting.services.events import event_to_timeline_dict
from vision_ops_alerting.services.notification_dispatch import dispatch_test_alert, email_sent_from_result
from vision_ops_alerting.services.rule_dispatch import RuleNotEnabledError, require_enabled_rule
from vision_ops_alerting.services.telemetry import collect_and_build, email_notification_status
from vision_ops_alerting.services.telegram_notify import telegram_notification_status

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


@router.get("/api/notifications/telegram/status")
def telegram_status():
    return telegram_notification_status()


@router.get("/api/notifications/recent")
def recent_notifications(db: Session = Depends(get_db), limit: int = 12):
    """Recent timeline events (incl. HAR deviations) for in-app feedback."""
    rows = (
        db.query(Event)
        .filter(Event.hidden_from_panel.is_(False))
        .order_by(Event.occurred_at.desc())
        .limit(max(1, min(limit, 50)))
        .all()
    )
    return {"events": [event_to_timeline_dict(e) for e in rows]}


def _send_test_alert(case_type: CaseType, db: Session) -> EmailSendResponse:
    if case_type not in TEST_CONTEXTS:
        raise HTTPException(status_code=400, detail="Unknown case type")
    ctx = TEST_CONTEXTS[case_type]
    try:
        rule = require_enabled_rule(db, case_type)
        db_template = resolve_template(db, case_type, rule.email_template_id)
        result = dispatch_test_alert(db, ctx, case_type, rule, db_template)
        db.commit()
        return EmailSendResponse(ok=True, event=email_sent_from_result(result))
    except RuleNotEnabledError as e:
        db.rollback()
        raise HTTPException(status_code=403, detail=str(e)) from e
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=400, detail=str(e)) from e


@router.post("/api/alerting/email/test/{case_type}", response_model=EmailSendResponse)
def send_test_email(case_type: CaseType, db: Session = Depends(get_db)):
    return _send_test_alert(case_type, db)


@router.post("/api/alerting/telegram/test/{case_type}", response_model=EmailSendResponse)
def send_test_telegram(case_type: CaseType, db: Session = Depends(get_db)):
    return _send_test_alert(case_type, db)


@router.get("/api/telemetry")
async def get_telemetry(db: Session = Depends(get_db)):
    return await collect_and_build(db)

from pydantic import BaseModel, Field

from vision_ops_alerting.services.har_alert_dispatcher import dispatch_har_action_test


class HarActionTestRequest(BaseModel):
    har_action_label: str = Field(alias="harActionLabel")

    model_config = {"populate_by_name": True}


class HarTestResponse(BaseModel):
    ok: bool = True
    event_id: str = Field(alias="eventId")
    har_action_label: str = Field(alias="harActionLabel")
    message: str

    model_config = {"populate_by_name": True}


def _har_test_response(event, label: str, channel: str) -> HarTestResponse:
    return HarTestResponse(
        eventId=event.id,
        harActionLabel=label,
        message=f"Test {channel} sent for «{label}» (timeline event {event.id}).",
    )


@router.post("/api/alerting/har/test-email", response_model=HarTestResponse)
def har_test_email(body: HarActionTestRequest, db: Session = Depends(get_db)):
    try:
        event = dispatch_har_action_test(db, har_action_label=body.har_action_label, channel="email")
        db.commit()
        return _har_test_response(event, body.har_action_label, "email")
    except RuleNotEnabledError as e:
        db.rollback()
        raise HTTPException(status_code=403, detail=str(e)) from e
    except ValueError as e:
        db.rollback()
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=400, detail=str(e)) from e


@router.post("/api/alerting/har/test-telegram", response_model=HarTestResponse)
def har_test_telegram(body: HarActionTestRequest, db: Session = Depends(get_db)):
    try:
        event = dispatch_har_action_test(db, har_action_label=body.har_action_label, channel="telegram")
        db.commit()
        return _har_test_response(event, body.har_action_label, "Telegram")
    except RuleNotEnabledError as e:
        db.rollback()
        raise HTTPException(status_code=403, detail=str(e)) from e
    except ValueError as e:
        db.rollback()
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=400, detail=str(e)) from e

