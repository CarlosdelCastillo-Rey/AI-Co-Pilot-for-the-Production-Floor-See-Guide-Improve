from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session

from vision_ops_alerting.agent import classify_case, dispatch_classified_email
from vision_ops_alerting.config import settings
from vision_ops_alerting.db.session import get_db, init_db
from vision_ops_alerting.routers import admin, advisor, alerts, analytics, auth, cameras, email_templates, har, notifications, timeline
from vision_ops_alerting.routers import settings as settings_router
from vision_ops_alerting.schemas import EmailSendResponse, EmailSent, IndustrialContext, Severity
from vision_ops_alerting.services.events import (
    create_event_from_context,
    log_email_delivery,
)
from vision_ops_alerting.services.email_templates import ensure_builtin_templates, resolve_template


from vision_ops_alerting.services.rule_dispatch import RuleNotEnabledError, require_enabled_rule


def _severity_from_rule(rule_severity: str, fallback: Severity) -> Severity:
    mapping = {"CRITICAL": "critical", "WARNING": "warning"}
    return mapping.get(rule_severity, fallback)  # type: ignore[return-value]


@asynccontextmanager
async def lifespan(_app: FastAPI):
    init_db()
    yield


app = FastAPI(
    title="VisionOps Alerting",
    description="Strands-based alert classification, email notifications, and event persistence.",
    version="0.2.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in settings.cors_origins.split(",") if o.strip()],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(admin.router)
app.include_router(auth.router)
app.include_router(alerts.router)
app.include_router(email_templates.router)
app.include_router(cameras.router)
app.include_router(har.router)
app.include_router(timeline.router)
app.include_router(analytics.router)
app.include_router(notifications.router)
app.include_router(settings_router.router)
app.include_router(advisor.router)


@app.get("/health")
def health():
    return {"ok": True, "service": "vision-ops-alerting"}


@app.post("/api/alerting/email", response_model=EmailSendResponse)
def send_email(ctx: IndustrialContext, db: Session = Depends(get_db)):
    try:
        classified = classify_case(ctx)
        rule = require_enabled_rule(db, classified.case_type)
        severity = _severity_from_rule(rule.severity, classified.severity)

        db_template = resolve_template(db, classified.case_type, rule.email_template_id)

        message_ids: list[str] = []
        dry_run = settings.dry_run
        template_id = classified.template_id
        if rule.notify_email:
            message_ids, dry_run, rendered = dispatch_classified_email(
                ctx, classified, db_template=db_template
            )
            template_id = rendered.template_id

        event = create_event_from_context(
            db,
            ctx,
            case_type=classified.case_type,
            severity=severity,
            rule_id=rule.id,
        )

        delivery_id = ""
        if rule.notify_email:
            delivery = log_email_delivery(
                db,
                event_id=event.id,
                template_id=template_id,
                to_emails=settings.to_emails,
                message_ids=message_ids,
                dry_run=dry_run,
            )
            delivery_id = delivery.id

        db.commit()

        return EmailSendResponse(
            ok=True,
            event=EmailSent(
                event_id=event.id,
                delivery_id=delivery_id,
                template_id=template_id,
                case_type=classified.case_type,
                severity=severity,
                to_emails=settings.to_emails if rule.notify_email else [],
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
