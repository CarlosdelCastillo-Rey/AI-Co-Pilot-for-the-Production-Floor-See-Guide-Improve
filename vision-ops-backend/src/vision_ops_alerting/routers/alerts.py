from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from vision_ops_alerting.auth_deps import get_current_user
from vision_ops_alerting.db.models import AlertRule, User, new_id
from vision_ops_alerting.db.session import get_db
from vision_ops_alerting.services.alert_actions import (
    HAR_CASE_TYPE,
    get_action,
    get_action_for_har_label,
    list_actions,
)
from vision_ops_alerting.services.rule_config import (
    config_json_for_har_rule,
    parse_rule_config,
    rule_har_action_label,
)

from vision_ops_alerting.services.events import delivery_to_dict, rule_to_dict

router = APIRouter(prefix="/api/alerts", tags=["alerts"])


@router.get("/actions")
def list_alert_actions():
    return list_actions()


class AlertRuleCreate(BaseModel):
    icon: str = "warning"
    title: str
    description: str
    zone: str
    case_type: str = Field(default="unknown", alias="caseType")
    severity: str = "WARNING"
    enabled: bool = True
    notify_email: bool = Field(default=True, alias="notifyEmail")
    notify_telegram: bool = Field(default=False, alias="notifyTelegram")
    email_template_id: str | None = Field(default=None, alias="emailTemplateId")
    confidence_threshold: float | None = Field(default=None, alias="confidenceThreshold")
    har_action_label: str | None = Field(default=None, alias="harActionLabel")
    notify_in_app: bool | None = Field(default=None, alias="notifyInApp")

    model_config = {"populate_by_name": True}


class AlertRuleUpdate(BaseModel):
    icon: str | None = None
    title: str | None = None
    description: str | None = None
    zone: str | None = None
    case_type: str | None = Field(default=None, alias="caseType")
    severity: str | None = None
    enabled: bool | None = None
    notify_email: bool | None = Field(default=None, alias="notifyEmail")
    notify_telegram: bool | None = Field(default=None, alias="notifyTelegram")
    email_template_id: str | None = Field(default=None, alias="emailTemplateId")
    confidence_threshold: float | None = Field(default=None, alias="confidenceThreshold")
    har_action_label: str | None = Field(default=None, alias="harActionLabel")
    notify_in_app: bool | None = Field(default=None, alias="notifyInApp")

    model_config = {"populate_by_name": True}


@router.get("/rules")
def list_rules(db: Session = Depends(get_db)):
    rules = db.query(AlertRule).order_by(AlertRule.created_at.asc()).all()
    return [rule_to_dict(r) for r in rules]


@router.post("/rules", status_code=201)
def create_rule(body: AlertRuleCreate, db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    from vision_ops_alerting.config import settings

    if body.case_type != HAR_CASE_TYPE:
        raise HTTPException(status_code=400, detail=f"Unsupported case type: {body.case_type}")
    if not body.har_action_label:
        raise HTTPException(status_code=400, detail="harActionLabel is required for HAR alerts")
    action = get_action_for_har_label(body.har_action_label)
    if not action:
        raise HTTPException(status_code=400, detail=f"Unknown HAR action: {body.har_action_label}")

    for existing in db.query(AlertRule).filter(AlertRule.case_type == HAR_CASE_TYPE).all():
        if rule_har_action_label(existing) == body.har_action_label:
            raise HTTPException(status_code=409, detail=f"Rule already exists for “{body.har_action_label}”")

    threshold = (
        body.confidence_threshold
        if body.confidence_threshold is not None
        else settings.har_notify_confidence_threshold
    )
    severity = body.severity or action.default_severity
    config_json = config_json_for_har_rule(
        har_action_label=body.har_action_label,
        confidence_threshold=threshold,
        event_severity=severity,
        notify_in_app=body.notify_in_app if body.notify_in_app is not None else False,
    )

    rule = AlertRule(
        id=new_id("rule"),
        icon=body.icon or action.icon,
        title=body.title,
        description=body.description,
        zone=body.zone,
        case_type=HAR_CASE_TYPE,
        severity=severity,
        enabled=body.enabled,
        notify_email=body.notify_email,
        notify_telegram=body.notify_telegram,
        email_template_id=body.email_template_id or "har.action_detected",
        config_json=config_json,
        updated_by=user.name,
    )
    db.add(rule)
    db.commit()
    db.refresh(rule)
    return rule_to_dict(rule)


@router.get("/rules/{rule_id}")
def get_rule(rule_id: str, db: Session = Depends(get_db)):
    rule = db.get(AlertRule, rule_id)
    if not rule:
        raise HTTPException(status_code=404, detail="Rule not found")
    return rule_to_dict(rule)


@router.patch("/rules/{rule_id}")
def update_rule(
    rule_id: str,
    body: AlertRuleUpdate,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    rule = db.get(AlertRule, rule_id)
    if not rule:
        raise HTTPException(status_code=404, detail="Rule not found")

    for field, attr in [
        (body.icon, "icon"),
        (body.title, "title"),
        (body.description, "description"),
        (body.zone, "zone"),
        (body.case_type, "case_type"),
        (body.severity, "severity"),
        (body.enabled, "enabled"),
        (body.notify_email, "notify_email"),
        (body.notify_telegram, "notify_telegram"),
        (body.email_template_id, "email_template_id"),
    ]:
        if field is not None:
            setattr(rule, attr, field)

    if any(
        x is not None
        for x in (body.confidence_threshold, body.har_action_label, body.severity, body.notify_in_app)
    ):
        rule.config_json = config_json_for_har_rule(
            har_action_label=body.har_action_label,
            confidence_threshold=body.confidence_threshold,
            event_severity=body.severity,
            notify_in_app=body.notify_in_app,
            existing=rule,
        )

    rule.updated_at = datetime.now(timezone.utc)
    rule.updated_by = user.name
    db.commit()
    db.refresh(rule)
    return rule_to_dict(rule)


@router.delete("/rules/{rule_id}", status_code=204)
def delete_rule(rule_id: str, db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    rule = db.get(AlertRule, rule_id)
    if not rule:
        raise HTTPException(status_code=404, detail="Rule not found")
    db.delete(rule)
    db.commit()


@router.post("/rules/{rule_id}/toggle")
def toggle_rule(rule_id: str, db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    import json

    rule = db.get(AlertRule, rule_id)
    if not rule:
        raise HTTPException(status_code=404, detail="Rule not found")
    rule.enabled = not rule.enabled
    cfg = parse_rule_config(rule)
    if not rule.enabled:
        if rule.severity != "DISABLED":
            cfg["savedSeverity"] = rule.severity
            rule.config_json = json.dumps(cfg)
            rule.severity = "DISABLED"
    elif rule.severity == "DISABLED":
        restored = cfg.pop("savedSeverity", None)
        rule.severity = restored or "INFO"
        rule.config_json = json.dumps(cfg) if cfg else rule.config_json
    rule.updated_at = datetime.now(timezone.utc)
    rule.updated_by = user.name
    db.commit()
    db.refresh(rule)
    return rule_to_dict(rule)


@router.get("/deliveries")
def list_deliveries(limit: int = 50, db: Session = Depends(get_db)):
    from vision_ops_alerting.db.models import AlertDelivery

    rows = db.query(AlertDelivery).order_by(AlertDelivery.sent_at.desc()).limit(limit).all()
    return [delivery_to_dict(d) for d in rows]
