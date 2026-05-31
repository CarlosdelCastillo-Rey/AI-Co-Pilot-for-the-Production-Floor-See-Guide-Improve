from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from vision_ops_alerting.auth_deps import get_current_user
from vision_ops_alerting.db.models import AlertRule, User, new_id
from vision_ops_alerting.db.session import get_db
from vision_ops_alerting.services.alert_actions import get_action, list_actions
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
    email_template_id: str | None = Field(default=None, alias="emailTemplateId")

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
    email_template_id: str | None = Field(default=None, alias="emailTemplateId")

    model_config = {"populate_by_name": True}


@router.get("/rules")
def list_rules(db: Session = Depends(get_db)):
    rules = db.query(AlertRule).order_by(AlertRule.created_at.asc()).all()
    return [rule_to_dict(r) for r in rules]


@router.post("/rules", status_code=201)
def create_rule(body: AlertRuleCreate, db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    action = get_action(body.case_type)
    if not action and body.case_type not in ("user_not_working", "user_left_position", "forklift_in_zone", "unknown"):
        raise HTTPException(status_code=400, detail=f"Unknown case type: {body.case_type}")

    rule = AlertRule(
        id=new_id("rule"),
        icon=body.icon or (action.icon if action else "warning"),
        title=body.title,
        description=body.description,
        zone=body.zone,
        case_type=body.case_type,
        severity=body.severity,
        enabled=body.enabled,
        notify_email=body.notify_email,
        email_template_id=body.email_template_id,
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
        (body.email_template_id, "email_template_id"),
    ]:
        if field is not None:
            setattr(rule, attr, field)

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
    rule = db.get(AlertRule, rule_id)
    if not rule:
        raise HTTPException(status_code=404, detail="Rule not found")
    rule.enabled = not rule.enabled
    if not rule.enabled:
        if rule.severity != "DISABLED":
            rule.severity = "DISABLED"
    elif rule.severity == "DISABLED":
        action = get_action(rule.case_type)
        rule.severity = action.default_severity if action else "WARNING"
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
