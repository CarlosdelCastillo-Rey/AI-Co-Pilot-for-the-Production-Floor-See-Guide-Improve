from __future__ import annotations

from sqlalchemy.orm import Session

from vision_ops_alerting.db.models import AlertRule, new_id
from vision_ops_alerting.schemas import CaseType
from vision_ops_alerting.services.alert_actions import ACTION_CATALOG, get_action
from vision_ops_alerting.services.events import find_matching_rule


class RuleNotEnabledError(Exception):
    pass


def require_enabled_rule(db: Session, case_type: CaseType) -> AlertRule:
    rule = find_matching_rule(db, case_type)
    if not rule:
        action = get_action(case_type)
        label = action.label if action else case_type
        raise RuleNotEnabledError(
            f"No enabled rule for “{label}”. Enable or create a rule in Alert Rules."
        )
    return rule


DEFAULT_TEMPLATE_BY_CASE = {
    "user_not_working": "operator.idle",
    "user_left_position": "operator.left_position",
    "forklift_in_zone": "forklift.zone_intrusion",
    "unknown": "generic.event",
}


def ensure_default_action_rules(db: Session) -> None:
    """Ensure one default rule exists per supported case type."""
    for action in ACTION_CATALOG:
        exists = (
            db.query(AlertRule.id)
            .filter(AlertRule.case_type == action.case_type)
            .first()
        )
        if exists:
            continue
        db.add(
            AlertRule(
                id=new_id("rule"),
                icon=action.icon,
                title=f"{action.label} Detection",
                description=action.description,
                zone=action.default_zone,
                case_type=action.case_type,
                severity=action.default_severity,
                enabled=action.default_enabled,
                notify_email=True,
                email_template_id=DEFAULT_TEMPLATE_BY_CASE.get(action.case_type),
            )
        )
