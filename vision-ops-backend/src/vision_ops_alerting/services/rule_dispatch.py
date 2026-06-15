from __future__ import annotations

from sqlalchemy.orm import Session

from vision_ops_alerting.config import settings
from vision_ops_alerting.db.models import AlertRule, new_id
from vision_ops_alerting.schemas import CaseType
from vision_ops_alerting.services.alert_actions import HAR_CASE_TYPE, build_action_catalog, get_action
from vision_ops_alerting.services.rule_config import (
    config_json_for_har_rule,
    rule_har_action_label,
)
from vision_ops_alerting.services.events import find_matching_rule

MOCK_CASE_TYPES = frozenset(
    {
        "user_not_working",
        "user_left_position",
        "forklift_in_zone",
        "unknown",
        "har_action_deviation",
    }
)

HAR_TEMPLATE_ID = "har.action_detected"


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




def require_enabled_har_rule(db: Session, har_action_label: str) -> AlertRule:
    from vision_ops_alerting.services.rule_config import find_har_rule_for_label

    rule = find_har_rule_for_label(db, har_action_label)
    if rule:
        return rule
    for row in db.query(AlertRule).filter(AlertRule.case_type == HAR_CASE_TYPE).all():
        if rule_har_action_label(row) == har_action_label:
            raise RuleNotEnabledError(
                f"Rule for «{har_action_label}» is disabled. Enable it in Alert Rules."
            )
    raise RuleNotEnabledError(
        f"No rule for «{har_action_label}». Create one on the Alerts page."
    )

def prune_mock_alert_rules(db: Session) -> int:
    """Remove legacy industrial / generic demo alert rules."""
    removed = 0
    for rule in db.query(AlertRule).all():
        if rule.case_type in MOCK_CASE_TYPES:
            db.delete(rule)
            removed += 1
            continue
        if rule.case_type == HAR_CASE_TYPE and not rule_har_action_label(rule):
            db.delete(rule)
            removed += 1
    if removed:
        db.flush()
    return removed


def _har_rule_exists(db: Session, har_action_label: str) -> bool:
    for rule in db.query(AlertRule).filter(AlertRule.case_type == HAR_CASE_TYPE).all():
        if rule_har_action_label(rule) == har_action_label:
            return True
    return False


def ensure_default_action_rules(db: Session) -> None:
    """Ensure one INFO alert rule per registered HAR action class."""
    prune_mock_alert_rules(db)

    for action in build_action_catalog():
        label = action.har_action_label
        if not label or _har_rule_exists(db, label):
            continue
        db.add(
            AlertRule(
                id=new_id("rule"),
                icon=action.icon,
                title=f"HAR — {label}",
                description=action.description,
                zone=action.default_zone,
                case_type=HAR_CASE_TYPE,
                severity=action.default_severity,
                enabled=action.default_enabled,
                notify_email=True,
                notify_telegram=True,
                email_template_id=HAR_TEMPLATE_ID,
                config_json=config_json_for_har_rule(
                    har_action_label=label,
                    confidence_threshold=settings.har_notify_confidence_threshold,
                    event_severity=action.default_severity,
                    notify_in_app=False,
                ),
            )
        )
