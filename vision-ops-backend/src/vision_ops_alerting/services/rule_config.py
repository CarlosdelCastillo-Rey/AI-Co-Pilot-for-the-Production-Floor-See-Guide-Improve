from __future__ import annotations

import json

from vision_ops_alerting.db.models import AlertRule

HAR_DETECT_CASE_TYPE = "har_action_detected"


def parse_rule_config(rule: AlertRule) -> dict:
    if not rule.config_json:
        return {}
    try:
        data = json.loads(rule.config_json)
        return data if isinstance(data, dict) else {}
    except json.JSONDecodeError:
        return {}


def rule_har_action_label(rule: AlertRule) -> str | None:
    cfg = parse_rule_config(rule)
    label = cfg.get("harActionLabel")
    return str(label) if label else None


def rule_confidence_threshold(rule: AlertRule, default: float) -> float:
    cfg = parse_rule_config(rule)
    try:
        val = float(cfg.get("confidenceThreshold", default))
        return max(0.05, min(0.99, val))
    except (TypeError, ValueError):
        return default


def rule_event_severity(rule: AlertRule) -> str:
    """Timeline / notification severity: info | warning | critical."""
    cfg = parse_rule_config(rule)
    raw = cfg.get("eventSeverity") or rule.severity or "INFO"
    if isinstance(raw, str):
        val = raw.upper()
        if val in ("CRITICAL", "WARNING", "INFO"):
            return val.lower()
    return "info"


def rule_notify_in_app(rule: AlertRule) -> bool:
    cfg = parse_rule_config(rule)
    if "notifyInApp" in cfg:
        return bool(cfg["notifyInApp"])
    return False


def rule_has_active_channel(rule: AlertRule) -> bool:
    """At least one delivery path enabled on the rule (matches /alerts UI)."""
    return bool(rule.notify_email or rule.notify_telegram or rule_notify_in_app(rule))


def config_json_for_har_rule(
    *,
    har_action_label: str | None = None,
    confidence_threshold: float | None = None,
    event_severity: str | None = None,
    notify_in_app: bool | None = None,
    existing: AlertRule | None = None,
) -> str:
    cfg = parse_rule_config(existing) if existing else {}
    if har_action_label is not None:
        cfg["harActionLabel"] = har_action_label
    if confidence_threshold is not None:
        cfg["confidenceThreshold"] = max(0.05, min(0.99, float(confidence_threshold)))
    if event_severity is not None:
        sev = event_severity.upper()
        if sev in ("CRITICAL", "WARNING", "INFO"):
            cfg["eventSeverity"] = sev.lower()
    if notify_in_app is not None:
        cfg["notifyInApp"] = bool(notify_in_app)
    return json.dumps(cfg)


def config_json_for_threshold(confidence_threshold: float | None) -> str | None:
    if confidence_threshold is None:
        return None
    return config_json_for_har_rule(confidence_threshold=confidence_threshold)


def find_har_rule_for_label(db, label: str) -> AlertRule | None:
    from sqlalchemy.orm import Session

    if not isinstance(db, Session):
        return None
    rules = (
        db.query(AlertRule)
        .filter(
            AlertRule.case_type == HAR_DETECT_CASE_TYPE,
            AlertRule.enabled.is_(True),
        )
        .all()
    )
    for rule in rules:
        if rule_har_action_label(rule) == label:
            return rule
    return None
