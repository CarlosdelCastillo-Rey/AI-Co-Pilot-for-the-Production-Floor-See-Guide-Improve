from __future__ import annotations

from dataclasses import dataclass

from vision_ops_alerting.services.har_class_labels import registered_har_action_labels

HAR_CASE_TYPE = "har_action_detected"


@dataclass(frozen=True)
class AlertAction:
    case_type: str
    label: str
    description: str
    icon: str
    default_zone: str
    default_severity: str  # CRITICAL | WARNING | INFO
    default_enabled: bool = True
    har_action_label: str | None = None


def build_action_catalog() -> list[AlertAction]:
    return [
        AlertAction(
            case_type=HAR_CASE_TYPE,
            label=name,
            har_action_label=name,
            description=f"Notify when live HAR detects «{name}» above the confidence threshold.",
            icon="smart_toy",
            default_zone="HAR LIVE",
            default_severity="INFO",
        )
        for name in registered_har_action_labels()
    ]


ACTION_CATALOG: list[AlertAction] = build_action_catalog()

ACTIONS_BY_TYPE: dict[str, AlertAction] = {a.case_type: a for a in ACTION_CATALOG}


def get_action(case_type: str) -> AlertAction | None:
    return ACTIONS_BY_TYPE.get(case_type)


def get_action_for_har_label(label: str) -> AlertAction | None:
    for action in ACTION_CATALOG:
        if action.har_action_label == label:
            return action
    return None


def list_actions() -> list[dict]:
    return [
        {
            "caseType": a.case_type,
            "harActionLabel": a.har_action_label,
            "label": a.label,
            "description": a.description,
            "icon": a.icon,
            "defaultZone": a.default_zone,
            "defaultSeverity": a.default_severity,
            "defaultEnabled": a.default_enabled,
        }
        for a in ACTION_CATALOG
    ]
