from __future__ import annotations

from dataclasses import dataclass

from vision_ops_alerting.schemas import CaseType


@dataclass(frozen=True)
class AlertAction:
    case_type: CaseType
    label: str
    description: str
    icon: str
    default_zone: str
    default_severity: str  # CRITICAL | WARNING
    default_enabled: bool = True


ACTION_CATALOG: list[AlertAction] = [
    AlertAction(
        case_type="user_not_working",
        label="Operator Idle",
        description="Alert when an operator is idle beyond the configured threshold on an assembly line.",
        icon="schedule",
        default_zone="LINE 4",
        default_severity="WARNING",
    ),
    AlertAction(
        case_type="user_left_position",
        label="Left Position",
        description="Alert when an operator leaves an assigned station or geofenced work area.",
        icon="exit_to_app",
        default_zone="ZONE A-14",
        default_severity="CRITICAL",
    ),
    AlertAction(
        case_type="forklift_in_zone",
        label="Forklift in Zone",
        description="Alert when a forklift enters a restricted or pedestrian-only zone.",
        icon="forklift",
        default_zone="ZONE B",
        default_severity="CRITICAL",
    ),
    AlertAction(
        case_type="unknown",
        label="Generic Alert",
        description="Fallback alert for unmatched vision events or custom prompt-based rules.",
        icon="notification_important",
        default_zone="CUSTOM",
        default_severity="WARNING",
        default_enabled=False,
    ),
]

ACTIONS_BY_TYPE: dict[CaseType, AlertAction] = {a.case_type: a for a in ACTION_CATALOG}


def get_action(case_type: str) -> AlertAction | None:
    return ACTIONS_BY_TYPE.get(case_type)  # type: ignore[arg-type]


def list_actions() -> list[dict]:
    return [
        {
            "caseType": a.case_type,
            "label": a.label,
            "description": a.description,
            "icon": a.icon,
            "defaultZone": a.default_zone,
            "defaultSeverity": a.default_severity,
            "defaultEnabled": a.default_enabled,
        }
        for a in ACTION_CATALOG
    ]
