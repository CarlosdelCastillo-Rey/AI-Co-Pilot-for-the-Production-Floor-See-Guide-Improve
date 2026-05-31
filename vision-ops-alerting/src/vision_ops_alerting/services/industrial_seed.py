from __future__ import annotations

from sqlalchemy.orm import Session

from vision_ops_alerting.db.models import IndustrialReasonCode
from vision_ops_alerting.services.plant_settings import get_plant_config

DEFAULT_REASON_CODES = [
    ("OP_IDLE", "Operator Idle", "Manpower", 1),
    ("OP_LEFT", "Left Position", "Manpower", 2),
    ("ZONE_BLOCK", "Zone Blocked", "Material Flow", 3),
    ("DEFECT_SKIP", "Skipped Step / Defect", "Quality", 4),
    ("EQUIP_FAULT", "Equipment Fault", "Equipment", 5),
    ("MATERIAL", "Material Shortage", "Material Flow", 6),
    ("FALSE_POS", "False Positive", "Vision System", 99),
]


def ensure_industrial_defaults(db: Session) -> None:
    for code, label, category, sort_order in DEFAULT_REASON_CODES:
        if db.get(IndustrialReasonCode, code):
            continue
        db.add(
            IndustrialReasonCode(
                code=code,
                label=label,
                category=category,
                sort_order=sort_order,
            )
        )

    get_plant_config(db)
