from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from vision_ops_alerting.auth_deps import get_current_user
from vision_ops_alerting.db.models import User
from vision_ops_alerting.db.session import get_db
from vision_ops_alerting.services.data_reset import purge_dynamic_alerting_data

router = APIRouter(prefix="/api/admin", tags=["admin"])


@router.post("/reset-dynamic-data")
def reset_dynamic_data(
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """Clear events, alerts, analytics rollups, and HAR activity — keep cameras & users."""
    counts = purge_dynamic_alerting_data(db)
    total = sum(counts.values())
    return {
        "ok": True,
        "clearedByTable": counts,
        "totalRowsCleared": total,
        "resetBy": user.email,
        "preserved": ["users", "cameras", "alert_rules", "email_templates", "plant_config", "industrial_reason_codes"],
    }
