from __future__ import annotations

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from vision_ops_alerting.auth_deps import get_current_user
from vision_ops_alerting.db.models import User
from vision_ops_alerting.db.session import get_db
from vision_ops_alerting.services.plant_settings import (
    KPI_DEFINITIONS,
    get_plant_config,
    plant_to_dict,
    update_plant_config,
)

router = APIRouter(prefix="/api/settings", tags=["settings"])


class PlantSettingsUpdate(BaseModel):
    siteName: str | None = None
    lineCostPerMinute: float | None = Field(None, ge=0)
    materialCostPerUnit: float | None = Field(None, ge=0)
    targetCycleSec: float | None = Field(None, gt=0)
    shiftHours: float | None = Field(None, gt=0)
    uptimeCriticalPenalty: float | None = Field(None, ge=0)
    uptimeWarningPenalty: float | None = Field(None, ge=0)
    uptimeFloorPct: float | None = Field(None, ge=0, le=100)
    uptimeCeilingPct: float | None = Field(None, ge=0, le=100)
    performanceFloorPct: float | None = Field(None, ge=0, le=100)
    performanceCeilingPct: float | None = Field(None, ge=0, le=100)
    qualityFloorPct: float | None = Field(None, ge=0, le=100)
    qualityCeilingPct: float | None = Field(None, ge=0, le=100)
    defaultClipDurationSec: int | None = Field(None, ge=1)
    downtimeCriticalThresholdPct: float | None = Field(None, ge=0, le=1)
    inferenceBasePerCamera: int | None = Field(None, ge=0)
    inferenceProbeBonus: int | None = Field(None, ge=0)
    inferenceEventMultiplier: int | None = Field(None, ge=0)
    inferenceMinPerCamera: int | None = Field(None, ge=0)


@router.get("/plant")
def get_plant_settings(db: Session = Depends(get_db)):
    return plant_to_dict(get_plant_config(db))


@router.patch("/plant")
def patch_plant_settings(
    body: PlantSettingsUpdate,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    data = body.model_dump(exclude_unset=True)
    config = update_plant_config(db, data, updated_by=user.name)
    db.commit()
    return plant_to_dict(config)


@router.get("/kpi-definitions")
def get_kpi_definitions():
    return {"items": KPI_DEFINITIONS}
