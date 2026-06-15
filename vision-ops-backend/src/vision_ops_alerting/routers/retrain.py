"""HAR incremental retraining API."""

from __future__ import annotations

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from vision_ops_alerting.db.session import get_db
from vision_ops_alerting.services.retrain import dataset_stats, job_state, start_retrain

router = APIRouter()


class RetrainRequest(BaseModel):
    backbone: str = Field(default="vjepa", pattern="^(vjepa|dinov2)$")
    epochs: int = Field(default=30, ge=5, le=200)


@router.post("/api/har/retrain")
def trigger_retrain(body: RetrainRequest) -> dict:
    started = start_retrain(backbone=body.backbone, epochs=body.epochs)
    if not started:
        return {"status": "already_running", "message": "A retrain job is already in progress."}
    return {"status": "started", "backbone": body.backbone, "epochs": body.epochs}


@router.get("/api/har/retrain/status")
def retrain_status() -> dict:
    return job_state()


@router.get("/api/har/retrain/dataset-stats")
def retrain_dataset_stats(db: Session = Depends(get_db)) -> dict:
    return dataset_stats(db)
