from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from vision_ops_alerting.advisor_agent import run_advisor
from vision_ops_alerting.db.session import get_db
from vision_ops_alerting.services.advisor_context import build_welcome
from vision_ops_alerting.services.camera_advisor import run_camera_advisor
from vision_ops_alerting.services.operational_snapshot import build_operational_snapshot

router = APIRouter(prefix="/api/advisor", tags=["advisor"])


class AdvisorChatRequest(BaseModel):
    message: str = ""
    page: str = "live"
    pageTitle: str | None = None
    camera_id: str | None = Field(None, alias="cameraId")
    alerts: list[dict[str, Any]] = Field(default_factory=list)


class CameraAdvisorChatRequest(BaseModel):
    camera_id: str = Field(..., alias="cameraId")
    message: str = ""
    session_id: str | None = Field(None, alias="sessionId")
    limit: int = Field(200, ge=20, le=500)


class AdvisorChatResponse(BaseModel):
    reply: str
    model: str
    usedFallback: bool
    snapshot: dict[str, Any]


@router.post("/chat", response_model=AdvisorChatResponse)
def advisor_chat(body: AdvisorChatRequest, db: Session = Depends(get_db)):
    result = run_advisor(
        db,
        message=body.message,
        page=body.page,
        page_title=body.pageTitle,
        camera_id=body.camera_id,
        client_alerts=body.alerts or None,
    )
    return AdvisorChatResponse(**result)


@router.post("/camera-chat", response_model=AdvisorChatResponse)
def camera_advisor_chat(body: CameraAdvisorChatRequest, db: Session = Depends(get_db)):
    result = run_camera_advisor(
        db,
        camera_id=body.camera_id,
        message=body.message,
        session_id=body.session_id,
        limit=body.limit,
    )
    return AdvisorChatResponse(
        reply=result["reply"],
        model=result["model"],
        usedFallback=result["usedFallback"],
        snapshot={"cameraId": result["cameraId"]},
    )


@router.get("/welcome")
def advisor_welcome(
    page: str = "live",
    pageTitle: str | None = None,
    db: Session = Depends(get_db),
):
    """Intro + page-specific status when the chat panel opens."""
    return build_welcome(db, page=page, page_title=pageTitle)


@router.get("/context")
def advisor_context(page: str = "live", db: Session = Depends(get_db)):
    """Debug / MCP parity: raw operational snapshot."""
    return build_operational_snapshot(db, page=page)
