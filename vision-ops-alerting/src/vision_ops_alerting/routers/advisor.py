from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from vision_ops_alerting.advisor_agent import run_advisor
from vision_ops_alerting.db.session import get_db
from vision_ops_alerting.services.advisor_context import build_welcome
from vision_ops_alerting.services.operational_snapshot import build_operational_snapshot

router = APIRouter(prefix="/api/advisor", tags=["advisor"])


class AdvisorChatRequest(BaseModel):
    message: str = ""
    page: str = "live"
    pageTitle: str | None = None
    alerts: list[dict[str, Any]] = Field(default_factory=list)


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
        client_alerts=body.alerts or None,
    )
    return AdvisorChatResponse(**result)


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
