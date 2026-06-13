from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


ActorType = Literal["operator", "forklift", "unknown"]
Severity = Literal["info", "warning", "critical"]
CaseType = Literal["user_not_working", "user_left_position", "forklift_in_zone", "unknown"]


class Actor(BaseModel):
    type: ActorType = "unknown"
    track_id: str | None = None
    name: str | None = None


class Evidence(BaseModel):
    idle_seconds: int | None = None
    roi: str | None = None
    bbox: list[float] | None = None
    snapshot_path: str | None = None
    clip_path: str | None = None


class Links(BaseModel):
    live_url: str | None = None
    timeline_url: str | None = None


class IndustrialContext(BaseModel):
    site_id: str
    line_id: str
    camera_id: str
    timestamp: str
    actor: Actor = Field(default_factory=Actor)
    evidence: Evidence = Field(default_factory=Evidence)
    links: Links = Field(default_factory=Links)

    # Optional upstream classification if already known
    case_type: CaseType | None = None
    severity: Severity | None = None


ResolutionStatus = Literal["OPEN", "ACKNOWLEDGED", "RESOLVED", "FALSE_POSITIVE"]


class ResolveEventBody(BaseModel):
    status: ResolutionStatus = "RESOLVED"
    reasonCode: str | None = None
    downtimeSeconds: int = Field(default=0, ge=0)
    scrapUnits: int = Field(default=0, ge=0)
    notes: str | None = None


class EmailSent(BaseModel):
    type: Literal["email_sent"] = "email_sent"
    event_id: str
    delivery_id: str
    template_id: str
    case_type: CaseType
    severity: Severity
    to_emails: list[str]
    message_ids: list[str] = Field(default_factory=list)
    dry_run: bool
    telegram_delivery_id: str = ""
    telegram_message_ids: list[str] = Field(default_factory=list)
    telegram_chat_ids: list[str] = Field(default_factory=list)
    telegram_dry_run: bool = False
    telegram_error: str | None = None


class EmailSendResponse(BaseModel):
    ok: bool
    event: EmailSent

