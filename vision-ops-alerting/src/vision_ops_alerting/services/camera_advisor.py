"""Per-camera HAR advisor (scoped to one feed's logs)."""

from __future__ import annotations

import json
import re
from typing import Any

from sqlalchemy.orm import Session
from strands import Agent
from strands.models.ollama import OllamaModel

from vision_ops_alerting.config import settings
from vision_ops_alerting.services.har_activity_store import logs_for_advisor

CAMERA_ADVISOR_SYSTEM = """\
You are VisionOps Camera AI — an assistant for ONE industrial camera feed only.

You receive JSON with cameraId, session (video), summary stats, and recent HAR activity logs
(action labels, confidence %, person detections). Answer ONLY about this camera's video and logs.
Do not discuss other cameras or plant-wide topics unless the user asks how this feed compares in general terms.

Be concise: 2–4 sentences or up to 3 bullets. Cite specific actions and percentages from the logs when relevant.
No markdown headings. No emojis. If data is empty, say live inference may still be starting.
"""


def _trim(text: str, max_len: int = 480) -> str:
    cleaned = re.sub(r"\n{3,}", "\n\n", str(text).strip())
    if len(cleaned) <= max_len:
        return cleaned
    return cleaned[: max_len - 1].rsplit(" ", 1)[0] + "…"


def _fallback_camera_reply(ctx: dict[str, Any], message: str) -> str:
    logs = ctx.get("logs") or []
    summary = ctx.get("summary") or {}
    if not logs:
        return (
            "No persisted activity logs yet for this camera. "
            "If Live shows detections but this message persists, check that alerting (:8001) "
            "is running and POST /api/har/activity is not returning errors."
        )
    latest = logs[-1]
    label = latest.get("predictedLabel", "—")
    conf = latest.get("confidence")
    pct = int(round(float(conf or 0) * 100)) if conf is not None else 0
    non_asm = summary.get("nonAssemblyRatePct", 0)
    if not message.strip() or message.lower() in ("hi", "hello", "help"):
        return (
            f"Latest action on this feed: {label} ({pct}%). "
            f"Today non-assembly rate is about {non_asm}%. Ask me about patterns, persons, or deviations."
        )
    return (
        f"On this feed the latest detection is {label} at {pct}%. "
        f"Non-assembly share today is {non_asm}%. See the log panel for the full history."
    )


def run_camera_advisor(
    db: Session,
    *,
    camera_id: str,
    message: str,
    session_id: str | None = None,
    limit: int = 200,
) -> dict[str, Any]:
    ctx = logs_for_advisor(db, camera_id=camera_id, limit=limit, session_id=session_id)
    user_msg = message.strip() or "Summarize what is happening on this camera right now."
    payload = {"cameraContext": ctx, "userMessage": user_msg}

    try:
        model = OllamaModel(
            model_id=settings.ollama_model,
            temperature=settings.advisor_temperature,
        )
        agent = Agent(model=model, system_prompt=CAMERA_ADVISOR_SYSTEM, tools=[])
        raw = agent(json.dumps(payload, ensure_ascii=False, default=str))
        reply = _trim(str(raw))
        return {
            "reply": reply,
            "model": settings.ollama_model,
            "usedFallback": False,
            "cameraId": camera_id,
        }
    except Exception:
        return {
            "reply": _fallback_camera_reply(ctx, message),
            "model": "fallback",
            "usedFallback": True,
            "cameraId": camera_id,
        }
