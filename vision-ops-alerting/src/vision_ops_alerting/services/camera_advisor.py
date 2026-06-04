"""Per-camera HAR advisor (scoped to one feed's logs)."""

from __future__ import annotations

import json
import re
from typing import Any

from sqlalchemy.orm import Session
from vision_ops_alerting.config import settings
from vision_ops_alerting.ollama_model import make_ollama_model
from vision_ops_alerting.strands_invoke import create_agent, invoke_agent
from vision_ops_alerting.services.camera_advisor_context import (
    build_camera_advisor_context,
    build_briefing_facts,
    parse_focus_minutes,
)

CAMERA_ADVISOR_SYSTEM = """\
You are VisionOps Camera AI — a conversational industrial vision copilot for ONE camera feed.

You receive:
- briefingFacts: a fact sheet extracted from video/session metadata, time windows (5/10/15/30 min), and inference logs
- focusWindow: stats for the user's requested time range
- recentLogLines: timestamped HAR reads (action, confidence %, persons)
- recentEvents: alerts on this camera
- logs: raw entries when you need specifics

Your job:
- Answer like a sharp floor supervisor talking to a colleague: warm, direct, specific.
- ALWAYS ground claims in briefingFacts and log lines — cite actions, %, counts, and time window.
- For "summary of last minutes" (or similar): give a structured narrative:
  1) what the clip/video is, 2) what dominated in the window, 3) confidence trend (low/high, rising/falling),
  4) non-assembly or deviations, 5) people in frame, 6) notable log lines or changes, 7) alerts if any.
- Use 4–8 sentences or short bullet lines (- prefix). No markdown headings (#). No emojis.
- Never say "see the log panel" as a substitute for summarizing — you ARE the log narrator.
- If data is sparse, say so and explain what to check (playback, alerting service, ingest).
"""


def _llm_error(exc: Exception) -> str:
    detail = str(exc).strip() or exc.__class__.__name__
    return f"ERROR: {detail}"


def _trim(text: str, max_len: int = 1400) -> str:
    cleaned = re.sub(r"\n{3,}", "\n\n", str(text).strip())
    if len(cleaned) <= max_len:
        return cleaned
    return cleaned[: max_len - 1].rsplit("\n", 1)[0] + "…"


def _build_agent_prompt(ctx: dict[str, Any], user_msg: str) -> str:
    slim = {
        "cameraId": ctx.get("cameraId"),
        "camera": ctx.get("camera"),
        "session": ctx.get("session"),
        "focusWindowMinutes": ctx.get("focusWindowMinutes"),
        "focusWindow": ctx.get("focusWindow"),
        "windows": ctx.get("windows"),
        "summary": ctx.get("summary"),
        "realtime": ctx.get("realtime"),
        "recentEvents": ctx.get("recentEvents"),
        "recentLogLines": ctx.get("recentLogLines"),
        "primaryActionLabel": ctx.get("primaryActionLabel"),
        "briefingFacts": ctx.get("briefingFacts"),
    }
    return (
        f"USER QUESTION:\n{user_msg}\n\n"
        f"BRIEFING (use these facts — do not invent):\n{ctx.get('briefingFacts') or build_briefing_facts(ctx)}\n\n"
        f"STRUCTURED CONTEXT JSON:\n{json.dumps(slim, ensure_ascii=False, default=str)}"
    )


def run_camera_advisor(
    db: Session,
    *,
    camera_id: str,
    message: str,
    session_id: str | None = None,
    limit: int = 200,
) -> dict[str, Any]:
    user_msg = message.strip() or "Give me a detailed summary of what happened on this camera in the last 10 minutes."
    focus_mins = parse_focus_minutes(user_msg)
    ctx = build_camera_advisor_context(
        db,
        camera_id=camera_id,
        session_id=session_id,
        limit=limit,
        focus_minutes=focus_mins,
    )

    temp = getattr(settings, "camera_advisor_temperature", None) or settings.advisor_temperature

    try:
        model = make_ollama_model(temperature=temp)
        agent = create_agent(model=model, system_prompt=CAMERA_ADVISOR_SYSTEM, tools=[])
        reply = _trim(invoke_agent(agent, _build_agent_prompt(ctx, user_msg)))
        if not reply.strip():
            return {
                "reply": "ERROR: LLM returned an empty response.",
                "model": settings.ollama_model,
                "usedFallback": False,
                "cameraId": camera_id,
            }
        return {
            "reply": reply,
            "model": settings.ollama_model,
            "usedFallback": False,
            "cameraId": camera_id,
        }
    except Exception as exc:
        return {
            "reply": _llm_error(exc),
            "model": settings.ollama_model,
            "usedFallback": False,
            "cameraId": camera_id,
        }
