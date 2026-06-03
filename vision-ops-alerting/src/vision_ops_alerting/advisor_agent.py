"""VisionOps floor advisor — Strands + Ollama (same model as alert classifier)."""

from __future__ import annotations

import json
import re
from typing import Any

from sqlalchemy.orm import Session
from strands import Agent
from strands.models.ollama import OllamaModel

from vision_ops_alerting.config import settings
from vision_ops_alerting.services.advisor_context import (
    enrich_snapshot_for_page,
    is_greeting,
    page_meta,
)
from vision_ops_alerting.services.advisor_tools import make_advisor_tools
from vision_ops_alerting.services.operational_snapshot import build_operational_snapshot

ADVISOR_SYSTEM_PROMPT = """\
You are VisionOps AI — a senior industrial operations advisor for a vision-monitored factory floor.

CRITICAL: The user is on a specific app screen (currentScreen in the payload). Tailor every answer to that screen first.
- On Alerts: discuss rules, templates, email/dry-run — NOT generic floor patrol unless incidents are open.
- On Live: use harLiveDashboard in currentScreen (or query_all_cameras_har_dashboard tool) for per-camera
  detected actions, confidence %, inference counts, and open HAR incidents. List each cam-har-* feed by name/id.
- On Timeline: open/ack/resolve workflow.
- On Analytics: OEE, COQ, KPIs vs incidents; use HAR summaries when camera is cam-har-*.
- On Settings: plant cost parameters.

You receive operational snapshot JSON, currentScreen, page, and userMessage.
When the user asks for all cameras, actions by camera, or incidents on Live, ALWAYS call
query_all_cameras_har_dashboard or read harLiveDashboard — never reply with only "streams look nominal".

For greetings (hi/hello): reply warmly in 1 sentence, state what you can do on THIS screen, optional 1-line plant status.
For substantive questions: use bullet lines per camera when listing actions. Be direct. No invented metrics. No markdown headings. No emojis.
"""


def _make_advisor_model() -> OllamaModel:
    return OllamaModel(
        model_id=settings.ollama_model,
        temperature=settings.advisor_temperature,
    )


def _trim_reply(text: str, max_len: int = 520) -> str:
    cleaned = re.sub(r"\n{3,}", "\n\n", str(text).strip())
    if len(cleaned) <= max_len:
        return cleaned
    cut = cleaned[: max_len - 1].rsplit(" ", 1)[0]
    return cut + "…"


def _fallback_greeting(snapshot: dict[str, Any], page: str, page_title: str | None) -> str:
    meta = page_meta(page, page_title)
    floor = snapshot.get("floor") or {}
    screen = snapshot.get("currentScreen") or {}

    parts = [f"Hi — I'm VisionOps AI on {meta['label']}."]

    if page == "alerts":
        active = screen.get("enabledRules", snapshot.get("alertRules", {}).get("active", 0))
        total = screen.get("totalRules", snapshot.get("alertRules", {}).get("total", 0))
        dry = "on (no emails sent)" if snapshot.get("emailDryRun") else "off"
        parts.append(
            f"I help with rules, templates, and MailerSend setup — {active}/{total} rules enabled, dry-run {dry}."
        )
        if floor.get("openCriticalCount") or floor.get("openCount"):
            parts.append(
                f"There are also {floor.get('openCount', 0)} open floor incident(s); check Timeline when ready."
            )
        else:
            parts.append("Floor queue is clear—good time to review rules or send a test alert.")
    elif page == "live":
        cams = snapshot.get("cameras", {})
        parts.append(
            f"I watch {cams.get('live', 0)}/{cams.get('total', 0)} live cameras and inference health—ask what needs attention."
        )
    elif page == "timeline":
        oc = floor.get("openCount", 0)
        parts.append(
            f"I help prioritize and close incidents — {oc} open today." if oc else "I help triage and resolve shift incidents — queue looks clear."
        )
    elif page == "analytics":
        parts.append(
            f"I connect KPIs to real incidents — uptime {floor.get('uptime', '—')} today; ask about OEE, COQ, or bottlenecks."
        )
    else:
        parts.append(f"I focus on {meta['focus']} — ask for ops, safety, or cost guidance.")

    return _trim_reply(" ".join(parts))


def _fallback_advice(
    db: Session,
    snapshot: dict[str, Any],
    message: str,
    page: str,
    page_title: str | None,
) -> str:
    if is_greeting(message):
        return _fallback_greeting(snapshot, page, page_title)

    if page == "live":
        from vision_ops_alerting.services.advisor_har_fallback import format_live_har_reply

        har_reply = format_live_har_reply(db, message)
        if har_reply:
            return har_reply

    meta = page_meta(page, page_title)
    floor = snapshot.get("floor") or {}
    cams = snapshot.get("cameras") or {}
    open_alerts = snapshot.get("openAlerts") or []
    critical = floor.get("openCriticalCount", 0)
    open_n = floor.get("openCount", 0)
    live = cams.get("live", 0)
    total = cams.get("total", 0)
    backend = snapshot.get("visionBackend") or {}
    screen = snapshot.get("currentScreen") or {}

    parts: list[str] = []

    # Page-first advice
    if page == "alerts":
        active = screen.get("enabledRules", snapshot.get("alertRules", {}).get("active", 0))
        total_rules = screen.get("totalRules", snapshot.get("alertRules", {}).get("total", 0))
        dry = snapshot.get("emailDryRun")
        parts.append(
            f"On Alerts: {active}/{total_rules} rules on; dry-run {'enabled' if dry else 'disabled'}."
        )
        disabled_email = [
            r.get("title", r.get("caseType"))
            for r in (screen.get("rules") or [])
            if r.get("enabled") and not r.get("notifyEmail")
        ]
        if disabled_email:
            parts.append(f"Rules without email: {', '.join(disabled_email[:3])}.")
        if not critical and not open_n:
            parts.append("Floor is quiet—use test email on a rule or enable a missing case type.")
        elif open_n:
            parts.append(f"{open_n} open floor incident(s)—rules may already be firing; check Timeline.")
    elif page == "live":
        parts.append(f"Live view: {live}/{total} cameras online.")
        screen_dash = (screen.get("harLiveDashboard") or {}).get("cameras") or []
        if screen_dash:
            for cam in screen_dash[:5]:
                cid = cam.get("cameraId", "?")
                act = cam.get("latestAction") or "—"
                pct = cam.get("latestConfidencePct")
                pct_s = f" ({pct}%)" if pct is not None else ""
                parts.append(f"{cid}: {act}{pct_s}.")
        elif critical:
            parts.append(f"{critical} critical alert(s)—open Timeline or the notification bell.")
        elif open_n:
            parts.append(f"{open_n} open incident(s) to track.")
        else:
            parts.append("Streams look nominal—watch inference load in the activity panel.")
    elif page == "timeline":
        if open_n:
            top = open_alerts[0].get("title", "top open item") if open_alerts else "open items"
            parts.append(f"{open_n} open—start with “{top}”.")
        else:
            parts.append("No open incidents—review resolved items for shift report.")
    elif page == "analytics":
        parts.append(f"Analytics: uptime {floor.get('uptime', '—')}, {floor.get('totalEventsToday', 0)} events today.")
        if open_n:
            parts.append("Open incidents may depress OEE—reconcile before sharing with finance.")
    else:
        if critical:
            top = next((a for a in open_alerts if a.get("severity") == "critical"), open_alerts[0] if open_alerts else None)
            title = top.get("title", "critical incident") if top else "critical incidents"
            parts.append(f"Prioritize {critical} critical alert(s)—“{title}” first.")
        elif open_n:
            parts.append(f"{open_n} open incident(s); check Timeline.")
        else:
            parts.append(f"On {meta['label']}: floor alerts clear.")

    if total and live < total:
        parts.append(f"{total - live} camera(s) offline—fix before trusting live KPIs.")
    if not backend.get("reachable"):
        parts.append("Vision backend (:8000) unreachable.")

    return _trim_reply(" ".join(parts[:6]), max_len=900)


def _is_generic_live_reply(reply: str) -> bool:
    low = reply.lower()
    generic_markers = (
        "streams look nominal",
        "watch inference load",
        "cameras online",
    )
    return any(m in low for m in generic_markers) and "cam-har" not in low and "•" not in reply


def _prefer_live_har_reply(db: Session, message: str, llm_reply: str) -> str:
    from vision_ops_alerting.services.advisor_har_fallback import format_live_har_reply

    har_reply = format_live_har_reply(db, message)
    if not har_reply:
        return llm_reply
    if _is_generic_live_reply(llm_reply) or len(llm_reply.strip()) < 120:
        return har_reply
    return llm_reply


def _prompt_snapshot(snapshot: dict[str, Any]) -> dict[str, Any]:
    """Copy snapshot for the LLM prompt without oversized nested payloads."""
    import copy

    slim = copy.deepcopy(snapshot)
    slim["openAlerts"] = (slim.get("openAlerts") or [])[:6]
    return slim


def run_advisor(
    db: Session,
    *,
    message: str,
    page: str = "live",
    page_title: str | None = None,
    camera_id: str | None = None,
    client_alerts: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    snapshot = enrich_snapshot_for_page(
        db,
        build_operational_snapshot(db, page=page),
        page,
        camera_id=camera_id,
    )
    if client_alerts:
        snapshot["clientAlerts"] = client_alerts[:10]

    meta = page_meta(page, page_title)
    user_block = {
        "page": page,
        "pageTitle": meta["label"],
        "currentScreen": snapshot.get("currentScreen"),
        "userMessage": message.strip() or "What should I focus on on this screen?",
        "operationalSnapshot": _prompt_snapshot(snapshot),
    }
    prompt = json.dumps(user_block, ensure_ascii=False)

    try:
        model = _make_advisor_model()
        tools = make_advisor_tools(db)
        agent = Agent(
            model=model,
            system_prompt=ADVISOR_SYSTEM_PROMPT,
            tools=tools,
        )
        raw = agent(prompt)
        reply = _trim_reply(str(raw), max_len=900 if page == "live" else 520)
        if page == "live":
            reply = _prefer_live_har_reply(db, message, reply)
        return {
            "reply": reply,
            "model": settings.ollama_model,
            "usedFallback": False,
            "page": page,
            "pageTitle": meta["label"],
            "snapshot": {
                "openCriticalCount": snapshot.get("floor", {}).get("openCriticalCount", 0),
                "openCount": snapshot.get("floor", {}).get("openCount", 0),
                "camerasLive": snapshot.get("cameras", {}).get("live", 0),
            },
        }
    except Exception:
        return {
            "reply": _fallback_advice(db, snapshot, message, page, page_title),
            "model": "fallback",
            "usedFallback": True,
            "page": page,
            "pageTitle": meta["label"],
            "snapshot": {
                "openCriticalCount": snapshot.get("floor", {}).get("openCriticalCount", 0),
                "openCount": snapshot.get("floor", {}).get("openCount", 0),
                "camerasLive": snapshot.get("cameras", {}).get("live", 0),
            },
        }
