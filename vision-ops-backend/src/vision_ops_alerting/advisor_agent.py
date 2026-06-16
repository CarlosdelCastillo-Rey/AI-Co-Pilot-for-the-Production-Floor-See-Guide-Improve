"""VisionOps floor advisor — Strands + Ollama (same model as alert classifier)."""

from __future__ import annotations

import json
import re
from typing import Any

from sqlalchemy.orm import Session

from vision_ops_alerting.config import settings
from vision_ops_alerting.ollama_model import make_ollama_model
from vision_ops_alerting.strands_invoke import create_agent, invoke_agent
from vision_ops_alerting.services.advisor_context import (
    enrich_snapshot_for_page,
    is_greeting,
    page_meta,
)
from vision_ops_alerting.services.advisor_tools import make_advisor_action_tools, make_advisor_tools
from vision_ops_alerting.services.operational_snapshot import build_operational_snapshot

ADVISOR_SYSTEM_PROMPT = """\
You are VisionOps AI — a conversational senior industrial operations advisor for a vision-monitored factory floor.

The user message is a natural-language question from an operator. Plant data may appear below as reference — it is NOT
something to document. NEVER explain JSON, keys, schemas, field names, or "this is a JSON/API response". NEVER use
markdown bold on field names like **serverAlerts**. Answer only what the operator asked, using facts from the data.

CRITICAL: Tailor every answer to the current screen (page / pageTitle).
- On Alerts: rules, templates, email/dry-run — not generic patrol unless incidents are open.
- On Live: per-camera HAR actions with confidence %, inference counts, open incidents. List each cam-har-* feed.
  Use the "Camera HAR briefing" section when present; otherwise query_all_cameras_har_dashboard. Never one vague line.
- On Timeline: open/ack/resolve with specific incident titles and severities.
- On Analytics: OEE, COQ, KPIs vs incidents.
- On Settings: plant cost parameters.

Style: helpful ops partner — direct, specific. Use bullet lines (- or •) for camera lists. Cite real counts and %.
No invented metrics. No markdown headings (#). No emojis. Do not say "check the UI" when data is already provided.

You can also TAKE ACTIONS when the operator asks:
- acknowledge_event / resolve_event / dismiss_event: Change incident status by event ID.
- toggle_alert_rule: Enable or disable an alert rule by rule ID.
- rename_person: Assign a display name to a person by their global_person_id.
- trigger_har_probe: Run a fresh HAR inference probe on a model.
- send_test_alert: Send a test alert email or Telegram notification.
- list_persons / get_person_report: Query person registry and activity.

Before executing acknowledge/resolve/dismiss/rename, confirm in one sentence what you are about to do.
After completing any action, confirm with a plain one-line summary of what changed.
"""


def _make_advisor_model():
    return make_ollama_model(temperature=settings.advisor_temperature)


def _trim_reply(text: str, max_len: int = 720) -> str:
    cleaned = re.sub(r"\n{3,}", "\n\n", str(text).strip())
    if len(cleaned) <= max_len:
        return cleaned
    cut = cleaned[: max_len - 1].rsplit(" ", 1)[0]
    return cut + "…"


def _clean_advisor_reply(text: str) -> str:
    """Drop tool-call JSON leaks and empty preamble lines."""
    lines: list[str] = []
    for line in str(text).splitlines():
        stripped = line.strip()
        if stripped.startswith('{"function"') or stripped.startswith('{"name"'):
            continue
        if stripped.startswith("The model's response is:"):
            continue
        lines.append(line)
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines).strip())


def _live_reply_uses_har_data(reply: str) -> bool:
    low = reply.lower()
    return "cam-har" in low or "cam_har" in low


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
    slim.pop("currentScreen", None)
    slim["openAlerts"] = (slim.get("openAlerts") or [])[:6]
    ui = slim.pop("clientAlerts", None)
    if ui:
        slim["uiRecentAlerts"] = ui[:6]
    return slim


def _is_meta_json_reply(reply: str) -> bool:
    """Detect when the model documents payload structure instead of answering."""
    low = reply.lower()
    markers = (
        "json response",
        "breakdown of the response",
        "surveillance system",
        "serveralerts",
        "clientalerts",
        "each object contains",
        "array of objects",
        "unique identifier for the alert",
        "this is a json",
        "here's a breakdown",
    )
    hits = sum(1 for m in markers if m in low)
    return hits >= 2 or "json response" in low or "**" in reply and "alert" in low


def _live_har_briefing_for_prompt(snapshot: dict[str, Any], message: str) -> str | None:
    from vision_ops_alerting.services.advisor_har_fallback import format_har_dashboard_briefing

    screen = snapshot.get("currentScreen") or {}
    dash = screen.get("harLiveDashboard")
    if not dash or not (dash.get("cameras") or []):
        return None
    return format_har_dashboard_briefing(dash, message=message, max_len=1400)


def _build_advisor_user_prompt(
    *,
    message: str,
    page: str,
    meta: dict[str, str],
    snapshot: dict[str, Any],
) -> str:
    """Natural-language prompt so the model answers the operator, not the JSON envelope."""
    user_msg = message.strip() or "What should I focus on on this screen?"
    screen = snapshot.get("currentScreen") or {}
    plant = _prompt_snapshot(snapshot)

    lines = [
        "Answer the operator's question in plain language using the plant data below.",
        "Do NOT describe JSON keys, schemas, or API structure.",
        "",
        f"Operator question: {user_msg}",
        f"Screen: {meta['label']} (page={page})",
    ]

    har_brief = _live_har_briefing_for_prompt(snapshot, user_msg) if page == "live" else None
    if har_brief:
        lines.extend(
            [
                "",
                "Camera HAR briefing (authoritative — your answer MUST list these cam-har-* feeds by id; "
                "do not invent camera names):",
                har_brief,
            ]
        )

    lines.extend(
        [
            "",
            "Plant reference data (use facts only; do not explain this block):",
            json.dumps(
                {
                    "floor": plant.get("floor"),
                    "cameras": plant.get("cameras"),
                    "openAlerts": plant.get("openAlerts"),
                    "uiRecentAlerts": plant.get("uiRecentAlerts"),
                    "alertRules": plant.get("alertRules"),
                    "visionBackend": plant.get("visionBackend"),
                    "currentScreen": {
                        k: v
                        for k, v in screen.items()
                        if k != "harLiveDashboard"
                    },
                },
                ensure_ascii=False,
            ),
        ]
    )
    if page == "live" and screen.get("harLiveDashboard") and not har_brief:
        lines.append(
            json.dumps({"harLiveDashboard": screen["harLiveDashboard"]}, ensure_ascii=False)
        )
    return "\n".join(lines)


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
    prompt = _build_advisor_user_prompt(
        message=message,
        page=page,
        meta=meta,
        snapshot=snapshot,
    )

    har_in_prompt = page == "live" and _live_har_briefing_for_prompt(snapshot, message) is not None

    try:
        model = _make_advisor_model()
        read_tools = [] if har_in_prompt else make_advisor_tools(db)
        tools = read_tools + make_advisor_action_tools()
        agent = create_agent(
            model=model,
            system_prompt=ADVISOR_SYSTEM_PROMPT,
            tools=tools,
        )
        reply = _clean_advisor_reply(invoke_agent(agent, prompt))
        reply = _trim_reply(reply, max_len=1200 if page == "live" else 720)
        if page == "live":
            from vision_ops_alerting.services.advisor_har_fallback import format_live_har_reply

            har_reply = format_live_har_reply(db, message, max_len=1200)
            if har_reply and (
                _is_meta_json_reply(reply)
                or not _live_reply_uses_har_data(reply)
                or re.search(r"camera-0[12]|warehouse|v-jepa", reply, re.I)
            ):
                reply = har_reply
        if not reply.strip():
            reply = "ERROR: LLM returned an empty response."
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
    except Exception as exc:
        detail = str(exc).strip() or exc.__class__.__name__
        return {
            "reply": f"ERROR: {detail}",
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
