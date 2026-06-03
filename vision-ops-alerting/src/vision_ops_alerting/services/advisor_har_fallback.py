"""Deterministic multi-camera HAR replies when Ollama/advisor agent is unavailable."""

from __future__ import annotations

import re
from typing import Any

from sqlalchemy.orm import Session

from vision_ops_alerting.services.har_activity_store import build_all_cameras_har_dashboard


def _wants_multi_camera_har(message: str) -> bool:
    q = message.lower()
    keys = (
        "all camera",
        "all cam",
        "each camera",
        "by camera",
        "per camera",
        "summary",
        "list",
        "actions",
        "incident",
        "har",
        "feed",
        "cam-har",
    )
    return any(k in q for k in keys)


def format_live_har_reply(db: Session, message: str, *, max_len: int = 1100) -> str | None:
    """Build a per-camera action/incident summary from SQLite HAR logs."""
    if not _wants_multi_camera_har(message):
        return None

    dash = build_all_cameras_har_dashboard(db, hours=24)
    cameras = dash.get("cameras") or []
    if not cameras:
        return None

    lines: list[str] = []
    lines.append(f"HAR live summary — {dash.get('cameraCount', len(cameras))} camera(s):")

    for cam in cameras:
        cid = cam.get("cameraId", "?")
        name = cam.get("name") or cid
        action = cam.get("latestAction") or "no detections yet"
        pct = cam.get("latestConfidencePct")
        pct_s = f" ({pct}%)" if pct is not None else ""
        actor = cam.get("actorName")
        who = f", {actor}" if actor else ""
        inf = cam.get("inferencesToday", 0)
        non_asm = cam.get("nonAssemblyRatePct", 0)
        actions = cam.get("actionsToday") or {}
        action_bits = ", ".join(f"{k}: {v}" for k, v in list(actions.items())[:4]) if actions else ""
        lines.append(
            f"• {name} [{cid}]: {action}{pct_s}{who} — {inf} inference(s) today, "
            f"{non_asm}% non-assembly"
        )
        if action_bits and re.search(r"action|list|summary", message, re.I):
            lines.append(f"  actions today: {action_bits}")

    har_inc = dash.get("openHarIncidents") or []
    if har_inc:
        lines.append("")
        lines.append("Open HAR incidents:")
        for ev in har_inc[:8]:
            lines.append(
                f"• [{ev.get('cameraId', '?')}] {ev.get('title', ev.get('description', 'incident'))} "
                f"({ev.get('severity', 'info')})"
            )
    elif re.search(r"incident", message, re.I):
        lines.append("")
        lines.append("No open HAR deviation incidents right now.")

    other = dash.get("openIncidents") or []
    if re.search(r"incident", message, re.I) and other and not har_inc:
        lines.append("")
        lines.append("Other open timeline incidents:")
        for ev in other[:5]:
            if ev.get("caseType") != "har_action_deviation":
                lines.append(f"• [{ev.get('cameraId', '?')}] {ev.get('title', 'event')}")

    text = "\n".join(lines)
    if len(text) > max_len:
        text = text[: max_len - 1].rsplit("\n", 1)[0] + "\n…"
    return text
