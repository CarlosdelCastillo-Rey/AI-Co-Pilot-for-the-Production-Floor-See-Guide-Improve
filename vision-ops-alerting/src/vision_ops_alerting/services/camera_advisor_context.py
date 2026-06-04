"""Rich per-camera context and briefing text for the HAR camera advisor."""

from __future__ import annotations

import re
from datetime import timedelta
from typing import Any

from sqlalchemy.orm import Session

from vision_ops_alerting.db.models import Camera, Event, HarActivityLog
from vision_ops_alerting.services.har_activity_store import (
    PRIMARY_ACTION,
    _as_utc_aware,
    _utc_now,
    activity_summary,
    analytics_realtime,
    log_to_dict,
    logs_for_advisor,
)

DEFAULT_WINDOW_MIN = 10


def parse_focus_minutes(message: str, default: int = DEFAULT_WINDOW_MIN) -> int:
    m = re.search(r"last\s+(\d+)\s*(?:min|minutes?)", message, re.I)
    if m:
        return max(1, min(int(m.group(1)), 60))
    if re.search(r"last\s+minutes?|recent(?:ly)?|just\s+now|past\s+few", message, re.I):
        return default
    return default


def _rows_since(
    db: Session,
    *,
    camera_id: str,
    minutes: int,
    session_id: str | None = None,
) -> list[HarActivityLog]:
    since = _utc_now() - timedelta(minutes=max(1, minutes))
    since_naive = since.replace(tzinfo=None)
    q = db.query(HarActivityLog).filter(
        HarActivityLog.camera_id == camera_id,
        HarActivityLog.occurred_at >= since_naive,
    )
    if session_id:
        q = q.filter(HarActivityLog.session_id == session_id)
    return q.order_by(HarActivityLog.occurred_at.asc()).all()


def _window_stats(rows: list[HarActivityLog]) -> dict[str, Any]:
    if not rows:
        return {
            "inferenceCount": 0,
            "empty": True,
            "dominantAction": None,
            "dominantSharePct": 0,
            "avgConfidencePct": None,
            "minConfidencePct": None,
            "maxConfidencePct": None,
            "nonAssemblyCount": 0,
            "labelCounts": {},
            "maxPersonCount": 0,
            "timeSpan": None,
        }

    by_label: dict[str, int] = {}
    confs: list[float] = []
    non_asm = 0
    max_persons = 0
    for r in rows:
        lbl = r.predicted_label or "unknown"
        by_label[lbl] = by_label.get(lbl, 0) + 1
        if r.confidence is not None:
            confs.append(float(r.confidence))
        if not r.is_primary_action:
            non_asm += 1
        max_persons = max(max_persons, r.person_count or 0)

    dominant = max(by_label, key=by_label.get)
    total = len(rows)
    first = _as_utc_aware(rows[0].occurred_at)
    last = _as_utc_aware(rows[-1].occurred_at)
    span = None
    if first and last:
        span = {
            "from": first.isoformat(),
            "to": last.isoformat(),
            "minutes": max(1, int((last - first).total_seconds() / 60) + 1),
        }

    avg = sum(confs) / len(confs) if confs else None
    return {
        "inferenceCount": total,
        "empty": False,
        "dominantAction": dominant,
        "dominantSharePct": round(by_label[dominant] / total * 100, 1),
        "avgConfidencePct": round(avg * 100, 1) if avg is not None else None,
        "minConfidencePct": round(min(confs) * 100, 1) if confs else None,
        "maxConfidencePct": round(max(confs) * 100, 1) if confs else None,
        "nonAssemblyCount": non_asm,
        "labelCounts": by_label,
        "maxPersonCount": max_persons,
        "timeSpan": span,
    }


def _format_log_line(entry: dict[str, Any]) -> str:
    raw_at = entry.get("occurredAt") or ""
    time_part = raw_at[11:19] if len(raw_at) >= 19 else raw_at
    label = entry.get("predictedLabel") or "—"
    conf = entry.get("confidence")
    pct = f"{int(round(float(conf) * 100))}%" if conf is not None else "—"
    persons = entry.get("personCount", 0)
    offset = entry.get("videoOffsetSec")
    off_s = f" @ {offset:.1f}s" if offset is not None else ""
    flag = " [non-assembly]" if entry.get("isPrimaryAction") is False else ""
    return f"{time_part}{off_s}: {label} ({pct}), {persons} person(s){flag}"


def _camera_meta(db: Session, camera_id: str) -> dict[str, Any] | None:
    cam = (
        db.query(Camera)
        .filter((Camera.id == camera_id) | (Camera.backend_camera_id == camera_id))
        .first()
    )
    if not cam:
        return None
    return {
        "id": cam.id,
        "name": cam.name,
        "location": cam.location,
        "zone": cam.zone,
        "status": cam.status,
        "modelId": cam.inference_model,
        "backendCameraId": cam.backend_camera_id,
    }


def _recent_events(db: Session, camera_id: str, *, limit: int = 6) -> list[dict[str, Any]]:
    ids = {camera_id}
    cam = (
        db.query(Camera)
        .filter((Camera.id == camera_id) | (Camera.backend_camera_id == camera_id))
        .first()
    )
    if cam:
        ids.add(cam.id)
        if cam.backend_camera_id:
            ids.add(cam.backend_camera_id)

    rows = (
        db.query(Event)
        .filter(Event.camera_id.in_(list(ids)))
        .order_by(Event.occurred_at.desc())
        .limit(limit)
        .all()
    )
    return [
        {
            "id": e.id,
            "title": e.title,
            "severity": e.severity,
            "caseType": e.case_type,
            "status": e.resolution_status,
            "occurredAt": e.occurred_at.isoformat() if e.occurred_at else None,
            "description": (e.description or "")[:200],
        }
        for e in rows
    ]


def build_camera_advisor_context(
    db: Session,
    *,
    camera_id: str,
    session_id: str | None = None,
    limit: int = 200,
    focus_minutes: int = DEFAULT_WINDOW_MIN,
) -> dict[str, Any]:
    base = logs_for_advisor(db, camera_id=camera_id, limit=limit, session_id=session_id)
    sid = session_id or (base.get("session") or {}).get("id")

    windows: dict[str, Any] = {}
    for mins in (5, 10, 15, 30):
        if mins <= focus_minutes or mins in (5, 10):
            rows = _rows_since(db, camera_id=camera_id, minutes=mins, session_id=sid)
            windows[f"last{mins}Min"] = _window_stats(rows)

    focus_rows = _rows_since(db, camera_id=camera_id, minutes=focus_minutes, session_id=sid)
    realtime = analytics_realtime(db, camera_id=camera_id, minutes=max(focus_minutes, 30))

    logs = base.get("logs") or []
    recent_lines = [_format_log_line(e) for e in logs[-25:]]

    ctx = {
        **base,
        "camera": _camera_meta(db, camera_id),
        "primaryActionLabel": PRIMARY_ACTION,
        "focusWindowMinutes": focus_minutes,
        "windows": windows,
        "focusWindow": _window_stats(focus_rows),
        "realtime": {
            "windowMinutes": realtime.get("windowMinutes"),
            "inferenceCount": realtime.get("inferenceCount"),
            "nonPrimaryCount": realtime.get("nonPrimaryCount"),
            "latest": realtime.get("latest"),
        },
        "recentLogLines": recent_lines,
        "recentEvents": _recent_events(db, camera_id),
    }
    ctx["briefingFacts"] = build_briefing_facts(ctx)
    return ctx


def build_briefing_facts(ctx: dict[str, Any]) -> str:
    """Deterministic fact sheet the LLM (and fallback) must ground answers in."""
    lines: list[str] = []
    cam = ctx.get("camera") or {}
    sess = ctx.get("session") or {}
    summary = ctx.get("summary") or {}
    focus = ctx.get("focusWindow") or {}
    mins = ctx.get("focusWindowMinutes", DEFAULT_WINDOW_MIN)

    name = cam.get("name") or ctx.get("cameraId", "?")
    video = sess.get("videoName") or "current clip"
    lines.append(f"Camera: {name} ({ctx.get('cameraId')}) — video: {video}")
    if cam.get("location"):
        lines.append(f"Location: {cam['location']}" + (f", zone {cam['zone']}" if cam.get("zone") else ""))
    if cam.get("modelId"):
        lines.append(f"HAR model: {cam['modelId']} · status: {cam.get('status', '—')}")

    if focus.get("empty"):
        lines.append(f"Last {mins} minutes: no persisted inference rows (live may still be warming up).")
    else:
        lines.append(
            f"Last {mins} min: {focus['inferenceCount']} inference(s), "
            f"dominant «{focus.get('dominantAction')}» ({focus.get('dominantSharePct')}% of reads), "
            f"avg confidence {focus.get('avgConfidencePct')}% "
            f"(range {focus.get('minConfidencePct')}–{focus.get('maxConfidencePct')}%), "
            f"{focus.get('nonAssemblyCount')} non-assembly read(s), up to {focus.get('maxPersonCount')} person(s) in frame."
        )
        counts = focus.get("labelCounts") or {}
        if len(counts) > 1:
            parts = [f"{k}: {v}" for k, v in sorted(counts.items(), key=lambda x: -x[1])[:6]]
            lines.append("Action mix in window: " + ", ".join(parts))

    today_total = summary.get("totalInferences", 0)
    if today_total:
        lines.append(
            f"Today: {today_total} inference(s), assemble share {summary.get('assembleSharePct', 0)}%, "
            f"non-assembly {summary.get('nonAssemblyRatePct', 0)}%."
        )
        by_label = summary.get("byLabel") or {}
        if by_label:
            today_parts = [f"{k} ({v})" for k, v in sorted(by_label.items(), key=lambda x: -x[1])[:5]]
            lines.append("Today label counts: " + ", ".join(today_parts))

    latest = (ctx.get("realtime") or {}).get("latest")
    if latest:
        conf = latest.get("confidence")
        pct = int(round(float(conf) * 100)) if conf is not None else "—"
        lines.append(
            f"Most recent log: {latest.get('predictedLabel')} at {pct}% "
            f"({latest.get('occurredAt', '')[:19]})"
        )

    events = ctx.get("recentEvents") or []
    if events:
        lines.append("Recent alerts/events on this camera:")
        for ev in events[:4]:
            lines.append(
                f"  - [{ev.get('severity')}] {ev.get('title')} ({ev.get('status')}) "
                f"@ {str(ev.get('occurredAt', ''))[:19]}"
            )
    else:
        lines.append("No timeline alerts tied to this camera in the recent window.")

    tail = ctx.get("recentLogLines") or []
    if tail:
        lines.append("Inference tail (oldest→newest, last entries):")
        for ln in tail[-12:]:
            lines.append(f"  {ln}")

    return "\n".join(lines)


def format_detailed_camera_reply(ctx: dict[str, Any], message: str) -> str:
    """Conversational, data-rich reply when the LLM is unavailable."""
    mins = parse_focus_minutes(message, default=ctx.get("focusWindowMinutes", DEFAULT_WINDOW_MIN))
    if ctx.get("focusWindowMinutes") != mins:
        ctx = {**ctx, "focusWindowMinutes": mins}
        ctx["briefingFacts"] = build_briefing_facts(ctx)

    facts = ctx.get("briefingFacts") or build_briefing_facts(ctx)
    cam = ctx.get("camera") or {}
    name = cam.get("name") or ctx.get("cameraId", "this feed")
    focus = ctx.get("focusWindow") or {}
    sess = ctx.get("session") or {}
    video = sess.get("videoName") or "the looped clip"

    if not (ctx.get("logs") or focus.get("inferenceCount")):
        return (
            f"I'm looking at {name} ({video}) but there are no saved inference rows yet. "
            "If Live shows predictions, confirm alerting (:8001) is up and HAR activity POSTs are succeeding. "
            "Once logs appear, ask again for a minute-by-minute breakdown."
        )

    parts: list[str] = []
    if not message.strip() or message.lower() in ("hi", "hello", "help"):
        parts.append(
            f"Hey — I'm your camera copilot for {name}. I read the HAR log, clip metadata, and any alerts on this feed. "
            f'Ask things like "summary of the last 10 minutes" or "what changed after the confidence dropped?"'
        )
    elif re.search(r"summar|last\s+\d+|last\s+min|recent", message, re.I):
        parts.append(f"Here's what happened on {name} over the last ~{mins} minutes on {video}:")
    else:
        parts.append(f"On {name} ({video}), here's what the logs show for your question:")

    if focus.get("empty"):
        parts.append(
            f"No inferences were stored in the last {mins} minutes. "
            "Check that the feed is playing and the backend is posting to /api/har/activity."
        )
    else:
        dom = focus.get("dominantAction", "—")
        share = focus.get("dominantSharePct", 0)
        avg = focus.get("avgConfidencePct")
        lo, hi = focus.get("minConfidencePct"), focus.get("maxConfidencePct")
        n = focus.get("inferenceCount", 0)
        non_asm = focus.get("nonAssemblyCount", 0)
        persons = focus.get("maxPersonCount", 0)

        parts.append(
            f"We recorded {n} inference(s). The line was mostly «{dom}» ({share}% of reads). "
            f"Confidence averaged {avg}% (spread {lo}%–{hi}%). "
            f"{'That is quite low — treat the label as weak evidence until confidence rises.' if avg is not None and avg < 25 else 'Confidence is in a normal band for this model.'}"
        )
        if non_asm:
            parts.append(
                f"{non_asm} read(s) were flagged non-assembly (not «{PRIMARY_ACTION}»). "
                "Worth watching if that count climbs."
            )
        else:
            parts.append(f"Everything in this window matched primary assembly («{PRIMARY_ACTION}») — no deviation flags in-window.")

        if persons:
            parts.append(f"Up to {persons} person(s) detected in frame during the window.")

        counts = focus.get("labelCounts") or {}
        if len(counts) > 1:
            mix = ", ".join(f"«{k}» ×{v}" for k, v in sorted(counts.items(), key=lambda x: -x[1]))
            parts.append(f"Full action mix: {mix}.")

        tail = ctx.get("recentLogLines") or []
        if tail:
            parts.append("Latest log lines: " + " → ".join(tail[-4:]))

    events = ctx.get("recentEvents") or []
    open_ev = [e for e in events if (e.get("status") or "OPEN") == "OPEN"]
    if open_ev:
        parts.append(
            "Open alert(s) on this camera: "
            + "; ".join(f"{e.get('title')} ({e.get('severity')})" for e in open_ev[:3])
            + "."
        )
    elif events:
        parts.append("No open alerts; latest closed items are in the timeline if you need audit trail.")

    summary = ctx.get("summary") or {}
    if summary.get("totalInferences"):
        parts.append(
            f"Shift context (today): {summary['totalInferences']} total reads, "
            f"{summary.get('nonAssemblyRatePct', 0)}% non-assembly rate."
        )

    parts.append(
        "\n(Data from VisionOps HAR activity store. Replies use live logs — if this looks stale, hit Fresh start or wait for new inferences.)"
    )
    return "\n\n".join(parts)
