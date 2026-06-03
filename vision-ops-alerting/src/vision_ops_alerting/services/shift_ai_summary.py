"""Deterministic shift AI summary for timeline sidebar and post-shift PDF."""

from __future__ import annotations

from datetime import date, datetime, timezone

from sqlalchemy.orm import Session

from vision_ops_alerting.db.models import Event
from vision_ops_alerting.services.event_workflow import build_timeline_stats
from vision_ops_alerting.services.industrial_analytics import (
    build_recommendation,
    compute_coq,
    compute_flow_efficiency,
    compute_oee,
    compute_pareto,
)
from vision_ops_alerting.services.plant_settings import get_plant_config
from vision_ops_alerting.services.har_activity_store import global_har_summary
from vision_ops_alerting.services.timeline_summary import build_shift_summary


def _current_shift_key() -> str:
    hour = datetime.now(timezone.utc).hour
    if 6 <= hour < 14:
        return "morning"
    if 14 <= hour < 22:
        return "evening"
    return "night"


def _shift_letter() -> str:
    hour = datetime.now(timezone.utc).hour
    if 6 <= hour < 14:
        return "A"
    if 14 <= hour < 22:
        return "B"
    return "C"


def _events_for_day(db: Session, target: str) -> list[Event]:
    start = datetime.fromisoformat(f"{target}T00:00:00+00:00")
    end = datetime.fromisoformat(f"{target}T23:59:59+00:00")
    return (
        db.query(Event)
        .filter(Event.occurred_at >= start, Event.occurred_at <= end)
        .order_by(Event.occurred_at.desc())
        .all()
    )


def _severity_counts(events: list[Event]) -> dict[str, int]:
    counts = {"critical": 0, "warning": 0, "info": 0, "normal": 0}
    for e in events:
        sev = (e.severity or "info").lower()
        if sev in counts:
            counts[sev] += 1
    return counts


def build_shift_ai_summary(db: Session, event_date: str | None = None) -> dict:
    target = event_date or date.today().isoformat()
    shift = _current_shift_key()
    plant = get_plant_config(db)
    summary = build_shift_summary(db, target)
    stats = build_timeline_stats(db, target)
    events = _events_for_day(db, target)
    sev = _severity_counts(events)

    oee_row = compute_oee(db, target, shift, None)
    coq = compute_coq(db, target, shift, None)
    flow = compute_flow_efficiency(oee_row)
    pareto = compute_pareto(db, target, shift, None)
    primary_rec = build_recommendation(db, target, None)

    open_count = stats.get("openCount", 0)
    open_critical = stats.get("openCriticalCount", 0)
    ack_count = stats.get("acknowledgedCount", 0)
    resolved_count = stats.get("resolvedCount", 0)
    false_pos = stats.get("falsePositiveCount", 0)
    total = summary.get("totalEvents") or len(events)
    uptime = summary.get("uptime", "—")
    all_clear = stats.get("allClear", True)

    if open_critical > 0:
        status_level = "critical"
        headline = "Action required"
        detail = f"{open_critical} critical incident(s) still open — prioritize triage before handoff"
    elif open_count > 0:
        status_level = "action_needed"
        headline = "Triage pending"
        detail = f"{open_count} incident(s) awaiting acknowledge or resolve"
    else:
        status_level = "all_clear"
        headline = "All clear"
        detail = "No open critical incidents at shift close"

    narrative_parts = [
        f"Shift review for {plant.site_name} on {summary.get('date', target)}: "
        f"{total} vision event(s) logged with {uptime} estimated uptime."
    ]
    if resolved_count:
        narrative_parts.append(
            f"{resolved_count} incident(s) closed on-shift"
            + (f" and {false_pos} marked false positive" if false_pos else "")
            + "."
        )
    if ack_count and open_count:
        narrative_parts.append(
            f"{ack_count} acknowledged and {open_count} still open in workflow."
        )
    elif ack_count:
        narrative_parts.append(f"{ack_count} incident(s) acknowledged and under review.")

    narrative = " ".join(narrative_parts)

    highlights: list[str] = []
    if sev["critical"]:
        highlights.append(f"{sev['critical']} critical alert(s) recorded today")
    if sev["warning"]:
        highlights.append(f"{sev['warning']} warning-level event(s) on the floor")
    highlights.append(f"OEE {oee_row['oee']:.1f}% · flow efficiency {flow:.1f}%")
    if coq.get("totalCostUsd", 0) > 0:
        highlights.append(f"Estimated cost of quality ${coq['totalCostUsd']:,.2f} (downtime + scrap)")
    if stats.get("avgAckSeconds") is not None:
        ack_s = stats["avgAckSeconds"]
        highlights.append(
            f"Avg time-to-ack {int(ack_s)}s" if ack_s < 60 else f"Avg time-to-ack {round(ack_s / 60)}m"
        )
    if summary.get("assets"):
        top = summary["assets"][0]
        highlights.append(f"Highest activity on {top['name']} ({top['events']} events)")

    har_plant = global_har_summary(db, hours=24)
    if har_plant.get("totalInferences", 0) > 0:
        highlights.append(
            f"HAR live: {har_plant['totalInferences']} inference(s), "
            f"{har_plant.get('nonPrimaryCount', 0)} non-assembly detection(s)"
        )
    har_deviations = sum(1 for e in events if e.case_type == "har_action_deviation")
    if har_deviations:
        highlights.append(f"{har_deviations} HAR deviation(s) promoted to Timeline today")

    suggestions: list[str] = []
    if open_critical:
        open_crit_events = [
            e for e in events
            if (e.resolution_status or "OPEN") == "OPEN" and e.severity == "critical"
        ]
        seen_suggestions: set[str] = set()
        for e in open_crit_events[:5]:
            cam = e.camera_id or e.line_id or "floor"
            msg = f"Acknowledge and resolve '{e.title}' on {cam}"
            if msg in seen_suggestions:
                continue
            seen_suggestions.add(msg)
            suggestions.append(msg)
            if len(suggestions) >= 3:
                break
        suggestions.append("Escalate unresolved critical items to shift supervisor before sign-off")
    elif open_count:
        suggestions.append("Clear the open queue — acknowledge then resolve or tag false positive")
        suggestions.append("Add reason codes and downtime minutes for COQ traceability")
    else:
        suggestions.append("Run a quick camera health check before next shift start")
        suggestions.append("Review alert thresholds if false-positive rate trends up")

    recommendations: list[str] = [primary_rec]
    if pareto.get("items"):
        top = pareto["items"][0]
        recommendations.append(
            f"Root-cause focus: '{top['label']}' accounts for {top['pct']}% of tagged closures — review SOP for {top['code']}"
        )
    if oee_row["oee"] < 75:
        recommendations.append(
            f"OEE below target ({oee_row['oee']:.1f}%) — inspect availability ({oee_row['availability']:.1f}%) "
            f"and quality ({oee_row['quality']:.1f}%) drivers"
        )
    if false_pos >= 2:
        recommendations.append(
            f"{false_pos} false positives today — tune vision rules or idle thresholds on noisy cameras"
        )
    if not recommendations:
        recommendations.append("Maintain current run rates and monitor shift handoff checklist")

    return {
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "date": target,
        "shift": shift,
        "shiftLabel": f"SHIFT {_shift_letter()}-{datetime.strptime(target, '%Y-%m-%d').day}",
        "siteName": plant.site_name,
        "currentStatus": status_level,
        "statusHeadline": headline,
        "statusDetail": detail,
        "allClear": all_clear,
        "narrative": narrative,
        "highlights": highlights[:6],
        "suggestions": suggestions[:5],
        "recommendations": recommendations[:5],
        "metrics": {
            "totalEvents": total,
            "openCount": open_count,
            "openCriticalCount": open_critical,
            "acknowledgedCount": ack_count,
            "resolvedCount": resolved_count,
            "falsePositiveCount": false_pos,
            "uptime": uptime,
            "uptimePct": summary.get("uptimePct"),
            "oee": oee_row["oee"],
            "availability": oee_row["availability"],
            "performance": oee_row["performance"],
            "quality": oee_row["quality"],
            "flowEfficiency": flow,
            "coqTotalUsd": coq.get("totalCostUsd", 0),
            "avgAckSeconds": stats.get("avgAckSeconds"),
        },
        "severityCounts": sev,
    }
