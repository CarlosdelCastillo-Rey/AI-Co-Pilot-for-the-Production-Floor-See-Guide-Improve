"""Page-aware copy and screen context for the VisionOps advisor."""

from __future__ import annotations

import re
from typing import Any

from sqlalchemy.orm import Session

from vision_ops_alerting.config import settings
from vision_ops_alerting.db.models import AlertRule, EmailTemplateRecord

# What the user is looking at — drives advice even when the floor is quiet.
PAGE_SCREEN: dict[str, dict[str, str]] = {
    "live": {
        "label": "Live Streams",
        "focus": "camera health, live HAR inference, batch model probes on mock clips, and edge latency",
        "actions": "summarize cam-har feeds, interpret probe results, and decide if an event needs escalation",
    },
    "timeline": {
        "label": "Timeline",
        "focus": "incident history, acknowledge/resolve workflow, and shift handoff",
        "actions": "prioritize open items, close loops with reason codes, and reduce downtime minutes",
    },
    "analytics": {
        "label": "Analytics",
        "focus": "OEE, COQ, heatmaps, bottlenecks, and shift KPI trends",
        "actions": "tie metrics to open incidents and find false-green dashboards",
    },
    "alerts": {
        "label": "Alerts & Rules",
        "focus": "alert rules, case types, email templates, and notification delivery",
        "actions": "enable the right rules, match templates to case types, and verify email is not in dry-run",
    },
    "settings": {
        "label": "Plant Settings",
        "focus": "plant KPI parameters, cost assumptions, and uptime formulas",
        "actions": "align cost/minute and shift hours with finance and ops reporting",
    },
    "har-hitl": {
        "label": "Person HITL",
        "focus": "HAR session review, person registry, track labels, and human corrections",
        "actions": "prioritize uncertain tracks, fix mislabels, and keep the person registry accurate",
    },
    "dashboard": {
        "label": "Dashboard",
        "focus": "overall plant status across live, timeline, and analytics",
        "actions": "navigate to the right screen for the current risk",
    },
}

GREETING_RE = re.compile(
    r"^(hi|hello|hey|hola|howdy|good\s+(morning|afternoon|evening)|what'?s\s+up|yo)[\s!.?]*$",
    re.I,
)


def is_greeting(message: str) -> bool:
    text = message.strip()
    if not text or len(text) > 40:
        return False
    return bool(GREETING_RE.match(text))


def page_meta(page: str, page_title: str | None = None) -> dict[str, str]:
    meta = PAGE_SCREEN.get(page, PAGE_SCREEN["dashboard"])
    label = page_title or meta["label"]
    return {**meta, "label": label, "pageId": page}


def enrich_snapshot_for_page(
    db: Session,
    snapshot: dict[str, Any],
    page: str,
    *,
    camera_id: str | None = None,
) -> dict[str, Any]:
    """Add screen-specific DB facts so the model knows what the user is viewing."""
    screen: dict[str, Any] = page_meta(page)

    if page == "alerts":
        rules = db.query(AlertRule).order_by(AlertRule.title).all()
        templates = (
            db.query(EmailTemplateRecord)
            .filter(EmailTemplateRecord.enabled.is_(True))
            .count()
        )
        screen["rules"] = [
            {
                "title": r.title,
                "caseType": r.case_type,
                "enabled": r.enabled,
                "notifyEmail": r.notify_email,
                "severity": r.severity,
            }
            for r in rules[:12]
        ]
        screen["enabledRules"] = sum(1 for r in rules if r.enabled)
        screen["totalRules"] = len(rules)
        screen["emailTemplatesEnabled"] = templates
        screen["emailDryRun"] = settings.dry_run

    if page == "live" or page == "analytics":
        from vision_ops_alerting.services.har_activity_store import (
            activity_summary,
            analytics_realtime,
            build_all_cameras_har_dashboard,
            slim_har_dashboard_for_advisor,
        )

        screen["harLiveDashboard"] = slim_har_dashboard_for_advisor(
            build_all_cameras_har_dashboard(db, hours=24)
        )
        if camera_id and camera_id.startswith("cam-har"):
            screen["harCameraId"] = camera_id
            screen["harSummary"] = activity_summary(db, camera_id=camera_id)
            screen["harRealtime"] = analytics_realtime(db, camera_id=camera_id, minutes=30)

    snapshot["currentScreen"] = screen
    return snapshot


def build_welcome(db: Session, *, page: str, page_title: str | None = None) -> dict[str, Any]:
    from vision_ops_alerting.services.operational_snapshot import build_operational_snapshot

    meta = page_meta(page, page_title)
    snapshot = enrich_snapshot_for_page(db, build_operational_snapshot(db, page=page), page)
    floor = snapshot.get("floor") or {}

    intro = (
        "I'm VisionOps AI — your ops advisor. I read live plant data (cameras, open alerts, rules, KPIs) "
        "and tailor answers to the screen you're on. Ask for safety, throughput, cost impact, or what to do next."
    )

    page_line = (
        f"You're on **{meta['label']}** — I focus on {meta['focus']}. "
        f"I can help you {meta['actions']}."
    ).replace("**", "")

    status_parts: list[str] = []
    if page == "alerts":
        rules = snapshot.get("currentScreen", {})
        active = rules.get("enabledRules", snapshot.get("alertRules", {}).get("active", 0))
        total = rules.get("totalRules", snapshot.get("alertRules", {}).get("total", 0))
        dry = "on" if snapshot.get("emailDryRun") else "off"
        status_parts.append(f"{active}/{total} alert rules enabled; email dry-run is {dry}.")
        if floor.get("openCriticalCount"):
            status_parts.append(
                f"Floor has {floor['openCriticalCount']} open critical incident(s)—handle those on Timeline first."
            )
        elif floor.get("openCount"):
            status_parts.append(f"{floor['openCount']} open incident(s) on the floor.")
        else:
            status_parts.append("No open floor incidents—good time to tune rules and test email templates.")
    elif page == "live":
        cams = snapshot.get("cameras", {})
        status_parts.append(f"{cams.get('live', 0)}/{cams.get('total', 0)} cameras live.")
        if floor.get("openCriticalCount"):
            status_parts.append(f"{floor['openCriticalCount']} critical alert(s) need attention.")
    elif page == "analytics":
        status_parts.append(f"Today's uptime {floor.get('uptime', '—')}; {floor.get('totalEventsToday', 0)} events logged.")
    elif page == "timeline":
        if floor.get("openCount"):
            status_parts.append(f"{floor['openCount']} open item(s) in workflow.")
        else:
            status_parts.append("Workflow queue is clear for today.")
    else:
        if floor.get("openCriticalCount"):
            status_parts.append(f"{floor['openCriticalCount']} critical alert(s) on the floor.")
        elif floor.get("allClear"):
            status_parts.append("Floor alerts are clear.")

    quick = _quick_prompts_for_page(page)

    return {
        "intro": intro,
        "pageContext": page_line,
        "status": " ".join(status_parts) if status_parts else "Plant snapshot loaded.",
        "welcome": f"{intro} {page_line} {' '.join(status_parts)}".strip(),
        "quickPrompts": quick,
        "page": page,
        "pageTitle": meta["label"],
    }


def _quick_prompts_for_page(page: str) -> list[str]:
    prompts: dict[str, list[str]] = {
        "alerts": [
            "Are my rules configured correctly?",
            "Should dry-run be off for production?",
            "Which case type should I enable next?",
        ],
        "live": [
            "Summary of all cameras and latest actions",
            "List HAR incidents by camera",
            "Which feed needs attention?",
        ],
        "timeline": [
            "What should I resolve first?",
            "Summarize open incidents.",
        ],
        "analytics": [
            "Is OEE aligned with open alerts?",
            "Biggest cost driver today?",
        ],
        "settings": [
            "Are plant cost assumptions reasonable?",
        ],
    }
    return prompts.get(page, ["What needs attention now?", "Summarize plant status."])
