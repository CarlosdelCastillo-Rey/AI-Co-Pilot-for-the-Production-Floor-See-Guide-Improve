from __future__ import annotations

from dataclasses import dataclass

from vision_ops_alerting.schemas import CaseType, IndustrialContext, Severity


@dataclass(frozen=True)
class EmailTemplate:
    template_id: str
    subject: str
    text: str
    html: str
    severity: Severity


def _render_common(ctx: IndustrialContext) -> dict[str, str]:
    actor = ctx.actor.name or ctx.actor.track_id or "unknown"
    return {
        "site_id": ctx.site_id,
        "line_id": ctx.line_id,
        "camera_id": ctx.camera_id,
        "timestamp": ctx.timestamp,
        "actor": actor,
        "live_url": ctx.links.live_url or "",
        "timeline_url": ctx.links.timeline_url or "",
        "idle_seconds": str(ctx.evidence.idle_seconds or ""),
        "roi": ctx.evidence.roi or "",
    }


TEMPLATES: dict[CaseType, EmailTemplate] = {
    "user_not_working": EmailTemplate(
        template_id="user_not_working_v1",
        subject="VisionOps: Operator idle ({{line_id}} / {{camera_id}})",
        text="Operator idle alert.",
        html="<p>Operator idle</p>",
        severity="warning",
    ),
    "user_left_position": EmailTemplate(
        template_id="user_left_position_v1",
        subject="VisionOps: Operator left position",
        text="Operator left position.",
        html="<p>Operator left position</p>",
        severity="critical",
    ),
    "forklift_in_zone": EmailTemplate(
        template_id="forklift_in_zone_v1",
        subject="VisionOps: Forklift in zone",
        text="Forklift in zone.",
        html="<p>Forklift in zone</p>",
        severity="critical",
    ),
    "unknown": EmailTemplate(
        template_id="unknown_v1",
        subject="VisionOps: Alert",
        text="VisionOps alert.",
        html="<p>VisionOps alert</p>",
        severity="info",
    ),
}


def render_template(template: EmailTemplate, ctx: IndustrialContext) -> EmailTemplate:
    vals = _render_common(ctx)

    def sub(s: str) -> str:
        for k, v in vals.items():
            s = s.replace(f"{{{{{k}}}}}", v)
        return s

    return EmailTemplate(
        template_id=template.template_id,
        subject=sub(template.subject),
        text=sub(template.text),
        html=sub(template.html),
        severity=template.severity,
    )
