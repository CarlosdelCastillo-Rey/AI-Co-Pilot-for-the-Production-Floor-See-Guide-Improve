from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

import uuid

from sqlalchemy.orm import Session

from vision_ops_alerting.db.models import EmailTemplateRecord, new_id
from vision_ops_alerting.email_layout import build_email_html, build_email_text, build_template_vars, substitute
from vision_ops_alerting.schemas import CaseType, IndustrialContext, Severity
from vision_ops_alerting.services.alert_snapshot import SnapshotAsset, snapshot_url_for_email
from vision_ops_alerting.templates import EmailTemplate


@dataclass(frozen=True)
class BuiltinEmailSpec:
    id: str
    slug: str
    name: str
    case_type: CaseType
    category: str
    severity_level: str
    headline: str
    body: str
    subject: str
    footer_reason: str
    layout: str
    snapshot_url: str = ""


BUILTIN_SPECS: list[BuiltinEmailSpec] = [
    BuiltinEmailSpec(
        id="operator.idle",
        slug="operator.idle",
        name="Operator Idle",
        case_type="user_not_working",
        category="Productivity · idle",
        severity_level="warning",
        headline="Station idle for {{idle_duration}}",
        body="No activity detected at **{{line_id}}** while **{{actor}}** is on station. V-JEPA motion score held below the idle threshold.",
        subject="VisionOps: Operator idle — {{line_id}} / {{camera_id}}",
        footer_reason="Productivity alerts",
        layout="idle_stats",
    ),
    BuiltinEmailSpec(
        id="operator.left_position",
        slug="operator.left_position",
        name="Operator Left Position",
        case_type="user_left_position",
        category="Operator event",
        severity_level="warning",
        headline="Operator left position",
        body="**{{actor}}** left the monitored station at **{{line_id}}** and has not returned. Camera **{{camera_id}}** flagged the vacancy.",
        subject="VisionOps: Operator left position — {{line_id}} / {{camera_id}}",
        footer_reason="Operator events",
        layout="standard",
        snapshot_url="https://lh3.googleusercontent.com/aida-public/AB6AXuD0WaxmzB30i4UvnXB1kC5UAjuno45jZ0-lYANMKEPRwQpqZH639_Ac7yPq9EJwxynwUc8jWfLtP6TuMgSHCc4R8QV2j8GXrckY0OSBfzsbliQXwp7qGaM_dgRn_CJ_-YN2M84FIR_4mTkNuSeOUqYZv-zFb-PGGtCtruaN4-mtsBu_sa6AvIDb6JnHCMjexxBix8FdInNk_8IbvGQsjiq1a0uDuXJABCY-cv8XDEYCM9YPSBwMnKCs_vAm8Ksn_xYUDxWv9393I",
    ),
    BuiltinEmailSpec(
        id="forklift.zone_intrusion",
        slug="forklift.zone_intrusion",
        name="Forklift Zone Intrusion",
        case_type="forklift_in_zone",
        category="Safety · geofence",
        severity_level="critical",
        headline="Forklift in restricted zone",
        body="**{{actor}}** entered restricted **{{zone}}** at **{{line_id}}**. Safety boundary breached — review immediately.",
        subject="VisionOps: Forklift in restricted zone — {{line_id}} / {{camera_id}}",
        footer_reason="Critical safety alerts",
        layout="standard",
        snapshot_url="https://lh3.googleusercontent.com/aida-public/AB6AXuD0WaxmzB30i4UvnXB1kC5UAjuno45jZ0-lYANMKEPRwQpqZH639_Ac7yPq9EJwxynwUc8jWfLtP6TuMgSHCc4R8QV2j8GXrckY0OSBfzsbliQXwp7qGaM_dgRn_CJ_-YN2M84FIR_4mTkNuSeOUqYZv-zFb-PGGtCtruaN4-mtsBu_sa6AvIDb6JnHCMjexxBix8FdInNk_8IbvGQsjiq1a0uDuXJABCY-cv8XDEYCM9YPSBwMnKCs_vAm8Ksn_xYUDxWv9393I",
    ),
    BuiltinEmailSpec(
        id="generic.event",
        slug="generic.event",
        name="Generic Vision Alert",
        case_type="unknown",
        category="Vision event",
        severity_level="info",
        headline="VisionOps alert",
        body="An autonomous vision event was detected on **{{camera_id}}** at **{{line_id}}**. Review the live feed and timeline for details.",
        subject="VisionOps: Alert — {{line_id}} / {{camera_id}}",
        footer_reason="Vision alerts",
        layout="standard",
    ),
    BuiltinEmailSpec(
        id="har.action_detected",
        slug="har.action_detected",
        name="HAR Action Detected",
        case_type="unknown",
        category="HAR · live detection",
        severity_level="info",
        headline="{{action_label}} — {{confidence_pct}}% confidence",
        body=(
            "Live HAR detected **{{action_label}}** at **{{confidence_pct}}%** on **{{camera_id}}** "
            "(model **{{model_id}}**)."
        ),
        subject="VisionOps: HAR action — {{action_label}} ({{confidence_pct}}%) · {{camera_id}}",
        footer_reason="HAR live action notifications",
        layout="har_logs",
    ),
    BuiltinEmailSpec(
        id="har.action_deviation",
        slug="har.action_deviation",
        name="HAR Action Deviation",
        case_type="unknown",
        category="HAR · deviation",
        severity_level="warning",
        headline="Non-assembly detected — {{action_label}} ({{confidence_pct}}%)",
        body=(
            "Camera **{{camera_id}}** detected **{{action_label}}** at **{{confidence_pct}}%** confidence. "
            "This deviates from the expected primary assembly task."
        ),
        subject="VisionOps: HAR deviation — {{action_label}} ({{confidence_pct}}%) · {{camera_id}}",
        footer_reason="HAR deviation alerts",
        layout="standard",
    ),
]


def _severity_from_level(level: str) -> Severity:
    return {"critical": "critical", "warning": "warning"}.get(level, "info")  # type: ignore[return-value]


def template_record_to_dict(row: EmailTemplateRecord) -> dict:
    return {
        "id": row.id,
        "slug": row.slug,
        "name": row.name,
        "caseType": row.case_type,
        "category": row.category,
        "severityLevel": row.severity_level,
        "headline": row.headline,
        "body": row.body,
        "subject": row.subject,
        "footerReason": row.footer_reason,
        "layout": row.layout,
        "snapshotUrl": row.snapshot_url,
        "isBuiltin": row.is_builtin,
        "enabled": row.enabled,
    }


def ensure_builtin_templates(db: Session) -> None:
    for spec in BUILTIN_SPECS:
        existing = db.get(EmailTemplateRecord, spec.id)
        if existing:
            continue
        db.add(
            EmailTemplateRecord(
                id=spec.id,
                slug=spec.slug,
                name=spec.name,
                case_type=spec.case_type,
                category=spec.category,
                severity_level=spec.severity_level,
                headline=spec.headline,
                body=spec.body,
                subject=spec.subject,
                footer_reason=spec.footer_reason,
                layout=spec.layout,
                snapshot_url=spec.snapshot_url or None,
                is_builtin=True,
                enabled=True,
            )
        )


def resolve_template(
    db: Session,
    case_type: CaseType,
    template_id: str | None = None,
) -> EmailTemplateRecord | None:
    if template_id:
        row = db.get(EmailTemplateRecord, template_id)
        if row and row.enabled:
            return row
    return (
        db.query(EmailTemplateRecord)
        .filter(
            EmailTemplateRecord.case_type == case_type,
            EmailTemplateRecord.enabled.is_(True),
        )
        .order_by(EmailTemplateRecord.is_builtin.desc(), EmailTemplateRecord.created_at.asc())
        .first()
    )


def render_email_template(
    row: EmailTemplateRecord,
    ctx: IndustrialContext,
    *,
    snapshot_asset: SnapshotAsset | None = None,
) -> EmailTemplate:
    snapshot = snapshot_url_for_email(snapshot_asset, row.snapshot_url or "")
    values = build_template_vars(ctx, footer_reason=row.footer_reason, snapshot_url=snapshot)
    line_display = values["line_id"]
    if row.case_type == "forklift_in_zone" and values.get("zone"):
        line_display = f'{values["line_id"]} · {values["zone"]}'

    subject = substitute(row.subject, values)
    text = build_email_text(category=row.category, headline=row.headline, body=row.body, values=values)
    html = build_email_html(
        slug=row.slug,
        category=row.category,
        headline=row.headline,
        body=row.body,
        severity_level=row.severity_level,
        values=values,
        layout=row.layout,
        line_display=line_display,
    )
    return EmailTemplate(
        template_id=row.id,
        subject=subject,
        text=text,
        html=html,
        severity=_severity_from_level(row.severity_level),
    )


def clone_template(db: Session, base_id: str, *, name: str, subject: str | None = None) -> EmailTemplateRecord | None:
    base = db.get(EmailTemplateRecord, base_id)
    if not base:
        return None
    row = EmailTemplateRecord(
        id=new_id("etpl"),
        slug=f"custom.{uuid.uuid4().hex[:8]}",
        name=name,
        case_type=base.case_type,
        category=base.category,
        severity_level=base.severity_level,
        headline=base.headline,
        body=base.body,
        subject=subject or base.subject,
        footer_reason=base.footer_reason,
        layout=base.layout,
        snapshot_url=base.snapshot_url,
        is_builtin=False,
        enabled=True,
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
    )
    db.add(row)
    db.flush()
    return row
