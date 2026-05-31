from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from vision_ops_alerting.auth_deps import get_current_user
from vision_ops_alerting.db.models import EmailTemplateRecord, User
from vision_ops_alerting.db.session import get_db
from vision_ops_alerting.routers.notifications import TEST_CONTEXTS
from vision_ops_alerting.schemas import CaseType
from vision_ops_alerting.services.email_templates import (
    clone_template,
    render_email_template,
    resolve_template,
    template_record_to_dict,
)

router = APIRouter(prefix="/api/alerts/email-templates", tags=["email-templates"])


class EmailTemplateCreate(BaseModel):
    name: str
    base_template_id: str = Field(alias="baseTemplateId")
    subject: str | None = None
    headline: str | None = None
    body: str | None = None
    category: str | None = None
    footer_reason: str | None = Field(default=None, alias="footerReason")
    enabled: bool = True

    model_config = {"populate_by_name": True}


class EmailTemplateUpdate(BaseModel):
    name: str | None = None
    subject: str | None = None
    headline: str | None = None
    body: str | None = None
    category: str | None = None
    footer_reason: str | None = Field(default=None, alias="footerReason")
    severity_level: str | None = Field(default=None, alias="severityLevel")
    enabled: bool | None = None

    model_config = {"populate_by_name": True}


@router.get("")
def list_email_templates(
    db: Session = Depends(get_db),
    case_type: str | None = Query(None, alias="caseType"),
):
    q = db.query(EmailTemplateRecord).order_by(
        EmailTemplateRecord.is_builtin.desc(),
        EmailTemplateRecord.name.asc(),
    )
    if case_type:
        q = q.filter(EmailTemplateRecord.case_type == case_type)
    return [template_record_to_dict(r) for r in q.all()]


@router.get("/{template_id}")
def get_email_template(template_id: str, db: Session = Depends(get_db)):
    row = db.get(EmailTemplateRecord, template_id)
    if not row:
        raise HTTPException(status_code=404, detail="Template not found")
    return template_record_to_dict(row)


@router.post("", status_code=201)
def create_email_template(body: EmailTemplateCreate, db: Session = Depends(get_db), _user: User = Depends(get_current_user)):
    row = clone_template(db, body.base_template_id, name=body.name, subject=body.subject)
    if not row:
        raise HTTPException(status_code=404, detail="Base template not found")
    if body.headline is not None:
        row.headline = body.headline
    if body.body is not None:
        row.body = body.body
    if body.category is not None:
        row.category = body.category
    if body.footer_reason is not None:
        row.footer_reason = body.footer_reason
    row.enabled = body.enabled
    row.updated_at = datetime.now(timezone.utc)
    db.commit()
    db.refresh(row)
    return template_record_to_dict(row)


@router.patch("/{template_id}")
def update_email_template(template_id: str, body: EmailTemplateUpdate, db: Session = Depends(get_db), _user: User = Depends(get_current_user)):
    row = db.get(EmailTemplateRecord, template_id)
    if not row:
        raise HTTPException(status_code=404, detail="Template not found")
    if row.is_builtin and any(
        v is not None for v in (body.subject, body.headline, body.body, body.category, body.severity_level)
    ):
        raise HTTPException(status_code=400, detail="Built-in templates cannot edit content. Clone to customize.")

    for field, attr in [
        (body.name, "name"),
        (body.subject, "subject"),
        (body.headline, "headline"),
        (body.body, "body"),
        (body.category, "category"),
        (body.footer_reason, "footer_reason"),
        (body.severity_level, "severity_level"),
        (body.enabled, "enabled"),
    ]:
        if field is not None:
            setattr(row, attr, field)

    row.updated_at = datetime.now(timezone.utc)
    db.commit()
    db.refresh(row)
    return template_record_to_dict(row)


@router.delete("/{template_id}", status_code=204)
def delete_email_template(template_id: str, db: Session = Depends(get_db), _user: User = Depends(get_current_user)):
    row = db.get(EmailTemplateRecord, template_id)
    if not row:
        raise HTTPException(status_code=404, detail="Template not found")
    if row.is_builtin:
        raise HTTPException(status_code=400, detail="Built-in templates cannot be deleted")
    db.delete(row)
    db.commit()


@router.post("/{template_id}/preview")
def preview_email_template(template_id: str, db: Session = Depends(get_db)):
    row = db.get(EmailTemplateRecord, template_id)
    if not row:
        raise HTTPException(status_code=404, detail="Template not found")
    case_type: CaseType = row.case_type  # type: ignore[assignment]
    ctx = TEST_CONTEXTS.get(case_type, TEST_CONTEXTS["unknown"])
    rendered = render_email_template(row, ctx)
    return {
        "templateId": row.id,
        "subject": rendered.subject,
        "html": rendered.html,
        "text": rendered.text,
    }
