from __future__ import annotations

import json
from dataclasses import dataclass

from mailersend import MailerSendClient, EmailBuilder
from strands import Agent
from strands.models.ollama import OllamaModel

from vision_ops_alerting.config import settings
from vision_ops_alerting.schemas import CaseType, IndustrialContext, Severity
from vision_ops_alerting.templates import EmailTemplate, TEMPLATES, render_template


@dataclass(frozen=True)
class ClassifiedCase:
    case_type: CaseType
    severity: Severity
    template_id: str


CLASSIFIER_SYSTEM_PROMPT = """\
You are a safety/operations alert classifier for an industrial vision system.

You MUST classify the given JSON payload into exactly one of these case types:
- user_not_working
- user_left_position
- forklift_in_zone
- unknown

Rules:
- Prefer using explicit signals in the payload (case_type, evidence.idle_seconds, actor.type, evidence.roi).
- If evidence.idle_seconds exists and is >= 300 and actor.type is operator => user_not_working.
- If actor.type is operator and payload indicates absence/left position (may be in free-text fields) => user_left_position.
- If actor.type is forklift OR evidence.roi mentions restricted/forklift zone => forklift_in_zone.
- If uncertain => unknown.

Output MUST be valid JSON with keys: case_type, severity.
severity MUST be one of: info, warning, critical.
"""


def _fallback_classify(ctx: IndustrialContext) -> ClassifiedCase:
    # Deterministic fallback for when no model is available / local model not running.
    if ctx.case_type in ("user_not_working", "user_left_position", "forklift_in_zone", "unknown"):
        ct = ctx.case_type
    else:
        ct = None

    if ct is None:
        if ctx.actor.type == "operator" and (ctx.evidence.idle_seconds or 0) >= 300:
            ct = "user_not_working"
        elif ctx.actor.type == "forklift":
            ct = "forklift_in_zone"
        else:
            ct = "unknown"

    template = TEMPLATES[ct]
    sev: Severity = ctx.severity or template.severity
    return ClassifiedCase(case_type=ct, severity=sev, template_id=template.template_id)


def classify_case(ctx: IndustrialContext) -> ClassifiedCase:
    try:
        model = OllamaModel(model_id=settings.ollama_model)
        agent = Agent(model=model, system_prompt=CLASSIFIER_SYSTEM_PROMPT)
        raw = agent(json.dumps(ctx.model_dump(mode="json"), ensure_ascii=False))
        data = json.loads(str(raw))
        case_type: CaseType = data.get("case_type", "unknown")
        severity: Severity = data.get("severity", TEMPLATES.get(case_type, TEMPLATES["unknown"]).severity)

        if case_type not in TEMPLATES:
            case_type = "unknown"
        if severity not in ("info", "warning", "critical"):
            severity = TEMPLATES[case_type].severity

        return ClassifiedCase(case_type=case_type, severity=severity, template_id=TEMPLATES[case_type].template_id)
    except Exception:
        return _fallback_classify(ctx)


def _extract_message_id(response) -> str | None:
    if hasattr(response, "get"):
        mid = response.get("id")
        if mid:
            return mid
    if hasattr(response, "data") and isinstance(response.data, dict):
        return response.data.get("id")
    return None


def _classified_for_type(case_type: CaseType, ctx: IndustrialContext) -> ClassifiedCase:
    tpl = TEMPLATES[case_type]
    sev: Severity = ctx.severity or tpl.severity
    return ClassifiedCase(case_type=case_type, severity=sev, template_id=tpl.template_id)


def _dispatch_email(
    ctx: IndustrialContext,
    classified: ClassifiedCase,
    *,
    db_template=None,
) -> tuple[list[str], bool, EmailTemplate]:
    if db_template is not None:
        from vision_ops_alerting.services.email_templates import render_email_template

        tpl = render_email_template(db_template, ctx)
    else:
        tpl = render_template(TEMPLATES[classified.case_type], ctx)

    if settings.dry_run:
        return [], True, tpl

    if not settings.mailersend_api_token:
        raise RuntimeError("Missing ALERTING_MAILERSEND_API_TOKEN")
    if not settings.from_email:
        raise RuntimeError("Missing ALERTING_FROM_EMAIL")
    if not settings.to_emails:
        raise RuntimeError("Missing ALERTING_TO_EMAIL")

    ms = MailerSendClient(api_key=settings.mailersend_api_token)
    message_ids: list[str] = []

    for recipient in settings.to_recipients():
        email = (
            EmailBuilder()
            .from_email(settings.from_email, settings.from_name)
            .to_many([recipient])
            .subject(tpl.subject)
            .html(tpl.html)
            .text(tpl.text)
            .build()
        )
        response = ms.emails.send(email)
        mid = _extract_message_id(response)
        if mid:
            message_ids.append(mid)

    return message_ids, False, tpl


def send_email_for_context(ctx: IndustrialContext, *, db_template=None) -> tuple[ClassifiedCase, list[str], bool]:
    classified = classify_case(ctx)
    message_ids, dry_run, tpl = dispatch_classified_email(ctx, classified, db_template=db_template)
    return ClassifiedCase(case_type=classified.case_type, severity=classified.severity, template_id=tpl.template_id), message_ids, dry_run


def dispatch_classified_email(
    ctx: IndustrialContext,
    classified: ClassifiedCase,
    *,
    db_template=None,
) -> tuple[list[str], bool, EmailTemplate]:
    return _dispatch_email(ctx, classified, db_template=db_template)


def send_email_for_case_type(
    ctx: IndustrialContext,
    case_type: CaseType,
    *,
    db_template=None,
) -> tuple[ClassifiedCase, list[str], bool]:
    classified = _classified_for_type(case_type, ctx)
    message_ids, dry_run, tpl = _dispatch_email(ctx, classified, db_template=db_template)
    return ClassifiedCase(case_type=classified.case_type, severity=classified.severity, template_id=tpl.template_id), message_ids, dry_run

