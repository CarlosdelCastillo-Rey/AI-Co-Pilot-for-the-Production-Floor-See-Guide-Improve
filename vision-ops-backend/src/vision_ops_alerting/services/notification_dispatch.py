from __future__ import annotations

from dataclasses import dataclass, field

from sqlalchemy.orm import Session

from vision_ops_alerting.agent import (
    ClassifiedCase,
    _classified_for_type,
    dispatch_classified_email,
    send_email_for_case_type,
)
from vision_ops_alerting.config import settings
from vision_ops_alerting.db.models import AlertRule
from vision_ops_alerting.schemas import CaseType, EmailSent, IndustrialContext, Severity
from vision_ops_alerting.services.events import (
    create_event_from_context,
    log_email_delivery,
    log_telegram_delivery,
)
from vision_ops_alerting.services.alert_snapshot import SnapshotAsset, enrich_context
from vision_ops_alerting.services.telegram_notify import dispatch_telegram


def _prepare_alert_context(ctx: IndustrialContext) -> tuple[IndustrialContext, SnapshotAsset | None]:
    return enrich_context(ctx)


@dataclass
class ChannelDispatch:
    delivery_id: str = ""
    message_ids: list[str] = field(default_factory=list)
    recipients: list[str] = field(default_factory=list)
    dry_run: bool = False
    template_id: str = ""
    error_message: str | None = None


@dataclass
class AlertDispatchResult:
    event_id: str
    case_type: CaseType
    severity: str
    template_id: str
    email: ChannelDispatch
    telegram: ChannelDispatch


def _dispatch_channels(
    ctx: IndustrialContext,
    classified: ClassifiedCase,
    rule: AlertRule,
    db_template,
    *,
    snapshot_asset: SnapshotAsset | None = None,
) -> tuple[str, ChannelDispatch, ChannelDispatch]:
    template_id = classified.template_id
    email = ChannelDispatch()
    telegram = ChannelDispatch()

    if rule.notify_email:
        try:
            message_ids, dry_run, rendered = dispatch_classified_email(
                ctx, classified, db_template=db_template, snapshot_asset=snapshot_asset
            )
            template_id = rendered.template_id
            email.message_ids = message_ids
            email.dry_run = dry_run
            email.recipients = settings.to_emails
            email.template_id = rendered.template_id
        except Exception as e:
            email.error_message = str(e)[:500]

    if rule.notify_telegram:
        try:
            tg_ids, tg_dry, tg_msg, tg_err = dispatch_telegram(ctx, classified, db_template=db_template)
            telegram.message_ids = tg_ids
            telegram.dry_run = tg_dry
            telegram.recipients = settings.telegram_chat_id_list
            telegram.template_id = tg_msg.template_id
            telegram.error_message = tg_err
        except Exception as e:
            telegram.error_message = str(e)[:500]

    return template_id, email, telegram


def dispatch_for_case_type(
    ctx: IndustrialContext,
    case_type: CaseType,
    rule: AlertRule,
    db_template,
    *,
    snapshot_asset: SnapshotAsset | None = None,
) -> tuple[ClassifiedCase, str, ChannelDispatch, ChannelDispatch]:
    email = ChannelDispatch()
    telegram = ChannelDispatch()

    if rule.notify_email:
        try:
            classified, message_ids, dry_run = send_email_for_case_type(
                ctx, case_type, db_template=db_template, snapshot_asset=snapshot_asset
            )
            email.message_ids = message_ids
            email.dry_run = dry_run
            email.recipients = settings.to_emails
            email.template_id = classified.template_id
        except Exception as e:
            classified = _classified_for_type(case_type, ctx)
            email.dry_run = False
            email.error_message = str(e)[:500]
    else:
        classified = _classified_for_type(case_type, ctx)
        email.dry_run = True

    template_id = classified.template_id

    if rule.notify_telegram:
        try:
            tg_ids, tg_dry, tg_msg, tg_err = dispatch_telegram(ctx, classified, db_template=db_template)
            telegram.message_ids = tg_ids
            telegram.dry_run = tg_dry
            telegram.recipients = settings.telegram_chat_id_list
            telegram.template_id = tg_msg.template_id
            telegram.error_message = tg_err
        except Exception as e:
            telegram.error_message = str(e)[:500]
    else:
        telegram.dry_run = True

    return classified, template_id, email, telegram


def persist_alert_dispatch(
    db: Session,
    *,
    ctx: IndustrialContext,
    classified: ClassifiedCase,
    rule: AlertRule,
    severity: str,
    template_id: str,
    email: ChannelDispatch,
    telegram: ChannelDispatch,
) -> AlertDispatchResult:
    event = create_event_from_context(
        db,
        ctx,
        case_type=classified.case_type,
        severity=severity,
        rule_id=rule.id,
    )

    if rule.notify_email:
        delivery = log_email_delivery(
            db,
            event_id=event.id,
            template_id=email.template_id or template_id,
            to_emails=email.recipients,
            message_ids=email.message_ids,
            dry_run=email.dry_run,
            error_message=email.error_message,
        )
        email.delivery_id = delivery.id

    if rule.notify_telegram:
        delivery = log_telegram_delivery(
            db,
            event_id=event.id,
            template_id=telegram.template_id or template_id,
            chat_ids=telegram.recipients,
            message_ids=telegram.message_ids,
            dry_run=telegram.dry_run,
            error_message=telegram.error_message,
        )
        telegram.delivery_id = delivery.id

    return AlertDispatchResult(
        event_id=event.id,
        case_type=classified.case_type,
        severity=severity,
        template_id=template_id,
        email=email,
        telegram=telegram,
    )


def dispatch_classified_alert(
    db: Session,
    ctx: IndustrialContext,
    classified: ClassifiedCase,
    rule: AlertRule,
    db_template,
    *,
    severity: str,
) -> AlertDispatchResult:
    ctx, snapshot_asset = _prepare_alert_context(ctx)
    template_id, email, telegram = _dispatch_channels(
        ctx, classified, rule, db_template, snapshot_asset=snapshot_asset
    )
    return persist_alert_dispatch(
        db,
        ctx=ctx,
        classified=classified,
        rule=rule,
        severity=severity,
        template_id=template_id,
        email=email,
        telegram=telegram,
    )


def dispatch_test_alert(
    db: Session,
    ctx: IndustrialContext,
    case_type: CaseType,
    rule: AlertRule,
    db_template,
) -> AlertDispatchResult:
    ctx, snapshot_asset = _prepare_alert_context(ctx)
    classified, template_id, email, telegram = dispatch_for_case_type(
        ctx, case_type, rule, db_template, snapshot_asset=snapshot_asset
    )
    severity = classified.severity
    return persist_alert_dispatch(
        db,
        ctx=ctx,
        classified=classified,
        rule=rule,
        severity=severity,
        template_id=template_id,
        email=email,
        telegram=telegram,
    )


def email_sent_from_result(result: AlertDispatchResult) -> EmailSent:
    return EmailSent(
        event_id=result.event_id,
        delivery_id=result.email.delivery_id,
        template_id=result.template_id,
        case_type=result.case_type,
        severity=result.severity,  # type: ignore[arg-type]
        to_emails=result.email.recipients if result.email.delivery_id or result.email.dry_run else [],
        message_ids=result.email.message_ids,
        dry_run=result.email.dry_run,
        telegram_delivery_id=result.telegram.delivery_id,
        telegram_message_ids=result.telegram.message_ids,
        telegram_chat_ids=result.telegram.recipients if result.telegram.delivery_id or result.telegram.dry_run else [],
        telegram_dry_run=result.telegram.dry_run,
        telegram_error=result.telegram.error_message,
    )
