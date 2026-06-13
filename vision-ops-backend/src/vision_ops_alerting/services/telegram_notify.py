from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import httpx

from vision_ops_alerting.agent import ClassifiedCase
from vision_ops_alerting.config import settings
from vision_ops_alerting.db.models import EmailTemplateRecord
from vision_ops_alerting.email_layout import build_template_vars, substitute
from vision_ops_alerting.schemas import IndustrialContext
from vision_ops_alerting.templates import TEMPLATES

CASE_TITLES = {
    "user_not_working": "Operator Idle",
    "user_left_position": "Operator Left Position",
    "forklift_in_zone": "Forklift Zone Intrusion",
    "unknown": "Vision Alert",
}

SEVERITY_EMOJI = {
    "critical": "🔴",
    "warning": "🟡",
    "info": "ℹ️",
}


@dataclass(frozen=True)
class TelegramSendResult:
    chat_id: str
    message_id: str | None = None
    error: str | None = None


@dataclass(frozen=True)
class TelegramMessage:
    template_id: str
    text: str
    photo_url: str | None = None
    photo_path: str | None = None


def _strip_markdown(text: str) -> str:
    return text.replace("**", "")


def _resolve_photo(ctx: IndustrialContext, snapshot_url: str = "") -> tuple[str | None, str | None]:
    candidates = [ctx.evidence.snapshot_path, snapshot_url]
    for candidate in candidates:
        if not candidate:
            continue
        if candidate.startswith("data:image"):
            continue
        if candidate.startswith(("http://", "https://")):
            if "localhost" in candidate or "127.0.0.1" in candidate:
                data = _download_image_bytes(candidate)
                if data:
                    return None, _write_temp_jpeg(data)
            return candidate, None
        if candidate.startswith("/api/"):
            data = _download_image_bytes(f"{settings.public_api_base.rstrip('/')}{candidate}")
            if data:
                return None, _write_temp_jpeg(data)
            continue
        path = Path(candidate)
        if path.is_file():
            return None, str(path)
    return None, None


def _download_image_bytes(url: str) -> bytes | None:
    try:
        response = httpx.get(url, timeout=10, follow_redirects=True)
        if response.status_code == 200 and "image" in response.headers.get("content-type", ""):
            return response.content
    except Exception:
        return None
    return None


def _write_temp_jpeg(data: bytes) -> str:
    from vision_ops_alerting.services.alert_snapshot import cache_alert_jpeg

    asset = cache_alert_jpeg(data, prefix="tg")
    return str(asset.local_path)


def render_telegram_message(
    ctx: IndustrialContext,
    classified: ClassifiedCase,
    *,
    db_template: EmailTemplateRecord | None = None,
) -> TelegramMessage:
    if db_template:
        values = build_template_vars(
            ctx,
            footer_reason=db_template.footer_reason,
            snapshot_url=db_template.snapshot_url or "",
        )
        headline = substitute(db_template.headline, values)
        body = _strip_markdown(substitute(db_template.body, values))
        category = db_template.category
        template_id = db_template.id
        severity = db_template.severity_level
        photo_url, photo_path = _resolve_photo(ctx, db_template.snapshot_url or "")
    else:
        tpl = TEMPLATES[classified.case_type]
        values = build_template_vars(ctx, footer_reason="VisionOps alert", snapshot_url="")
        headline = CASE_TITLES.get(classified.case_type, "Vision Alert")
        body = f"Alert on {ctx.line_id} / {ctx.camera_id}"
        category = "Vision event"
        template_id = tpl.template_id
        severity = classified.severity
        photo_url, photo_path = _resolve_photo(ctx)

    emoji = SEVERITY_EMOJI.get(severity, "ℹ️")
    lines = [
        f"{emoji} VisionOps — {headline}",
        category,
        f"Line: {values['line_id']} · Camera: {values['camera_id']}",
        f"Actor: {values.get('actor', 'unknown')}",
        body,
        f"Severity: {severity.upper()}",
        f"Time: {values.get('timestamp_display', values['timestamp'])}",
    ]
    if values.get("live_url"):
        lines.append(f"Live: {values['live_url']}")
    if values.get("timeline_url"):
        lines.append(f"Timeline: {values['timeline_url']}")

    return TelegramMessage(
        template_id=template_id,
        text="\n".join(lines),
        photo_url=photo_url,
        photo_path=photo_path,
    )


def _api_base() -> str:
    return f"https://api.telegram.org/bot{settings.telegram_bot_token}"


def _extract_message_id(data: dict) -> str | None:
    if not data.get("ok"):
        return None
    result = data.get("result")
    if isinstance(result, dict) and result.get("message_id") is not None:
        return str(result["message_id"])
    return None


def _send_text(client: httpx.Client, chat_id: str, text: str) -> TelegramSendResult:
    try:
        response = client.post(
            f"{_api_base()}/sendMessage",
            json={"chat_id": chat_id, "text": text},
            timeout=30,
        )
        data = response.json()
        if data.get("ok"):
            return TelegramSendResult(chat_id=chat_id, message_id=_extract_message_id(data))
        return TelegramSendResult(
            chat_id=chat_id,
            error=data.get("description") or response.text[:200],
        )
    except Exception as e:
        return TelegramSendResult(chat_id=chat_id, error=str(e)[:200])


def _send_photo_url(client: httpx.Client, chat_id: str, url: str, caption: str) -> TelegramSendResult:
    try:
        response = client.post(
            f"{_api_base()}/sendPhoto",
            json={"chat_id": chat_id, "photo": url, "caption": caption[:1024]},
            timeout=60,
        )
        data = response.json()
        if data.get("ok"):
            return TelegramSendResult(chat_id=chat_id, message_id=_extract_message_id(data))
        return TelegramSendResult(
            chat_id=chat_id,
            error=data.get("description") or response.text[:200],
        )
    except Exception as e:
        return TelegramSendResult(chat_id=chat_id, error=str(e)[:200])


def _send_photo_file(client: httpx.Client, chat_id: str, path: str, caption: str) -> TelegramSendResult:
    try:
        with Path(path).open("rb") as photo:
            response = client.post(
                f"{_api_base()}/sendPhoto",
                data={"chat_id": chat_id, "caption": caption[:1024]},
                files={"photo": photo},
                timeout=60,
            )
        data = response.json()
        if data.get("ok"):
            return TelegramSendResult(chat_id=chat_id, message_id=_extract_message_id(data))
        return TelegramSendResult(
            chat_id=chat_id,
            error=data.get("description") or response.text[:200],
        )
    except Exception as e:
        return TelegramSendResult(chat_id=chat_id, error=str(e)[:200])


def dispatch_telegram(
    ctx: IndustrialContext,
    classified: ClassifiedCase,
    *,
    db_template: EmailTemplateRecord | None = None,
) -> tuple[list[str], bool, TelegramMessage, str | None]:
    message = render_telegram_message(ctx, classified, db_template=db_template)

    if settings.dry_run:
        return [], True, message, None

    if not settings.telegram_bot_token:
        raise RuntimeError("Missing VISIONOPS_TELEGRAM_BOT_TOKEN")
    if not settings.telegram_chat_id_list:
        raise RuntimeError("Missing VISIONOPS_TELEGRAM_CHAT_IDS")

    message_ids: list[str] = []
    errors: list[str] = []

    with httpx.Client(timeout=60) as client:
        for chat_id in settings.telegram_chat_id_list:
            if message.photo_url or message.photo_path:
                if message.photo_url:
                    photo_result = _send_photo_url(client, chat_id, message.photo_url, message.text)
                else:
                    photo_result = _send_photo_file(client, chat_id, message.photo_path or "", message.text)
                if photo_result.message_id:
                    message_ids.append(photo_result.message_id)
                elif photo_result.error:
                    errors.append(f"{chat_id} photo: {photo_result.error}")
                    text_result = _send_text(client, chat_id, message.text)
                    if text_result.message_id:
                        message_ids.append(text_result.message_id)
                    elif text_result.error:
                        errors.append(f"{chat_id}: {text_result.error}")
            else:
                text_result = _send_text(client, chat_id, message.text)
                if text_result.message_id:
                    message_ids.append(text_result.message_id)
                elif text_result.error:
                    errors.append(f"{chat_id}: {text_result.error}")

    if not message_ids and errors:
        raise RuntimeError(f"Telegram delivery failed: {'; '.join(errors)}")

    error_message = "; ".join(errors) if errors else None
    return message_ids, False, message, error_message


def telegram_notification_status() -> dict:
    configured = bool(settings.telegram_bot_token and settings.telegram_chat_id_list)
    masked = [f"…{cid[-4:]}" if len(cid) > 4 else cid for cid in settings.telegram_chat_id_list]
    return {
        "provider": "telegram_bot_api",
        "channel": "telegram",
        "configured": configured,
        "dryRun": settings.dry_run,
        "chatIds": masked,
        "chatCount": len(settings.telegram_chat_id_list),
        "status": (
            "ready"
            if configured and not settings.dry_run
            else ("dry_run" if settings.dry_run else "not_configured")
        ),
    }
