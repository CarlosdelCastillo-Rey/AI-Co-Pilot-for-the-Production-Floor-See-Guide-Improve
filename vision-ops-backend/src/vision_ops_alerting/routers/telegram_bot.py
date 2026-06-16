"""Telegram bot: /list employees, /report <name>, AI Q&A when mentioned."""
from __future__ import annotations

import logging
import re
import tempfile
import threading
import time
from difflib import get_close_matches
from typing import Any

import httpx
from fastapi import APIRouter

from vision_ops_alerting.config import settings

logger = logging.getLogger(__name__)
router = APIRouter(tags=["telegram-bot"])

_POLL_INTERVAL = 2          # seconds between getUpdates calls
_POLL_TIMEOUT  = 30         # long-poll timeout
_offset: int   = 0
_last_list: list[dict] = []  # cached from last /list call, used for /report <n>
_bot_thread: threading.Thread | None = None
_bot_username: str = ""     # fetched from getMe on startup, used to detect group mentions


# ── Telegram helpers ──────────────────────────────────────────────────────────

def _api(method: str) -> str:
    return f"https://api.telegram.org/bot{settings.telegram_bot_token}/{method}"


def _send_text(client: httpx.Client, chat_id: str | int, text: str) -> None:
    try:
        client.post(_api("sendMessage"), json={"chat_id": chat_id, "text": text}, timeout=30)
    except Exception as exc:
        logger.warning("bot sendMessage failed: %s", exc)


def _send_document(client: httpx.Client, chat_id: str | int, pdf_bytes: bytes, filename: str, caption: str) -> None:
    try:
        client.post(
            _api("sendDocument"),
            data={"chat_id": str(chat_id), "caption": caption[:1024]},
            files={"document": (filename, pdf_bytes, "application/pdf")},
            timeout=60,
        )
    except Exception as exc:
        logger.warning("bot sendDocument failed: %s", exc)


def _send_photo(client: httpx.Client, chat_id: str | int, path: str, caption: str) -> None:
    try:
        with open(path, "rb") as f:
            client.post(
                _api("sendPhoto"),
                data={"chat_id": str(chat_id), "caption": caption[:1024]},
                files={"photo": f},
                timeout=30,
            )
    except Exception as exc:
        logger.warning("bot sendPhoto failed: %s", exc)


def _send_typing(client: httpx.Client, chat_id: str | int) -> None:
    """Show 'typing...' indicator in the chat while the AI is thinking."""
    try:
        client.post(_api("sendChatAction"), json={"chat_id": chat_id, "action": "typing"}, timeout=5)
    except Exception:
        pass


def _fetch_bot_username(client: httpx.Client) -> str:
    """Return the bot's own @username (without @) so we can detect group mentions."""
    try:
        r = client.get(_api("getMe"), timeout=10)
        return r.json().get("result", {}).get("username", "")
    except Exception:
        return ""


# ── Data helpers ──────────────────────────────────────────────────────────────

def _backend_url(path: str) -> str:
    return f"http://127.0.0.1:8000{path}"


def _fetch_persons(client: httpx.Client) -> list[dict]:
    try:
        r = client.get(_backend_url("/api/har/v2/persons?limit=100"), timeout=10)
        r.raise_for_status()
        return r.json().get("persons", [])
    except Exception as exc:
        logger.warning("bot: fetch persons failed: %s", exc)
        return []


def _fetch_report(client: httpx.Client, person_id: str) -> dict | None:
    try:
        r = client.get(_backend_url(f"/api/har/v2/persons/{person_id}/report"), timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception as exc:
        logger.warning("bot: fetch report failed: %s", exc)
        return None


def _fetch_thumbnail(client: httpx.Client, url: str) -> bytes | None:
    if not url:
        return None
    try:
        full = _backend_url(url) if url.startswith("/") else url
        r = client.get(full, timeout=10)
        if r.status_code == 200:
            return r.content
    except Exception:
        pass
    return None


# ── Command handlers ──────────────────────────────────────────────────────────

def _handle_list(client: httpx.Client, chat_id: str | int) -> None:
    global _last_list
    persons = _fetch_persons(client)
    if not persons:
        _send_text(client, chat_id, "No employees registered yet. Run a HAR session first.")
        return

    _last_list = persons
    lines = ["Registered Employees\n"]
    for i, p in enumerate(persons, 1):
        name = p.get("display_name") or "Unknown"
        dominant = p.get("dominant_action") or "--"
        n = int(p.get("n_appearances") or 0)
        lines.append(f"{i}. {name}  |  {dominant}  |  {n} events")
    first_tokens = (persons[0].get("display_name") or "").split()
    example_name = first_tokens[0] if first_tokens else "name"
    lines.append(
        f"\nSend /report <name or number> to get a PDF report.\n"
        f"Example: /report 1  or  /report {example_name}"
    )
    _send_text(client, chat_id, "\n".join(lines))


def _resolve_person(query: str) -> dict | None:
    """Match query (number or name substring) against _last_list."""
    if not _last_list:
        return None
    q = query.strip()
    if q.isdigit():
        idx = int(q) - 1
        if 0 <= idx < len(_last_list):
            return _last_list[idx]
        return None
    names = [p.get("display_name") or "" for p in _last_list]
    matches = get_close_matches(q, names, n=1, cutoff=0.4)
    if matches:
        for p in _last_list:
            if p.get("display_name") == matches[0]:
                return p
    ql = q.lower()
    for p in _last_list:
        if ql in (p.get("display_name") or "").lower():
            return p
    return None


def _handle_report(client: httpx.Client, chat_id: str | int, query: str) -> None:
    global _last_list
    if not query.strip():
        _send_text(client, chat_id,
                   "Usage: /report <name or number>\nSend /list first to see available employees.")
        return

    if not _last_list:
        _last_list = _fetch_persons(client)

    person = _resolve_person(query)
    if person is None:
        _send_text(client, chat_id,
                   f"Could not find \"{query}\" in the registry.\nSend /list to see all employees.")
        return

    pid = person["global_person_id"]
    name = person.get("display_name") or "Unknown"
    _send_text(client, chat_id, f"Generating report for {name}...")

    report = _fetch_report(client, pid)
    if report is None:
        _send_text(client, chat_id, f"Failed to fetch report data for {name}.")
        return

    thumb_bytes = _fetch_thumbnail(client, person.get("thumbnail_url") or "")

    # Pick the single highest-confidence crop for each unique action label.
    action_crops: dict[str, bytes] = {}
    snapshots = report.get("snapshots") or []
    snapshots_by_conf = sorted(snapshots, key=lambda s: -(s.get("confidence") or 0))
    for snap in snapshots_by_conf:
        action = (snap.get("action_label") or "").strip()
        crop_url = snap.get("crop_url") or ""
        if not action or action in action_crops or not crop_url:
            continue
        crop_bytes = _fetch_thumbnail(client, crop_url)
        if crop_bytes:
            action_crops[action] = crop_bytes
        if len(action_crops) >= 12:
            break

    try:
        from vision_ops_alerting.services.person_report_pdf import generate_person_pdf
        pdf_bytes = generate_person_pdf(
            report,
            thumbnail_bytes=thumb_bytes,
            action_crops=action_crops or None,
        )
    except Exception as exc:
        logger.warning("bot: PDF generation failed: %s", exc)
        _send_text(client, chat_id, f"PDF generation failed: {exc}")
        return

    dominant = (report.get("metrics") or {}).get("dominant_action") or "--"
    n_events = (report.get("metrics") or {}).get("n_events") or 0
    caption = (
        f"Person Report -- {name}\n"
        f"Dominant: {dominant}  x  {n_events} events\n"
        f"VisionOps Industrial AI"
    )
    safe_name = name.lower().replace(" ", "_")[:30]
    filename = f"visionops-report-{safe_name}.pdf"
    _send_document(client, chat_id, pdf_bytes, filename, caption)

    if thumb_bytes:
        try:
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                tmp.write(thumb_bytes)
                _send_photo(client, chat_id, tmp.name, f"Last seen crop -- {name}")
        except Exception:
            pass


def _handle_ack(client: httpx.Client, chat_id: str | int, event_id: str) -> None:
    """Acknowledge a timeline event by ID."""
    if not event_id.strip():
        _send_text(client, chat_id, "Usage: /ack <event_id>")
        return
    try:
        r = client.patch(_backend_url(f"/api/timeline/events/{event_id.strip()}/ack"), timeout=10)
        r.raise_for_status()
        _send_text(client, chat_id, f"Event {event_id.strip()} acknowledged.")
    except Exception as exc:
        _send_text(client, chat_id, f"Failed to acknowledge event: {exc}")


def _handle_resolve(client: httpx.Client, chat_id: str | int, args: str) -> None:
    """Resolve a timeline event: /resolve <event_id> [notes]"""
    parts = args.strip().split(" ", 1)
    event_id = parts[0]
    notes = parts[1] if len(parts) > 1 else ""
    if not event_id:
        _send_text(client, chat_id, "Usage: /resolve <event_id> [notes]")
        return
    try:
        r = client.patch(
            _backend_url(f"/api/timeline/events/{event_id}/resolve"),
            json={"notes": notes},
            timeout=10,
        )
        r.raise_for_status()
        _send_text(client, chat_id, f"Event {event_id} resolved.")
    except Exception as exc:
        _send_text(client, chat_id, f"Failed to resolve event: {exc}")


def _handle_help(client: httpx.Client, chat_id: str | int) -> None:
    _send_text(
        client,
        chat_id,
        "VisionOps AI Bot\n\n"
        "Commands:\n"
        "/list -- Show all registered employees\n"
        "/report <name or #> -- PDF report for an employee\n"
        "/ack <event_id> -- Acknowledge a timeline incident\n"
        "/resolve <event_id> [notes] -- Resolve a timeline incident\n\n"
        "AI assistant:\n"
        "Just ask me anything about the production floor -- cameras, actions, alerts, persons.\n"
        "I can also take actions: acknowledge alerts, rename persons, run HAR probes, and more.\n\n"
        "Examples:\n"
        "  What is happening on the floor right now?\n"
        "  Acknowledge event abc-123\n"
        "  Rename person xyz-456 to Carlos\n"
        "  /list\n"
        "  /report Emilio",
    )


# ── AI assistant handler ──────────────────────────────────────────────────────

def _handle_ai_response(client: httpx.Client, chat_id: str | int, question: str) -> None:
    """Route any free-form message to the VisionOps advisor and reply."""
    if not question.strip():
        _handle_help(client, chat_id)
        return

    _send_typing(client, chat_id)

    try:
        r = client.post(
            _backend_url("/api/advisor/chat"),
            json={"message": question.strip(), "page": "live"},
            timeout=60,
        )
        r.raise_for_status()
        reply = (r.json().get("reply") or "").strip()
        if not reply:
            reply = "The advisor returned an empty response. Try /list or ask about cameras and alerts."
    except Exception as exc:
        logger.warning("bot AI response failed: %s", exc)
        reply = "Could not reach the VisionOps advisor. Is the server running?"

    _send_text(client, chat_id, reply)


# ── Update dispatcher ─────────────────────────────────────────────────────────

def _strip_mentions(text: str) -> str:
    """Remove all @username tokens from text so the AI gets a clean question."""
    return re.sub(r"@\S+\s*", "", text).strip()


def _bot_is_mentioned(msg: dict, bot_username: str) -> bool:
    """
    Return True when the bot's @username appears in the message entities.
    Telegram marks each @mention as an entity of type 'mention'; we compare
    the extracted text to the bot's username (case-insensitive).
    """
    entities = msg.get("entities") or []
    full_text = msg.get("text") or ""
    for ent in entities:
        if ent.get("type") != "mention":
            continue
        offset = ent.get("offset", 0)
        length = ent.get("length", 0)
        mention = full_text[offset : offset + length].lstrip("@")
        if mention.lower() == bot_username.lower():
            return True
    return False


def _dispatch(client: httpx.Client, update: dict[str, Any]) -> None:
    global _bot_username
    msg = update.get("message") or update.get("edited_message")
    if not isinstance(msg, dict):
        return
    chat_id = (msg.get("chat") or {}).get("id")
    text = str(msg.get("text") or "").strip()
    if not chat_id or not text:
        return

    chat_type = (msg.get("chat") or {}).get("type", "private")
    is_private = chat_type == "private"

    # Fetch bot username once if not yet known.
    if not _bot_username:
        _bot_username = _fetch_bot_username(client)

    # Strip bot @mention suffix from command (e.g. /list@VisionOpsBot -> /list)
    parts = text.split(" ", 1)
    cmd = parts[0].split("@")[0].lower()
    rest = parts[1] if len(parts) > 1 else ""

    if cmd in ("/list", "list"):
        _handle_list(client, chat_id)
    elif cmd in ("/report", "report"):
        _handle_report(client, chat_id, rest.strip())
    elif cmd in ("/ack",):
        _handle_ack(client, chat_id, rest.strip())
    elif cmd in ("/resolve",):
        _handle_resolve(client, chat_id, rest.strip())
    elif cmd in ("/start", "/help", "help"):
        _handle_help(client, chat_id)
    elif is_private or _bot_is_mentioned(msg, _bot_username):
        # Private chat: respond to everything.
        # Group chat: respond only when the bot is explicitly @mentioned.
        # Strip all @usernames so the AI gets the clean question.
        question = _strip_mentions(text) or text
        _handle_ai_response(client, chat_id, question)
    # else: group message without a mention — ignore silently.


# ── Polling loop ──────────────────────────────────────────────────────────────

def _poll_loop() -> None:
    global _offset, _bot_username
    logger.info("Telegram bot polling started")
    with httpx.Client(timeout=_POLL_TIMEOUT + 5) as client:
        if not _bot_username:
            _bot_username = _fetch_bot_username(client)
            if _bot_username:
                logger.info("Telegram bot username: @%s", _bot_username)
        while True:
            try:
                r = client.get(
                    _api("getUpdates"),
                    params={"offset": _offset, "timeout": _POLL_TIMEOUT, "allowed_updates": ["message"]},
                    timeout=_POLL_TIMEOUT + 10,
                )
                data = r.json()
                if not data.get("ok"):
                    time.sleep(_POLL_INTERVAL)
                    continue
                for update in data.get("result", []):
                    uid = update.get("update_id", 0)
                    if uid >= _offset:
                        _offset = uid + 1
                    try:
                        _dispatch(client, update)
                    except Exception as exc:
                        logger.warning("bot dispatch error: %s", exc)
            except httpx.TimeoutException:
                pass
            except Exception as exc:
                logger.warning("bot poll error: %s", exc)
                time.sleep(_POLL_INTERVAL)


def start_bot_polling() -> None:
    global _bot_thread
    if not settings.telegram_bot_token:
        logger.info("Telegram bot: no token configured, polling disabled")
        return
    if _bot_thread and _bot_thread.is_alive():
        return
    _bot_thread = threading.Thread(target=_poll_loop, name="tg-bot-poll", daemon=True)
    _bot_thread.start()
    logger.info("Telegram bot polling thread started")


# ── Webhook endpoint (for production deployments) ─────────────────────────────

@router.post("/api/telegram/webhook")
async def telegram_webhook(payload: dict) -> dict:
    """Receive a Telegram webhook update (use when running behind a public URL)."""
    with httpx.Client(timeout=60) as client:
        try:
            _dispatch(client, payload)
        except Exception as exc:
            logger.warning("webhook dispatch error: %s", exc)
    return {"ok": True}


@router.get("/api/telegram/bot/status")
def bot_status() -> dict:
    return {
        "polling": bool(_bot_thread and _bot_thread.is_alive()),
        "token_configured": bool(settings.telegram_bot_token),
        "last_list_count": len(_last_list),
        "offset": _offset,
    }
