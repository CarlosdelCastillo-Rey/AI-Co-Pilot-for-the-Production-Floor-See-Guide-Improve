from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime, timezone

import httpx
from sqlalchemy import func, text
from sqlalchemy.orm import Session

from vision_ops_alerting.config import settings
from vision_ops_alerting.db.models import HealthMetricSample, new_id


@dataclass(frozen=True)
class ProbeResult:
    service: str
    label: str
    status: str
    latency_ms: float | None
    value_pct: float
    display_value: str
    detail: str


SERVICES = (
    ("vision_backend", "Vision Backend (FastAPI)"),
    ("vision_models", "Vision Models (DINO / V-JEPA)"),
    ("ollama_llm", "Advisor LLM (Ollama)"),
    ("alerting_email", "Alerting + Email (MailerSend)"),
    ("alerting_telegram", "Alerting + Telegram (Bot API)"),
)


def _ollama_model_available(tags_payload: dict, model_id: str) -> bool:
    names: list[str] = []
    for item in tags_payload.get("models") or []:
        if isinstance(item, dict):
            names.append(str(item.get("name") or ""))
    if not names:
        return False
    base = model_id.split(":")[0]
    for name in names:
        n = name.split(":")[0]
        if n == base or name.startswith(f"{base}:"):
            return True
    return False


async def probe_ollama() -> ProbeResult:
    host = settings.ollama_host.rstrip("/")
    model_id = settings.ollama_model
    url = f"{host}/api/tags"
    t0 = time.perf_counter()
    try:
        async with httpx.AsyncClient(timeout=4.0) as client:
            res = await client.get(url)
        latency = (time.perf_counter() - t0) * 1000
        if res.status_code != 200:
            return ProbeResult(
                service="ollama_llm",
                label="Advisor LLM (Ollama)",
                status="down",
                latency_ms=round(latency, 1),
                value_pct=0.0,
                display_value="DOWN",
                detail=f"HTTP {res.status_code} @ {host}",
            )
        data = res.json()
        has_model = _ollama_model_available(data, model_id)
        if has_model:
            return ProbeResult(
                service="ollama_llm",
                label="Advisor LLM (Ollama)",
                status="ok",
                latency_ms=round(latency, 1),
                value_pct=100.0,
                display_value="ACTIVE",
                detail=f"{model_id} @ {host}",
            )
        return ProbeResult(
            service="ollama_llm",
            label="Advisor LLM (Ollama)",
            status="degraded",
            latency_ms=round(latency, 1),
            value_pct=40.0,
            display_value="NO MODEL",
            detail=f"Ollama up but '{model_id}' not pulled — run: ollama pull {model_id}",
        )
    except Exception as e:
        err = str(e).strip()[:120]
        hint = (
            f"Use Ollama.app: brew install --cask ollama && open -a Ollama && ollama pull {model_id} "
            "(not brew install ollama formula)"
        )
        if "connection" in err.lower() or "connect" in err.lower():
            detail = f"{host} not reachable — {hint}"
        else:
            detail = f"{host} — {err} — {hint}"
        return ProbeResult(
            service="ollama_llm",
            label="Advisor LLM (Ollama)",
            status="down",
            latency_ms=None,
            value_pct=0.0,
            display_value="DOWN",
            detail=detail,
        )


def _uptime_pct(db: Session, service: str) -> float:
    rows = (
        db.query(HealthMetricSample.status)
        .filter(HealthMetricSample.service == service)
        .order_by(HealthMetricSample.recorded_at.desc())
        .limit(20)
        .all()
    )
    if not rows:
        return 100.0
    ok = sum(1 for (s,) in rows if s == "ok")
    return round(ok / len(rows) * 100, 1)


async def probe_vision_backend() -> ProbeResult:
    url = f"{settings.public_api_base.rstrip('/')}/health"
    t0 = time.perf_counter()
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            res = await client.get(url)
        latency = (time.perf_counter() - t0) * 1000
        ok = res.status_code == 200
        return ProbeResult(
            service="vision_backend",
            label="Vision Backend (FastAPI)",
            status="ok" if ok else "down",
            latency_ms=round(latency, 1),
            value_pct=100.0 if ok else 0.0,
            display_value=f"{round(latency, 1)}ms",
            detail=f"HTTP {res.status_code}" if ok else f"HTTP {res.status_code}",
        )
    except Exception as e:
        return ProbeResult(
            service="vision_backend",
            label="Vision Backend (FastAPI)",
            status="down",
            latency_ms=None,
            value_pct=0.0,
            display_value="DOWN",
            detail=str(e)[:200],
        )


async def probe_vision_models() -> ProbeResult:
    url = f"{settings.public_api_base.rstrip('/')}/api/vision/status"
    t0 = time.perf_counter()
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            res = await client.get(url)
        latency = (time.perf_counter() - t0) * 1000
        if res.status_code != 200:
            return ProbeResult(
                service="vision_models",
                label="Vision Models (DINO / V-JEPA)",
                status="down",
                latency_ms=round(latency, 1),
                value_pct=0.0,
                display_value="DOWN",
                detail=f"HTTP {res.status_code}",
            )
        data = res.json()
        ready = bool(data.get("ready"))
        cameras = len(data.get("cameras") or {})
        pct = 100.0 if ready else 50.0
        return ProbeResult(
            service="vision_models",
            label="Vision Models (DINO / V-JEPA)",
            status="ok" if ready else "degraded",
            latency_ms=round(latency, 1),
            value_pct=pct,
            display_value=f"{cameras} cams" if ready else "LOADING",
            detail="ready" if ready else "models warming up",
        )
    except Exception as e:
        return ProbeResult(
            service="vision_models",
            label="Vision Models (DINO / V-JEPA)",
            status="down",
            latency_ms=None,
            value_pct=0.0,
            display_value="DOWN",
            detail=str(e)[:200],
        )


def probe_sqlite(db: Session) -> ProbeResult:
    t0 = time.perf_counter()
    try:
        db.execute(text("SELECT 1"))
        latency = (time.perf_counter() - t0) * 1000
        bar = max(10.0, min(100.0, 100.0 - latency))
        return ProbeResult(
            service="sqlite_db",
            label="SQLite Database",
            status="ok",
            latency_ms=round(latency, 2),
            value_pct=bar,
            display_value=f"{round(latency, 2)}ms",
            detail="connected",
        )
    except Exception as e:
        return ProbeResult(
            service="sqlite_db",
            label="SQLite Database",
            status="down",
            latency_ms=None,
            value_pct=0.0,
            display_value="DOWN",
            detail=str(e)[:200],
        )


def probe_alerting_email(db: Session) -> ProbeResult:
    configured = bool(settings.mailersend_api_token and settings.from_email and settings.to_emails)
    if settings.dry_run:
        status = "degraded"
        pct = 60.0
        display = "DRY-RUN"
        detail = "emails not sent (dry run)"
    elif configured:
        status = "ok"
        pct = 100.0
        display = "READY"
        detail = f"{len(settings.to_emails)} recipient(s)"
    else:
        status = "down"
        pct = 0.0
        display = "NOT CONFIGURED"
        detail = "missing token or recipients"

    uptime = _uptime_pct(db, "alerting_email")
    return ProbeResult(
        service="alerting_email",
        label="Alerting + Email (MailerSend)",
        status=status,
        latency_ms=None,
        value_pct=pct if status != "ok" else uptime,
        display_value=display if status != "ok" else f"{uptime}%",
        detail=detail,
    )


async def probe_alerting_telegram() -> ProbeResult:
    configured = bool(settings.telegram_bot_token and settings.telegram_chat_id_list)
    if settings.dry_run:
        status = "degraded"
        pct = 60.0
        display = "DRY-RUN"
        detail = "messages not sent (dry run)"
    elif not configured:
        status = "down"
        pct = 0.0
        display = "NOT CONFIGURED"
        detail = "missing bot token or chat IDs"
    else:
        t0 = time.perf_counter()
        try:
            async with httpx.AsyncClient(timeout=4.0) as client:
                res = await client.get(f"https://api.telegram.org/bot{settings.telegram_bot_token}/getMe")
            latency = (time.perf_counter() - t0) * 1000
            if res.status_code == 200 and res.json().get("ok"):
                bot_name = res.json().get("result", {}).get("username", "bot")
                status = "ok"
                pct = 100.0
                display = "READY"
                detail = f"@{bot_name} · {len(settings.telegram_chat_id_list)} chat(s)"
            else:
                status = "down"
                pct = 0.0
                display = "AUTH FAILED"
                detail = res.text[:120]
                latency = None
        except Exception as e:
            latency = None
            status = "down"
            pct = 0.0
            display = "UNREACHABLE"
            detail = str(e)[:200]

    return ProbeResult(
        service="alerting_telegram",
        label="Alerting + Telegram (Bot API)",
        status=status,
        latency_ms=latency if configured and not settings.dry_run else None,
        value_pct=pct,
        display_value=display,
        detail=detail,
    )


def store_sample(db: Session, result: ProbeResult) -> None:
    db.add(
        HealthMetricSample(
            id=new_id("hm"),
            service=result.service,
            status=result.status,
            latency_ms=result.latency_ms,
            value_pct=result.value_pct,
            detail=result.detail,
            recorded_at=datetime.now(timezone.utc),
        )
    )


def history_bars(db: Session, service: str, limit: int = 10) -> list[float]:
    rows = (
        db.query(HealthMetricSample.value_pct)
        .filter(HealthMetricSample.service == service)
        .order_by(HealthMetricSample.recorded_at.desc())
        .limit(limit)
        .all()
    )
    bars = [r[0] for r in reversed(rows)]
    while len(bars) < limit:
        bars.insert(0, bars[0] if bars else 50.0)
    return bars[-limit:]


async def collect_and_build(db: Session) -> dict:
    vb = await probe_vision_backend()
    vm = await probe_vision_models()
    ollama = await probe_ollama()
    sq = probe_sqlite(db)
    em = probe_alerting_email(db)
    tg = await probe_alerting_telegram()

    for r in (vb, vm, ollama, sq, em, tg):
        store_sample(db, r)
    db.commit()

    metrics = []
    for result in (vb, vm, ollama, em, tg):
        bars = history_bars(db, result.service)
        metrics.append(
            {
                "service": result.service,
                "label": result.label,
                "status": result.status,
                "value": result.display_value,
                "detail": result.detail,
                "latencyMs": result.latency_ms,
                "bars": [int(min(100, max(5, b))) for b in bars],
                "highlight": result.status == "ok",
            }
        )

    return {
        "collectedAt": datetime.now(timezone.utc).isoformat(),
        "metrics": metrics,
    }


def email_notification_status() -> dict:
    configured = bool(settings.mailersend_api_token and settings.from_email and settings.to_emails)
    return {
        "provider": "mailersend",
        "channel": "email",
        "configured": configured,
        "dryRun": settings.dry_run,
        "fromEmail": settings.from_email or None,
        "toEmails": settings.to_emails,
        "status": "ready" if configured and not settings.dry_run else ("dry_run" if settings.dry_run else "not_configured"),
    }
