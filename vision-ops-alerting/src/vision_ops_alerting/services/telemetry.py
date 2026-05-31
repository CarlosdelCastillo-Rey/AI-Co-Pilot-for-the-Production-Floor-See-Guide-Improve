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
    ("alerting_email", "Alerting + Email (MailerSend)"),
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
    url = f"{settings.vision_backend_url.rstrip('/')}/health"
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
    url = f"{settings.vision_backend_url.rstrip('/')}/api/vision/status"
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
    sq = probe_sqlite(db)
    em = probe_alerting_email(db)

    for r in (vb, vm, sq, em):
        store_sample(db, r)
    db.commit()

    metrics = []
    for result in (vb, vm, em):
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
