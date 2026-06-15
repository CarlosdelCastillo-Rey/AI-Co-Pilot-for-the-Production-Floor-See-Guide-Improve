"""Unified VisionOps FastAPI application (vision + ops)."""

from __future__ import annotations

import logging
import signal
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session

from vision_ops_alerting.agent import classify_case
from vision_ops_alerting.db.session import get_db, init_db
from vision_ops_alerting.routers import (
    admin,
    advisor,
    alert_snapshots,
    alerts,
    analytics,
    auth,
    cameras as ops_cameras,
    email_templates,
    har as ops_har,
    har_v2,
    notifications,
    retrain,
    timeline,
)
from vision_ops_alerting.routers import settings as settings_router
from vision_ops_alerting.schemas import EmailSendResponse, IndustrialContext, Severity
from vision_ops_alerting.services.email_templates import resolve_template
from vision_ops_alerting.services.notification_dispatch import (
    dispatch_classified_alert,
    email_sent_from_result,
)
from vision_ops_alerting.services.rule_dispatch import RuleNotEnabledError, require_enabled_rule
from vision_ops_backend.config import settings
from vision_ops_backend.face.sface_live import SFaceLiveEngine
from vision_ops_backend.routers import cameras, faces, har, vision
from vision_ops_backend.webcam import WebcamCapture

logger = logging.getLogger(__name__)


def _severity_from_rule(rule_severity: str, fallback: Severity) -> Severity:
    mapping = {"CRITICAL": "critical", "WARNING": "warning"}
    return mapping.get(rule_severity, fallback)  # type: ignore[return-value]


@asynccontextmanager
async def lifespan(app: FastAPI):
    def _stop_har_streams() -> None:
        bench = getattr(app.state, "har_bench", None)
        live = getattr(app.state, "har_live", None)
        if bench is not None:
            bench.stop()
        if live is not None:
            live.stop_all()

    def _on_reload_signal(signum, frame) -> None:  # noqa: ARG001
        _stop_har_streams()

    signal.signal(signal.SIGTERM, _on_reload_signal)

    init_db()

    webcam: WebcamCapture | None = None
    if settings.webcam_enabled:
        face_engine = None
        if settings.face_enabled:
            face_engine = SFaceLiveEngine(
                owner_name=settings.owner_name,
                match_threshold=settings.face_match_threshold,
            )
        webcam = WebcamCapture(
            camera_index=settings.camera_index,
            face_engine=face_engine,
        )
        webcam.start()
    app.state.webcam = webcam
    from vision_ops_alerting.services.cameras import set_runtime_webcam

    set_runtime_webcam(webcam)

    har_live = None
    har_bench = None
    if settings.har_enabled and settings.har_live_enabled:
        from vision_ops_backend.vision.har.har_bench import get_har_bench_manager
        from vision_ops_backend.vision.har.live_stream import get_har_live_manager

        har_live = get_har_live_manager()
        har_live.start_all()
        app.state.har_live = har_live

        har_bench = get_har_bench_manager()
        har_bench.start()
        app.state.har_bench = har_bench

    if settings.har_enabled and settings.har_auto_probe_on_start:

        def _auto_har_probe() -> None:
            try:
                from vision_ops_backend.vision.har.probe_runner import run_all_har_probes

                run_all_har_probes(reshuffle_videos=True)
            except Exception as exc:
                logger.warning("HAR auto-probe on start failed: %s", exc)

        import threading

        threading.Thread(target=_auto_har_probe, daemon=True).start()

    yield
    signal.signal(signal.SIGTERM, signal.SIG_DFL)
    _stop_har_streams()
    if webcam is not None:
        webcam.stop()


app = FastAPI(
    title="VisionOps",
    description="Unified vision inference, HAR, alerts, timeline, and AI advisor.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Vision layer
app.include_router(cameras.router)
app.include_router(faces.router)
app.include_router(vision.router)
app.include_router(har.router)

# Ops layer (ex-alerting)
app.include_router(admin.router)
app.include_router(alert_snapshots.router)
app.include_router(auth.router)
app.include_router(alerts.router)
app.include_router(email_templates.router)
app.include_router(ops_cameras.router)
app.include_router(ops_har.router)
app.include_router(har_v2.router)
app.include_router(retrain.router)
app.include_router(timeline.router)
app.include_router(analytics.router)
app.include_router(notifications.router)
app.include_router(settings_router.router)
app.include_router(advisor.router)


@app.get("/health")
def unified_health():
    return {
        "status": "ok",
        "service": "vision-ops-backend",
        "vision": {
            "webcam": settings.webcam_enabled,
            "har": settings.har_enabled and settings.har_live_enabled,
        },
    }


@app.post("/api/alerting/email", response_model=EmailSendResponse)
def send_email(ctx: IndustrialContext, db: Session = Depends(get_db)):
    try:
        classified = classify_case(ctx)
        rule = require_enabled_rule(db, classified.case_type)
        severity = _severity_from_rule(rule.severity, classified.severity)
        db_template = resolve_template(db, classified.case_type, rule.email_template_id)

        result = dispatch_classified_alert(
            db,
            ctx,
            classified,
            rule,
            db_template,
            severity=severity,
        )
        db.commit()

        return EmailSendResponse(ok=True, event=email_sent_from_result(result))
    except RuleNotEnabledError as e:
        db.rollback()
        raise HTTPException(status_code=403, detail=str(e)) from e
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=400, detail=str(e)) from e
