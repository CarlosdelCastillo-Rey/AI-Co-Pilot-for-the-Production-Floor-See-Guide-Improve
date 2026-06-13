#!/usr/bin/env python3
"""
VisionOps DB context MCP server (stdio).

Exposes the same SQLite operational queries as the advisor Strands tools.
Run from vision-ops-backend/:

  uv sync --extra mcp
  uv run python mcp/db_context_server.py

Cursor MCP config example:

  {
    "mcpServers": {
      "visionops-db": {
        "command": "uv",
        "args": ["run", "python", "mcp/db_context_server.py"],
        "cwd": "/path/to/vision-ops-backend"
      }
    }
  }
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Ensure package import when launched as script
_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

from vision_ops_alerting.db.models import Camera, Event  # noqa: E402
from vision_ops_alerting.db.session import SessionLocal, init_db  # noqa: E402
from vision_ops_alerting.services.event_workflow import build_timeline_stats  # noqa: E402
from vision_ops_alerting.services.events import event_to_timeline_dict  # noqa: E402
from vision_ops_alerting.services.operational_snapshot import build_operational_snapshot  # noqa: E402
from vision_ops_alerting.services.har_activity_store import (  # noqa: E402
    activity_summary,
    analytics_plant_actions,
    build_all_cameras_har_dashboard,
    global_har_summary,
    list_activity_logs,
)
from vision_ops_alerting.services.timeline_summary import build_shift_summary  # noqa: E402


def main() -> None:
    try:
        from mcp.server.fastmcp import FastMCP
    except ImportError:
        print(
            "Install MCP extras: uv sync --extra mcp",
            file=sys.stderr,
        )
        sys.exit(1)

    init_db()
    mcp = FastMCP("visionops-db")

    @mcp.tool()
    def get_operational_snapshot(page: str = "live") -> str:
        """Full plant snapshot: cameras, alerts, uptime, vision backend."""
        with SessionLocal() as db:
            return json.dumps(build_operational_snapshot(db, page=page), default=str)

    @mcp.tool()
    def list_open_alerts(limit: int = 10) -> str:
        """Open (unresolved) timeline incidents."""
        lim = max(1, min(limit, 20))
        with SessionLocal() as db:
            rows = (
                db.query(Event)
                .filter((Event.resolution_status == "OPEN") | (Event.resolution_status.is_(None)))
                .order_by(Event.occurred_at.desc())
                .limit(lim)
                .all()
            )
            return json.dumps([event_to_timeline_dict(e) for e in rows], default=str)

    @mcp.tool()
    def list_cameras() -> str:
        """Enabled cameras with status and inference model."""
        with SessionLocal() as db:
            cameras = (
                db.query(Camera)
                .filter(Camera.enabled.is_(True))
                .order_by(Camera.sort_order, Camera.name)
                .all()
            )
            items = [
                {
                    "id": c.id,
                    "name": c.name,
                    "zone": c.zone,
                    "status": c.status,
                    "inferenceModel": c.inference_model,
                }
                for c in cameras
            ]
            return json.dumps(
                {"total": len(items), "live": sum(1 for i in items if i["status"] == "live"), "cameras": items},
                default=str,
            )

    @mcp.tool()
    def get_shift_summary(event_date: str = "") -> str:
        """Shift KPIs for YYYY-MM-DD (empty = today)."""
        target = event_date.strip() or None
        with SessionLocal() as db:
            summary = build_shift_summary(db, target)
            stats = build_timeline_stats(db, target)
            return json.dumps({**summary, **stats}, default=str)

    @mcp.tool()
    def get_har_activity_summary(camera_id: str, event_date: str = "") -> str:
        """HAR action log summary for one camera (cam-har-01, etc.)."""
        with SessionLocal() as db:
            return json.dumps(
                activity_summary(db, camera_id=camera_id, target_date=event_date.strip() or None),
                default=str,
            )

    @mcp.tool()
    def get_har_activity_logs(camera_id: str, limit: int = 30) -> str:
        """Recent HAR activity log rows for one camera."""
        lim = max(1, min(limit, 80))
        with SessionLocal() as db:
            logs, total = list_activity_logs(db, camera_id=camera_id, limit=lim)
            return json.dumps({"total": total, "logs": logs}, default=str)

    @mcp.tool()
    def get_har_plant_summary(hours: int = 24) -> str:
        """Plant-wide HAR inference counts in the last N hours."""
        with SessionLocal() as db:
            return json.dumps(global_har_summary(db, hours=max(1, min(hours, 72))), default=str)

    @mcp.tool()
    def get_all_cameras_har_dashboard(hours: int = 24) -> str:
        """Per HAR camera: latest action, confidence, today's counts, open incidents."""
        with SessionLocal() as db:
            return json.dumps(
                build_all_cameras_har_dashboard(db, hours=max(1, min(hours, 72))),
                default=str,
            )

    @mcp.tool()
    def get_har_plant_analytics(event_date: str = "") -> str:
        """Plant-wide HAR action productivity, severity tags, and action CoQ estimate."""
        with SessionLocal() as db:
            return json.dumps(
                analytics_plant_actions(db, target_date=event_date.strip() or None),
                default=str,
            )

    mcp.run()


if __name__ == "__main__":
    main()
