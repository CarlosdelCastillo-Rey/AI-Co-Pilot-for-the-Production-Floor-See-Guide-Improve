"""Persist per-track action history — delegates to versioned HarSessionLogger."""

from __future__ import annotations

from lib.session_log import HarSessionLogger, list_sessions, new_session_id

__all__ = ["HarSessionLogger", "append_track_event", "load_session_log", "list_sessions", "new_session_id"]


def append_track_event(session_id: str, **kwargs):
    """Legacy shim — prefer HarSessionLogger.log_inference directly."""
    from lib.paths import MEMORY_DIR
    import json
    import time
    from pathlib import Path

    MEMORY_DIR.mkdir(parents=True, exist_ok=True)
    path = MEMORY_DIR / f"{session_id}.jsonl"
    row = {"ts": time.time(), "session_id": session_id, **kwargs}
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return path


def load_session_log(session_id: str) -> list[dict]:
    from lib.paths import SESSIONS_DIR
    import json

    # New layout: search dated folders
    for events in SESSIONS_DIR.glob(f"*/*{session_id}*/events.jsonl"):
        rows = []
        for line in events.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
        return rows

    from lib.paths import MEMORY_DIR

    path = MEMORY_DIR / f"{session_id}.jsonl"
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
