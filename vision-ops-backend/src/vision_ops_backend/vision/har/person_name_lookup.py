"""In-process person registry name lookup (no HTTP)."""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

_name_cache: dict[str, str] = {}


def lookup_person_display_name(global_person_id: str) -> str | None:
    gid = (global_person_id or "").strip()
    if not gid:
        return None
    cached = _name_cache.get(gid)
    if cached:
        return cached

    try:
        from vision_ops_alerting.db.session import SessionLocal
        from vision_ops_alerting.services.har_session_store import get_person

        with SessionLocal() as db:
            data = get_person(db, gid)
    except Exception as exc:
        logger.debug("Person name lookup failed for %s: %s", gid, exc)
        return None

    if not data:
        return None
    name = (data.get("display_name") or "").strip()
    if name:
        _name_cache[gid] = name
    return name or None


def seed_person_display_name(global_person_id: str, display_name: str) -> None:
    gid = (global_person_id or "").strip()
    name = (display_name or "").strip()
    if gid and name:
        _name_cache[gid] = name


def hydrate_track_names_from_registry(
    track_global: dict[int, str],
    track_names: dict[int, str],
) -> None:
    for tid, gid in track_global.items():
        if track_names.get(tid):
            continue
        name = lookup_person_display_name(gid)
        if name:
            track_names[tid] = name
