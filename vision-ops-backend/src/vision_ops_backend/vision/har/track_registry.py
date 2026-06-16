"""Map ephemeral ByteTrack ids → registry person ids and human display names."""

from __future__ import annotations

from typing import Any


def absorb_v2_event(
    track_global: dict[int, str],
    track_names: dict[int, str],
    track_id: int,
    response: dict[str, Any] | None,
) -> None:
    """Cache global_person_id + display_name from a HAR v2 session event response."""
    if not response:
        return
    # record_session_event returns a flat dict (no "event" wrapper)
    gid = response.get("global_person_id")
    if gid:
        track_global[track_id] = str(gid)
    name = (response.get("display_name") or "").strip()
    if name:
        track_names[track_id] = name


def enrich_track_predictions(
    tracks: list[dict[str, Any]],
    track_global: dict[int, str],
    track_names: dict[int, str],
) -> list[dict[str, Any]]:
    """Attach registry fields for overlay, ingest, and live UI."""
    for tr in tracks:
        tid = int(tr["track_id"])
        gid = track_global.get(tid)
        if gid:
            tr["global_person_id"] = gid
        name = track_names.get(tid)
        if name:
            tr["display_name"] = name
    return tracks
