"""Persistent person registry — stable global_person_id via appearance embeddings."""

from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from lib.paths import OUTPUTS_DIR

PERSON_REGISTRY_DIR = OUTPUTS_DIR / "person_registry"
REGISTRY_DB = PERSON_REGISTRY_DIR / "registry.db"
DEFAULT_MATCH_THRESHOLD = 0.82


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _normalize(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32).reshape(-1)
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-8 else v


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(_normalize(a), _normalize(b)))


class PersonRegistry:
    """
    Maps ephemeral YOLO track IDs → stable `global_person_id` using HAR/V-JEPA embeddings.

    Same person re-entering frame or appearing in a later session can be linked if their
    appearance embedding is similar enough (cosine ≥ threshold).
    """

    def __init__(
        self,
        db_path: Path | None = None,
        *,
        match_threshold: float = DEFAULT_MATCH_THRESHOLD,
    ) -> None:
        self.db_path = Path(db_path or REGISTRY_DB)
        self.match_threshold = match_threshold
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS persons (
                    global_person_id TEXT PRIMARY KEY,
                    display_name     TEXT,
                    embedding_dim    INTEGER NOT NULL,
                    centroid         BLOB NOT NULL,
                    n_appearances    INTEGER DEFAULT 0,
                    first_seen_at    TEXT NOT NULL,
                    last_seen_at     TEXT NOT NULL,
                    meta_json        TEXT
                );

                CREATE TABLE IF NOT EXISTS appearances (
                    id               INTEGER PRIMARY KEY AUTOINCREMENT,
                    global_person_id TEXT NOT NULL,
                    session_id       TEXT NOT NULL,
                    track_id         INTEGER NOT NULL,
                    event_id         TEXT,
                    video            TEXT,
                    frame_idx        INTEGER,
                    action_label     TEXT,
                    confidence       REAL,
                    match_score      REAL,
                    ts               TEXT NOT NULL,
                    FOREIGN KEY (global_person_id) REFERENCES persons(global_person_id)
                );

                CREATE INDEX IF NOT EXISTS idx_app_person ON appearances(global_person_id);
                CREATE INDEX IF NOT EXISTS idx_app_session ON appearances(session_id);
                """
            )

    def _load_persons(self) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM persons").fetchall()
        out = []
        for r in rows:
            out.append(
                {
                    "global_person_id": r["global_person_id"],
                    "display_name": r["display_name"],
                    "centroid": np.frombuffer(r["centroid"], dtype=np.float32),
                    "n_appearances": r["n_appearances"],
                    "first_seen_at": r["first_seen_at"],
                    "last_seen_at": r["last_seen_at"],
                }
            )
        return out

    def resolve(
        self,
        embedding: np.ndarray,
        *,
        session_id: str,
        track_id: int,
        event_id: str | None = None,
        video: str | None = None,
        frame_idx: int | None = None,
        action_label: str | None = None,
        confidence: float | None = None,
        display_name: str | None = None,
    ) -> tuple[str, float]:
        """
        Match embedding to existing person or register new one.
        Returns (global_person_id, match_score).
        """
        emb = _normalize(embedding)
        persons = self._load_persons()
        best_id: str | None = None
        best_score = -1.0
        for p in persons:
            score = _cosine(emb, p["centroid"])
            if score > best_score:
                best_score = score
                best_id = p["global_person_id"]

        now = _utc_now()
        if best_id is not None and best_score >= self.match_threshold:
            global_id = best_id
            self._update_centroid(global_id, emb, now)
        else:
            global_id = f"person-{uuid.uuid4().hex[:10]}"
            best_score = 1.0
            with self._connect() as conn:
                conn.execute(
                    """
                    INSERT INTO persons
                    (global_person_id, display_name, embedding_dim, centroid,
                     n_appearances, first_seen_at, last_seen_at, meta_json)
                    VALUES (?, ?, ?, ?, 0, ?, ?, ?)
                    """,
                    (
                        global_id,
                        display_name,
                        emb.size,
                        emb.tobytes(),
                        now,
                        now,
                        json.dumps({"source": "auto_reid"}),
                    ),
                )

        self._log_appearance(
            global_id,
            session_id=session_id,
            track_id=track_id,
            event_id=event_id,
            video=video,
            frame_idx=frame_idx,
            action_label=action_label,
            confidence=confidence,
            match_score=best_score,
            ts=now,
        )
        return global_id, best_score

    def _update_centroid(self, global_id: str, emb: np.ndarray, now: str) -> None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT centroid, n_appearances FROM persons WHERE global_person_id = ?",
                (global_id,),
            ).fetchone()
            if row is None:
                return
            old = np.frombuffer(row["centroid"], dtype=np.float32)
            n = int(row["n_appearances"])
            # Running average in embedding space, then re-normalize
            new_centroid = _normalize((old * n + emb) / (n + 1))
            conn.execute(
                """
                UPDATE persons
                SET centroid = ?, last_seen_at = ?
                WHERE global_person_id = ?
                """,
                (new_centroid.tobytes(), now, global_id),
            )

    def _log_appearance(self, global_id: str, **kwargs: Any) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO appearances
                (global_person_id, session_id, track_id, event_id, video,
                 frame_idx, action_label, confidence, match_score, ts)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    global_id,
                    kwargs.get("session_id"),
                    kwargs.get("track_id"),
                    kwargs.get("event_id"),
                    kwargs.get("video"),
                    kwargs.get("frame_idx"),
                    kwargs.get("action_label"),
                    kwargs.get("confidence"),
                    kwargs.get("match_score"),
                    kwargs.get("ts"),
                ),
            )
            conn.execute(
                "UPDATE persons SET n_appearances = n_appearances + 1, last_seen_at = ? "
                "WHERE global_person_id = ?",
                (kwargs.get("ts"), global_id),
            )

    def list_persons(self, limit: int = 100) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM persons ORDER BY last_seen_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [dict(r) for r in rows]

    def person_history(self, global_person_id: str, limit: int = 200) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM appearances
                WHERE global_person_id = ?
                ORDER BY ts DESC LIMIT ?
                """,
                (global_person_id, limit),
            ).fetchall()
        return [dict(r) for r in rows]

    def session_track_map(self, session_id: str) -> dict[int, str]:
        """track_id → global_person_id for one session."""
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT DISTINCT track_id, global_person_id FROM appearances
                WHERE session_id = ?
                """,
                (session_id,),
            ).fetchall()
        return {int(r["track_id"]): r["global_person_id"] for r in rows}

    def get_person(self, global_person_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM persons WHERE global_person_id = ?",
                (global_person_id,),
            ).fetchone()
        return dict(row) if row else None

    def set_display_name(self, global_person_id: str, display_name: str) -> bool:
        name = (display_name or "").strip()
        if not name:
            return False
        with self._connect() as conn:
            cur = conn.execute(
                "UPDATE persons SET display_name = ? WHERE global_person_id = ?",
                (name, global_person_id),
            )
            return cur.rowcount > 0

    def person_sessions_summary(self, global_person_id: str) -> list[dict[str, Any]]:
        """One row per (session, track) with action stats."""
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT session_id, track_id, video,
                       COUNT(*) AS n_appearances,
                       MIN(frame_idx) AS frame_first,
                       MAX(frame_idx) AS frame_last,
                       MIN(ts) AS first_ts,
                       MAX(ts) AS last_ts
                FROM appearances
                WHERE global_person_id = ?
                GROUP BY session_id, track_id, video
                ORDER BY last_ts DESC
                """,
                (global_person_id,),
            ).fetchall()
            out: list[dict[str, Any]] = []
            for r in rows:
                row = dict(r)
                actions = conn.execute(
                    """
                    SELECT action_label, COUNT(*) AS n
                    FROM appearances
                    WHERE global_person_id = ? AND session_id = ? AND track_id = ?
                      AND action_label IS NOT NULL AND action_label != ''
                    GROUP BY action_label
                    ORDER BY n DESC LIMIT 1
                    """,
                    (global_person_id, row["session_id"], row["track_id"]),
                ).fetchone()
                row["dominant_action"] = actions["action_label"] if actions else ""
                out.append(row)
        return out
