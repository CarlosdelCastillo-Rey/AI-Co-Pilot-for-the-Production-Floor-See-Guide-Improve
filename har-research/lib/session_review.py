"""Load HAR session logs, summarize per track, and support track-level human review."""

from __future__ import annotations

import csv
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from lib.paths import HUMAN_LABELS_DIR, SESSIONS_DIR
from lib.session_log import list_sessions

TRACK_LABELS_CSV = HUMAN_LABELS_DIR / "track_labels.csv"
TRACK_LABEL_COLUMNS = [
    "label_id",
    "session_id",
    "track_id",
    "video",
    "person_verdict",   # yes | no | unknown
    "action_notes",
    "reviewer",
    "reviewed_at",
    "n_events",
    "dominant_action",
    "dominant_confidence",
    "global_person_id",
    "display_name",
]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def session_dir_from_row(row: dict[str, Any]) -> Path:
    rel = row.get("session_dir") or ""
    return SESSIONS_DIR / rel


def load_manifest(session_dir: Path) -> dict[str, Any]:
    path = session_dir / "manifest.json"
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def load_summary(session_dir: Path) -> dict[str, Any]:
    path = session_dir / "summary.json"
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def load_events(session_dir: Path) -> list[dict[str, Any]]:
    path = session_dir / "events.jsonl"
    if not path.is_file():
        return []
    out: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            out.append(json.loads(line))
    return out


def resolve_artifact(session_dir: Path, rel: str | None) -> Path | None:
    if not rel:
        return None
    p = SESSIONS_DIR / rel
    return p if p.is_file() else None


def list_registered_persons(limit: int = 100) -> list[dict[str, Any]]:
    from lib.person_registry import PersonRegistry

    return PersonRegistry().list_persons(limit=limit)


def person_history_df(global_person_id: str, limit: int = 200):
    import pandas as pd

    from lib.person_registry import PersonRegistry

    rows = PersonRegistry().person_history(global_person_id, limit=limit)
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def persons_dataframe(limit: int = 100):
    import pandas as pd

    rows = list_registered_persons(limit=limit)
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def events_dataframe(session_dir: Path):
    import pandas as pd

    events = load_events(session_dir)
    if not events:
        return pd.DataFrame()
    df = pd.DataFrame(events)
    for col in ("frame_path", "crop_path", "embedding_path"):
        if col in df.columns:
            df[f"{col}_abs"] = df[col].apply(
                lambda r: str(resolve_artifact(session_dir, r) or "")
            )
    return df.sort_values(["track_id", "frame_idx"]).reset_index(drop=True)


def tracks_summary(session_dir: Path) -> list[dict[str, Any]]:
    """One row per track_id with inference stats."""
    import pandas as pd

    df = events_dataframe(session_dir)
    if df.empty:
        return []

    manifest = load_manifest(session_dir)
    video = manifest.get("video") or ""
    session_id = manifest.get("session_id") or session_dir.name.split("_")[0]

    rows: list[dict[str, Any]] = []
    for tid, grp in df.groupby("track_id"):
        grp = grp.sort_values("frame_idx")
        labels = grp["action_label"].fillna(grp["raw_label"]).astype(str)
        labels = labels[labels.str.len() > 0]
        dominant = labels.mode().iloc[0] if len(labels) else ""
        dom_rows = grp[grp["action_label"] == dominant] if dominant else grp.iloc[0:0]
        dom_conf = float(dom_rows["confidence"].max()) if len(dom_rows) else 0.0

        gid = None
        if "global_person_id" in grp.columns:
            gids = grp["global_person_id"].dropna().astype(str)
            gids = gids[gids.str.len() > 0]
            if len(gids):
                gid = gids.mode().iloc[0]

        first_crop = None
        first_frame = None
        for _, ev in grp.iterrows():
            cp = ev.get("crop_path_abs") or ev.get("frame_path_abs")
            if cp:
                first_crop = cp
                break
        for _, ev in grp.iterrows():
            fp = ev.get("frame_path_abs")
            if fp:
                first_frame = fp
                break

        rows.append(
            {
                "session_id": session_id,
                "track_id": int(tid),
                "video": video,
                "global_person_id": gid or "",
                "n_inferences": len(grp),
                "frame_first": int(grp["frame_idx"].min()),
                "frame_last": int(grp["frame_idx"].max()),
                "dominant_action": dominant,
                "dominant_confidence": round(dom_conf, 4),
                "label_changes": int(grp["label_changed"].fillna(False).astype(bool).sum()),
                "uncertain_count": int(grp["uncertain"].fillna(False).astype(bool).sum()),
                "sample_crop": first_crop or "",
                "sample_frame": first_frame or "",
            }
        )
    return sorted(rows, key=lambda r: r["track_id"])


def ensure_track_labels_store() -> Path:
    HUMAN_LABELS_DIR.mkdir(parents=True, exist_ok=True)
    if not TRACK_LABELS_CSV.is_file():
        with TRACK_LABELS_CSV.open("w", encoding="utf-8", newline="") as f:
            csv.DictWriter(f, fieldnames=TRACK_LABEL_COLUMNS).writeheader()
    return TRACK_LABELS_CSV


def load_track_labels_df():
    import pandas as pd

    ensure_track_labels_store()
    return pd.read_csv(TRACK_LABELS_CSV)


def append_track_label(row: dict[str, Any]) -> dict[str, Any]:
    ensure_track_labels_store()
    record = {col: row.get(col, "") for col in TRACK_LABEL_COLUMNS}
    if not record["label_id"]:
        record["label_id"] = f"tl-{uuid.uuid4().hex[:10]}"
    if not record["reviewed_at"]:
        record["reviewed_at"] = _utc_now()

    gid = str(record.get("global_person_id") or "").strip()
    name = str(record.get("display_name") or "").strip()
    verdict = str(record.get("person_verdict") or "").strip().lower()
    if gid and name and verdict == "yes":
        from lib.person_registry import PersonRegistry

        PersonRegistry().set_display_name(gid, name)

    with TRACK_LABELS_CSV.open("a", encoding="utf-8", newline="") as f:
        csv.DictWriter(f, fieldnames=TRACK_LABEL_COLUMNS).writerow(record)
    return record


def track_label_lookup(session_id: str) -> dict[int, dict[str, Any]]:
    if not TRACK_LABELS_CSV.is_file():
        return {}
    out: dict[int, dict[str, Any]] = {}
    with TRACK_LABELS_CSV.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("session_id") != session_id:
                continue
            try:
                out[int(row["track_id"])] = dict(row)
            except (TypeError, ValueError):
                continue
    return out


def plot_track_timeline(session_dir: Path, track_id: int, *, ax=None):
    """Frame index vs predicted action confidence for one track."""
    import matplotlib.pyplot as plt

    df = events_dataframe(session_dir)
    sub = df[df["track_id"] == track_id].sort_values("frame_idx")
    if sub.empty:
        raise ValueError(f"No events for track {track_id}")

    if ax is None:
        _, ax = plt.subplots(figsize=(10, 3))
    ax.plot(sub["frame_idx"], sub["confidence"], marker="o", ms=4, lw=1)
    for _, row in sub.iterrows():
        lbl = row.get("action_label") or row.get("raw_label") or "?"
        ax.annotate(
            str(lbl)[:14],
            (row["frame_idx"], row["confidence"]),
            fontsize=7,
            alpha=0.85,
            rotation=25,
            ha="left",
        )
    ax.set_xlabel("Frame index")
    ax.set_ylabel("Confidence")
    ax.set_title(f"Track #{track_id} — inference timeline")
    ax.grid(True, alpha=0.3)
    return ax


def launch_track_tagger(
    session_dir: Path,
    *,
    reviewer: str = "supervisor",
):
    """Tag each YOLO track as person / not person for the whole session."""
    import ipywidgets as widgets
    from IPython.display import clear_output, display

    session_dir = Path(session_dir)
    tracks = tracks_summary(session_dir)
    if not tracks:
        print("No tracks in session — run eval with logging first.")
        return None

    manifest = load_manifest(session_dir)
    session_id = manifest.get("session_id") or session_dir.name.split("_")[0]
    existing = track_label_lookup(session_id)

    state = {"index": 0, "saved": 0}

    img = widgets.Image(format="jpeg", width=420, height=420)
    status = widgets.HTML()
    notes = widgets.Textarea(
        placeholder="Notes about this track (optional)",
        layout=widgets.Layout(width="420px", height="50px"),
    )
    display_name = widgets.Text(
        placeholder="Person name (optional — saved to registry)",
        layout=widgets.Layout(width="420px"),
    )
    person_toggle = widgets.ToggleButtons(
        options=[("✓ Real person", "yes"), ("✗ Not a person", "no"), ("? Unsure", "unknown")],
        value="yes",
        layout=widgets.Layout(width="420px"),
    )

    def _render() -> None:
        tr = tracks[state["index"]]
        path = tr.get("sample_crop") or tr.get("sample_frame")
        if path and Path(path).is_file():
            img.value = Path(path).read_bytes()
        else:
            img.value = b""
        prev = existing.get(tr["track_id"], {})
        prev_v = prev.get("person_verdict") or "—"
        gid = tr.get("global_person_id") or ""
        prev_name = prev.get("display_name") or ""
        if not prev_name and gid:
            from lib.person_registry import PersonRegistry

            person = PersonRegistry().get_person(gid)
            if person and person.get("display_name"):
                prev_name = person["display_name"]
        display_name.value = str(prev_name) if prev_name else ""
        gid_line = f"<br>Global person: <code>{gid}</code>" if gid else ""
        status.value = (
            f"<b>Track {state['index'] + 1}/{len(tracks)}</b> · "
            f"<code>#{tr['track_id']}</code> · frames {tr['frame_first']}–{tr['frame_last']}"
            f"{gid_line}<br>"
            f"Inferences: {tr['n_inferences']} · dominant: <code>{tr['dominant_action'] or '—'}</code> "
            f"({tr['dominant_confidence']:.1%}) · prev tag: <b>{prev_v}</b>"
        )

    def _save(advance: bool = True) -> None:
        tr = tracks[state["index"]]
        append_track_label(
            {
                "session_id": session_id,
                "track_id": tr["track_id"],
                "video": tr.get("video") or manifest.get("video") or "",
                "person_verdict": str(person_toggle.value),
                "action_notes": notes.value.strip(),
                "reviewer": reviewer,
                "n_events": tr["n_inferences"],
                "dominant_action": tr.get("dominant_action") or "",
                "dominant_confidence": tr.get("dominant_confidence") or "",
                "global_person_id": tr.get("global_person_id") or "",
                "display_name": display_name.value.strip(),
            }
        )
        existing[tr["track_id"]] = {
            "person_verdict": person_toggle.value,
            "display_name": display_name.value.strip(),
        }
        state["saved"] += 1
        notes.value = ""
        if advance and state["index"] < len(tracks) - 1:
            state["index"] += 1
        _render()

    btn_save = widgets.Button(description="Save track tag", button_style="success")
    btn_next = widgets.Button(description="Skip →")
    btn_prev = widgets.Button(description="← Prev")
    btn_save.on_click(lambda _: _save(True))
    btn_next.on_click(lambda _: (_inc(), _render()))
    btn_prev.on_click(lambda _: (_dec(), _render()))

    def _inc() -> None:
        state["index"] = min(len(tracks) - 1, state["index"] + 1)

    def _dec() -> None:
        state["index"] = max(0, state["index"] - 1)

    ui = widgets.VBox(
        [
            widgets.HTML(f"<b>Session</b> {session_id} · {manifest.get('video') or ''}"),
            status,
            img,
            widgets.HTML("<b>Is this track a real person?</b>"),
            person_toggle,
            widgets.HTML("<b>Name this person</b> (links to global registry when Re-ID ran)"),
            display_name,
            notes,
            widgets.HBox([btn_prev, btn_save, btn_next]),
        ]
    )
    _render()
    clear_output(wait=True)
    display(ui)
    return None


def plot_person_history(global_person_id: str, *, ax=None, limit: int = 500):
    """Scatter timeline of all appearances for one global person (across sessions)."""
    import matplotlib.pyplot as plt

    from lib.person_registry import PersonRegistry

    hist = PersonRegistry().person_history(global_person_id, limit=limit)
    if not hist:
        raise ValueError(f"No appearances for {global_person_id}")

    import pandas as pd

    df = pd.DataFrame(hist).sort_values("ts")
    df["session_short"] = df["session_id"].astype(str).str.slice(0, 14)
    if ax is None:
        _, ax = plt.subplots(figsize=(11, 3.5))

    sessions = df["session_short"].unique()
    palette = plt.cm.tab10(range(len(sessions)))
    color_map = {s: palette[i] for i, s in enumerate(sessions)}

    for _, row in df.iterrows():
        ax.scatter(
            row["ts"],
            float(row.get("confidence") or 0),
            c=[color_map.get(row["session_short"], "#888")],
            s=40,
            alpha=0.85,
        )
        lbl = str(row.get("action_label") or "?")[:12]
        ax.annotate(lbl, (row["ts"], float(row.get("confidence") or 0)), fontsize=6, alpha=0.7)

    ax.set_xlabel("Time (UTC)")
    ax.set_ylabel("Confidence")
    person = PersonRegistry().get_person(global_person_id) or {}
    title_name = person.get("display_name") or global_person_id
    ax.set_title(f"{title_name} — {len(df)} appearances · {len(sessions)} session(s)")
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=25, ha="right")
    return ax


def launch_person_registry_viewer(*, limit: int = 100):
    """Browse registered persons, cross-session history, and edit display names."""
    import ipywidgets as widgets
    from IPython.display import clear_output, display

    from lib.person_registry import PersonRegistry, REGISTRY_DB

    reg = PersonRegistry()
    persons = reg.list_persons(limit=limit)
    if not persons:
        print(f"No persons in registry yet — run 05 eval with logging (DB: {REGISTRY_DB})")
        return None

    options = []
    for p in persons:
        gid = p["global_person_id"]
        name = p.get("display_name") or "(unnamed)"
        n = p.get("n_appearances") or 0
        options.append((f"{name} · {gid} · {n} sightings", gid))

    picker = widgets.Dropdown(options=options, description="Person", layout=widgets.Layout(width="520px"))
    name_input = widgets.Text(description="Name", layout=widgets.Layout(width="420px"))
    status = widgets.HTML()
    sessions_html = widgets.HTML()
    history_out = widgets.Output()

    def _refresh(gid: str) -> None:
        person = reg.get_person(gid) or {}
        name_input.value = person.get("display_name") or ""
        status.value = (
            f"<b>{gid}</b><br>"
            f"Appearances: {person.get('n_appearances', 0)} · "
            f"First: {person.get('first_seen_at', '?')} · "
            f"Last: {person.get('last_seen_at', '?')}"
        )
        rows = reg.person_sessions_summary(gid)
        if rows:
            trs = "".join(
                f"<tr><td>{r['session_id'][:16]}…</td>"
                f"<td>#{r['track_id']}</td><td>{r.get('video') or ''}</td>"
                f"<td>{r['n_appearances']}</td>"
                f"<td>{r.get('dominant_action') or '—'}</td></tr>"
                for r in rows
            )
            sessions_html.value = (
                "<b>Sessions / tracks</b>"
                "<table style='font-size:12px;border-collapse:collapse'>"
                "<tr><th>Session</th><th>Track</th><th>Video</th><th>Inferences</th><th>Top action</th></tr>"
                + trs + "</table>"
            )
        else:
            sessions_html.value = "<i>No session rows</i>"

        with history_out:
            history_out.clear_output(wait=True)
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(11, 3.5))
            try:
                plot_person_history(gid, ax=ax)
            except ValueError as e:
                ax.text(0.5, 0.5, str(e), ha="center", va="center", transform=ax.transAxes)
            plt.tight_layout()
            plt.show()

    def _on_pick(change) -> None:
        if change["new"]:
            _refresh(change["new"])

    def _save_name(_btn) -> None:
        gid = picker.value
        if gid and name_input.value.strip():
            reg.set_display_name(gid, name_input.value.strip())
            status.value = status.value + f"<br><span style='color:green'>Saved name: {name_input.value.strip()}</span>"

    picker.observe(_on_pick, names="value")
    btn_save = widgets.Button(description="Save name", button_style="success")
    btn_save.on_click(_save_name)

    ui = widgets.VBox([
        widgets.HTML("<h3>Person registry</h3><p>Stable IDs from Re-ID across videos/sessions.</p>"),
        picker,
        status,
        widgets.HBox([name_input, btn_save]),
        sessions_html,
        widgets.HTML("<b>Appearance timeline (all sessions)</b>"),
        history_out,
    ])
    _refresh(picker.value)
    clear_output(wait=True)
    display(ui)
    return None
