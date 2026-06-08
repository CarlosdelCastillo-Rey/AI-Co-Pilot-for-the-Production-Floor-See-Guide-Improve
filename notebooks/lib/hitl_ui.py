"""Tinder-style human review UI (ipywidgets) for HAR session crops."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from lib.constants import TRAINABLE_ACTIONS
from lib.human_labels import append_label, build_review_queue, save_review_queue


def launch_hitl_carousel(
    *,
    limit: int = 100,
    reviewer: str = "supervisor",
    on_saved: Callable[[dict[str, Any]], None] | None = None,
):
    """
    Interactive review widget for Jupyter (single panel per cell).

    Two-step labels per crop:
    1. Person? — yes / no / don't know
    2. Action? — OK / wrong / maybe / skip (+ optional correct class)
    """
    import ipywidgets as widgets
    from IPython.display import clear_output, display

    queue = build_review_queue(limit=limit)
    if not queue:
        print("No review candidates — run notebook 04 eval first (or import v1 sessions).")
        return None

    state = {"index": 0, "queue": queue, "saved": 0, "saved_ids": set()}

    img = widgets.Image(format="jpeg", width=480, height=360)
    status = widgets.HTML()
    saved_lbl = widgets.HTML(value="<i>Saved: 0</i>")
    notes = widgets.Textarea(
        value="",
        placeholder="Optional notes",
        layout=widgets.Layout(width="480px", height="60px"),
    )
    correct_label = widgets.Dropdown(
        options=[("— pick if action wrong —", "")] + [(a, a) for a in TRAINABLE_ACTIONS],
        description="Correct action:",
        layout=widgets.Layout(width="480px"),
    )
    person_toggle = widgets.ToggleButtons(
        options=[
            ("✓ Is a person", "yes"),
            ("✗ Not a person", "no"),
            ("? Person unsure", "unknown"),
        ],
        value="yes",
        description="",
        tooltips=[
            "YOLO crop shows a real operator",
            "False detection / object / empty crop",
            "Cannot tell from this crop",
        ],
        layout=widgets.Layout(width="480px"),
    )

    def _render() -> None:
        item = state["queue"][state["index"]]
        path = Path(item["image_path"])
        if path.is_file():
            img.value = path.read_bytes()
        pred = item.get("predicted_label") or "—"
        conf = float(item.get("confidence") or 0)
        ent = float(item.get("entropy") or 0)
        person_labels = {"yes": "Is a person", "no": "Not a person", "unknown": "Person unsure"}
        person_lbl = person_labels.get(str(person_toggle.value), person_toggle.value)
        status.value = (
            f"<b>{state['index'] + 1}/{len(state['queue'])}</b> · "
            f"pred: <code>{pred}</code> ({conf:.1%}) · entropy {ent:.2f}<br>"
            f"video: {item.get('video') or '—'} · track {item.get('track_id')} · "
            f"frame {item.get('frame_idx')} · <b>Person:</b> {person_lbl}"
        )
        correct_label.value = ""
        if person_toggle.value != "yes":
            person_toggle.value = "yes"

    def _persist(
        action_verdict: str,
        *,
        person_verdict: str | None = None,
        advance: bool = True,
    ) -> bool:
        """Write row to CSV. Returns False if validation blocked save."""
        person = person_verdict if person_verdict is not None else str(person_toggle.value)
        item = state["queue"][state["index"]]
        eid = item["event_id"]
        correct = str(correct_label.value or "").strip()

        if action_verdict == "no" and not correct and person == "yes":
            status.value += "<br><span style='color:red'>Pick correct action when marking Wrong.</span>"
            return False

        if eid in state["saved_ids"]:
            if advance:
                _advance_only()
            return True

        record = append_label(
            {
                "event_id": eid,
                "session_dir": item["session_dir"],
                "crop_path": item.get("crop_path") or item["image_path"],
                "frame_path": item.get("frame_path") or "",
                "embedding_path": item.get("embedding_path") or "",
                "predicted_label": item.get("predicted_label"),
                "confidence": item.get("confidence"),
                "entropy": item.get("entropy"),
                "priority_score": item.get("priority_score"),
                "action_verdict": action_verdict,
                "correct_label": correct if action_verdict == "no" else "",
                "person_verdict": person,
                "reviewer": reviewer,
                "source": "hitl_ui",
                "video": item.get("video") or "",
                "track_id": item.get("track_id"),
                "frame_idx": item.get("frame_idx"),
                "notes": notes.value.strip(),
            }
        )
        state["saved_ids"].add(eid)
        state["saved"] += 1
        saved_lbl.value = f"<i>Saved: {state['saved']}</i>"
        if on_saved:
            on_saved(record)
        if advance:
            _advance_only()
        return True

    def _save_verdict(action_verdict: str, *, person_verdict: str | None = None) -> None:
        _persist(action_verdict, person_verdict=person_verdict, advance=True)

    def _advance_only() -> None:
        if state["index"] < len(state["queue"]) - 1:
            state["index"] += 1
        notes.value = ""
        _render()

    def _next_with_save() -> None:
        """Save current person/action choices, then go to next crop."""
        item = state["queue"][state["index"]]
        if item["event_id"] in state["saved_ids"]:
            _advance_only()
            return
        correct = str(correct_label.value or "").strip()
        person = str(person_toggle.value)
        if person == "no":
            _persist("dont_know", person_verdict="no")
        elif person == "unknown":
            _persist("dont_know", person_verdict="unknown")
        elif correct:
            _persist("no")
        else:
            _persist("dont_know")

    def _advance() -> None:
        _advance_only()

    def _back() -> None:
        state["index"] = max(0, state["index"] - 1)
        _render()

    btn_yes = widgets.Button(description="✓ Action OK", button_style="success")
    btn_no = widgets.Button(description="✗ Wrong action", button_style="danger")
    btn_maybe = widgets.Button(description="? Action maybe")
    btn_skip = widgets.Button(description="Skip action")
    btn_not_person = widgets.Button(description="Save: not a person", button_style="warning")
    btn_prev = widgets.Button(description="← Prev")
    btn_next = widgets.Button(description="Save & Next →", button_style="info")

    btn_yes.on_click(lambda _: _save_verdict("yes"))
    btn_no.on_click(lambda _: _save_verdict("no"))
    btn_maybe.on_click(lambda _: _save_verdict("maybe"))
    btn_skip.on_click(lambda _: _save_verdict("dont_know"))
    btn_not_person.on_click(lambda _: _save_verdict("dont_know", person_verdict="no"))
    btn_prev.on_click(lambda _: _back())
    btn_next.on_click(lambda _: _next_with_save())

    person_header = widgets.HTML("<b>1 — Is this crop a person?</b>")
    action_header = widgets.HTML("<b>2 — Is the predicted action correct?</b>")
    row_action = widgets.HBox([btn_yes, btn_no, btn_maybe, btn_skip])
    row_nav = widgets.HBox([btn_not_person, btn_prev, btn_next, saved_lbl])

    ui = widgets.VBox(
        [
            status,
            img,
            person_header,
            person_toggle,
            action_header,
            row_action,
            correct_label,
            notes,
            row_nav,
        ]
    )

    _render()
    save_review_queue(limit=limit)

    # Replace entire cell output — avoids stacking when reload_lib_modules() resets module state
    clear_output(wait=True)
    display(ui)
    return None


def queue_stats(limit: int = 200) -> dict[str, Any]:
    q = build_review_queue(limit=limit)
    return {"n_candidates": len(q), "top_priority": q[0] if q else None}
