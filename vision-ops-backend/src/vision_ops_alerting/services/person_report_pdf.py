"""Generate a styled person activity report PDF (fpdf2 + Pillow)."""
from __future__ import annotations

import io
import math
import tempfile
import uuid
from datetime import datetime, timezone
from typing import Any

BLUE_DARK = (0, 72, 154)
BLUE_MID = (0, 89, 187)
BLUE_LIGHT = (0, 112, 234)
BLUE_BG = (238, 243, 255)
OK_GREEN = (31, 138, 91)
WARN_AMB = (183, 121, 31)
INK = (20, 24, 29)
DIM = (90, 98, 108)
FAINT = (148, 155, 165)
LINE = (231, 234, 239)
WHITE = (255, 255, 255)
CARD = (255, 255, 255)


def _conf_color(conf: float) -> tuple[int, int, int]:
    if conf >= 0.70:
        return OK_GREEN
    if conf >= 0.50:
        return WARN_AMB
    return (192, 54, 44)


def generate_person_pdf(
    report: dict[str, Any],
    *,
    thumbnail_bytes: bytes | None = None,
    action_crops: dict[str, bytes] | None = None,
) -> bytes:
    from fpdf import FPDF

    name = report.get("display_name") or "Unknown"
    pid = report.get("global_person_id", "")
    metrics = report.get("metrics", {})
    actions = report.get("action_breakdown", [])
    sessions = report.get("sessions", [])

    first_seen = metrics.get("first_seen_at") or report.get("first_seen_at") or ""
    last_seen = metrics.get("last_seen_at") or report.get("last_seen_at") or ""
    dominant = metrics.get("dominant_action") or "—"
    avg_conf = float(metrics.get("avg_confidence") or 0)
    n_events = int(metrics.get("n_events") or 0)
    n_sessions = int(metrics.get("n_sessions") or len(sessions))
    n_tracks = int(metrics.get("n_tracks") or 0)
    n_label_changes = int(metrics.get("n_label_changes") or 0)
    avg_reid = float(metrics.get("avg_reid_match") or 0)

    def _fmt_dt(s: str) -> str:
        if not s:
            return "—"
        try:
            dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
            return dt.strftime("%d %b %Y  %H:%M UTC")
        except Exception:
            return s[:19]

    generated = datetime.now(timezone.utc).strftime("%d %b %Y  %H:%M UTC")
    report_id = f"PAR-{datetime.now(timezone.utc).strftime('%Y%m%d')}-{uuid.uuid4().hex[:6].upper()}"

    pdf = FPDF(orientation="P", unit="mm", format="A4")
    pdf.set_auto_page_break(auto=True, margin=18)
    pdf.add_page()
    W = pdf.w  # 210

    # ── Header band ──────────────────────────────────────────────────────────
    pdf.set_fill_color(*BLUE_DARK)
    pdf.rect(0, 0, W, 52, "F")

    # accent circle
    pdf.set_fill_color(*BLUE_MID)
    pdf.ellipse(W - 38, -20, 70, 70, "F")

    # Logo mark
    pdf.set_fill_color(*WHITE)
    pdf.set_draw_color(*WHITE)
    pdf.rect(10, 8, 14, 14, "F")
    pdf.set_fill_color(*BLUE_DARK)
    pdf.set_font("Helvetica", "B", 9)
    pdf.set_text_color(*BLUE_DARK)
    pdf.set_xy(10, 10.5)
    pdf.cell(14, 9, "VO", align="C")

    # Brand
    pdf.set_text_color(*WHITE)
    pdf.set_font("Helvetica", "B", 13)
    pdf.set_xy(27, 8)
    pdf.cell(80, 7, "VisionOps")
    pdf.set_font("Helvetica", "", 7.5)
    pdf.set_xy(27, 15)
    pdf.set_text_color(200, 215, 240)
    pdf.cell(80, 5, "Industrial AI  ·  Edge")

    # Report ID top-right
    pdf.set_font("Helvetica", "", 8)
    pdf.set_text_color(200, 215, 240)
    pdf.set_xy(W - 68, 8)
    pdf.cell(58, 5, f"REPORT  {report_id}", align="R")
    pdf.set_xy(W - 68, 14)
    pdf.cell(58, 5, "Person Activity Report", align="R")

    # Person name
    pdf.set_text_color(*WHITE)
    pdf.set_font("Helvetica", "B", 22)
    pdf.set_xy(10, 26)
    pdf.cell(W - 20, 12, name, align="L")

    # Sub-line
    pdf.set_font("Helvetica", "", 8.5)
    pdf.set_text_color(200, 215, 240)
    pdf.set_xy(10, 39)
    pdf.cell(80, 6, f"ID: {pid}")
    pdf.set_xy(W - 80, 39)
    pdf.cell(70, 6, f"Generated: {generated}", align="R")

    # ── Thumbnail ─────────────────────────────────────────────────────────────
    y_body = 60
    thumb_w = 36
    thumb_x = W - thumb_w - 12
    if thumbnail_bytes:
        try:
            from PIL import Image
            img = Image.open(io.BytesIO(thumbnail_bytes)).convert("RGB")
            img.thumbnail((180, 240))
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                img.save(tmp.name, "JPEG", quality=88)
                pdf.set_draw_color(*LINE)
                pdf.rect(thumb_x - 1, y_body - 1, thumb_w + 2, thumb_w + 2)
                pdf.image(tmp.name, thumb_x, y_body, thumb_w, thumb_w)
        except Exception:
            thumbnail_bytes = None

    content_w = (W - 24 - (thumb_w + 14 if thumbnail_bytes else 0))

    # ── KPI strip ────────────────────────────────────────────────────────────
    kpis = [
        ("Appearances", str(n_events)),
        ("Sessions", str(n_sessions)),
        ("Tracks", str(n_tracks)),
        ("Avg Conf.", f"{avg_conf:.0%}"),
        ("Re-ID Score", f"{avg_reid:.2f}"),
    ]
    kpi_n = len(kpis)
    kpi_w = content_w / kpi_n
    kpi_h = 22
    kpi_y = y_body

    for i, (label, value) in enumerate(kpis):
        kx = 12 + i * kpi_w
        pdf.set_fill_color(*BLUE_BG)
        pdf.set_draw_color(*LINE)
        pdf.rect(kx, kpi_y, kpi_w - 2, kpi_h, "FD")
        pdf.set_font("Helvetica", "", 6.5)
        pdf.set_text_color(*FAINT)
        pdf.set_xy(kx + 3, kpi_y + 3)
        pdf.cell(kpi_w - 6, 4, label.upper())
        pdf.set_font("Helvetica", "B", 15)
        pdf.set_text_color(*BLUE_DARK)
        pdf.set_xy(kx + 3, kpi_y + 8)
        pdf.cell(kpi_w - 6, 10, value)

    # Dominant action banner
    banner_y = kpi_y + kpi_h + 4
    pdf.set_fill_color(*OK_GREEN)
    pdf.rect(12, banner_y, content_w - 2, 9, "F")
    pdf.set_font("Helvetica", "", 7)
    pdf.set_text_color(*WHITE)
    pdf.set_xy(16, banner_y + 1)
    pdf.cell(40, 7, "DOMINANT ACTION")
    pdf.set_font("Helvetica", "B", 9)
    pdf.set_xy(55, banner_y + 1)
    pdf.cell(content_w - 46, 7, dominant)

    # Timeline strip
    tl_y = banner_y + 12
    pdf.set_font("Helvetica", "", 7)
    pdf.set_text_color(*DIM)
    pdf.set_xy(12, tl_y)
    pdf.cell(content_w // 2, 5, f"First seen: {_fmt_dt(first_seen)}")
    pdf.set_xy(12 + content_w // 2, tl_y)
    pdf.cell(content_w // 2, 5, f"Last seen: {_fmt_dt(last_seen)}", align="R")

    # ── Action Breakdown ─────────────────────────────────────────────────────
    sec_y = max(tl_y + 8, y_body + thumb_w + 4 if thumbnail_bytes else tl_y + 8)

    def _section(title: str, count: int | None = None) -> float:
        nonlocal sec_y
        pdf.set_fill_color(*BLUE_DARK)
        pdf.rect(12, sec_y, 3, 7, "F")
        pdf.set_font("Helvetica", "B", 10)
        pdf.set_text_color(*INK)
        pdf.set_xy(18, sec_y)
        pdf.cell(60, 7, title)
        if count is not None:
            pdf.set_fill_color(*BLUE_BG)
            pdf.set_font("Helvetica", "B", 7)
            pdf.set_text_color(*BLUE_MID)
            pdf.set_xy(80, sec_y + 1)
            pdf.cell(16, 5, str(count), align="C", fill=True)
        pdf.set_draw_color(*LINE)
        pdf.line(12, sec_y + 7.5, W - 12, sec_y + 7.5)
        sec_y += 11
        return sec_y

    _section("Action Breakdown", len(actions))

    bar_max_w = content_w - 30
    for act in actions[:8]:
        label = act.get("action") or "—"
        share = float(act.get("share") or 0)
        count = int(act.get("count") or 0)
        conf = float(act.get("avg_confidence") or 0)
        bar_w = max(2.0, bar_max_w * share)

        pdf.set_font("Helvetica", "", 8)
        pdf.set_text_color(*DIM)
        pdf.set_xy(12, sec_y)
        pdf.cell(55, 5, label[:28])

        # bar background
        pdf.set_fill_color(*LINE)
        pdf.rect(70, sec_y + 1, bar_max_w, 3.5, "F")
        # bar fill
        pdf.set_fill_color(*BLUE_LIGHT)
        pdf.rect(70, sec_y + 1, bar_w, 3.5, "F")

        # count + share
        pdf.set_font("Helvetica", "B", 7.5)
        pdf.set_text_color(*INK)
        pdf.set_xy(70 + bar_max_w + 2, sec_y)
        pdf.cell(18, 5, f"{count}  {share:.0%}", align="R")

        # conf dot
        pdf.set_fill_color(*_conf_color(conf))
        pdf.ellipse(W - 22, sec_y + 1.2, 3, 3, "F")
        pdf.set_font("Helvetica", "", 6.5)
        pdf.set_text_color(*DIM)
        pdf.set_xy(W - 18, sec_y)
        pdf.cell(8, 5, f"{conf:.0%}")

        sec_y += 6.5

    # Legend
    pdf.set_font("Helvetica", "", 6)
    pdf.set_text_color(*FAINT)
    pdf.set_xy(W - 38, sec_y)
    pdf.cell(26, 4, "* avg confidence", align="R")
    sec_y += 6

    # ── Label changes note ───────────────────────────────────────────────────
    if n_label_changes:
        pdf.set_fill_color(255, 247, 230)
        pdf.set_draw_color(236, 220, 181)
        pdf.rect(12, sec_y, W - 24, 8, "FD")
        pdf.set_font("Helvetica", "", 7.5)
        pdf.set_text_color(*WARN_AMB)
        pdf.set_xy(15, sec_y + 1.5)
        pdf.cell(W - 30, 5, f"Action label changed {n_label_changes}x during observation window")
        sec_y += 11

    # ── Sessions ─────────────────────────────────────────────────────────────
    sec_y += 2
    _section("Sessions", len(sessions))

    # Table header
    col_w = [(content_w * p) for p in [0.32, 0.22, 0.24, 0.22]]
    headers = ["Video", "Dominant Action", "Appearances", "Re-ID Score"]
    pdf.set_fill_color(*BLUE_DARK)
    pdf.rect(12, sec_y - 1, W - 24, 7, "F")
    pdf.set_font("Helvetica", "B", 7)
    pdf.set_text_color(*WHITE)
    cx = 14
    for h, cw in zip(headers, col_w):
        pdf.set_xy(cx, sec_y)
        pdf.cell(cw, 5, h)
        cx += cw
    sec_y += 7

    for i, sess in enumerate(sessions[:12]):
        video = (sess.get("video") or "—")[:22]
        dom = (sess.get("dominant_action") or "—")[:20]
        n_app = str(sess.get("n_appearances") or "—")
        reid = f"{sess.get('match_score', 0):.3f}" if sess.get("match_score") else "—"

        bg = CARD if i % 2 == 0 else (248, 249, 252)
        pdf.set_fill_color(*bg)
        pdf.set_draw_color(*LINE)
        pdf.rect(12, sec_y, W - 24, 7, "FD")

        pdf.set_font("Helvetica", "", 7.5)
        pdf.set_text_color(*DIM)
        cx = 14
        for val, cw in zip([video, dom, n_app, reid], col_w):
            pdf.set_xy(cx, sec_y + 1)
            pdf.cell(cw, 5, val)
            cx += cw
        sec_y += 7

    # ── Action Detection Gallery ──────────────────────────────────────────────
    if action_crops:
        sec_y += 6
        _section("Action Detection Snapshots", len(action_crops))

        cols = 3
        cell_w = (W - 24) / cols          # ~62 mm per column
        img_h = 46.0                       # image height in mm
        caption_h = 9.0                    # space below image for label + conf
        cell_h = img_h + caption_h + 3    # total cell height

        # Order crops by action_breakdown share so dominant action appears first
        ab_order = [a.get("action") for a in actions if a.get("action")]
        def _sort_key(item: tuple[str, bytes]) -> int:
            try:
                return ab_order.index(item[0])
            except ValueError:
                return 999

        ordered_crops = sorted(action_crops.items(), key=_sort_key)
        n_rows = math.ceil(len(ordered_crops) / cols)

        # If not enough room on current page, start a new one
        if sec_y + n_rows * cell_h > pdf.h - 22:
            pdf.add_page()
            sec_y = 20

        for idx, (action_label, crop_bytes) in enumerate(ordered_crops):
            col = idx % cols
            row = idx // cols
            cx = 12 + col * cell_w
            cy = sec_y + row * cell_h

            # Fetch the matching confidence from action_breakdown
            conf_val: float | None = None
            for ab in actions:
                if ab.get("action") == action_label:
                    conf_val = ab.get("avg_confidence")
                    break

            # Draw image
            try:
                from PIL import Image as PILImage
                pil_img = PILImage.open(io.BytesIO(crop_bytes)).convert("RGB")
                pil_img.thumbnail((260, 260))
                with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                    pil_img.save(tmp.name, "JPEG", quality=88)
                    pdf.set_draw_color(*LINE)
                    pdf.rect(cx, cy, cell_w - 2, img_h, "D")
                    pdf.image(tmp.name, cx + 1, cy + 1, cell_w - 4, img_h - 2)
            except Exception:
                pdf.set_fill_color(*LINE)
                pdf.rect(cx, cy, cell_w - 2, img_h, "F")

            # Action label
            pdf.set_font("Helvetica", "B", 7)
            pdf.set_text_color(*INK)
            pdf.set_xy(cx, cy + img_h + 1.5)
            pdf.cell(cell_w - 2, 4, action_label[:28], align="C")

            # Confidence dot + value
            if conf_val is not None:
                dot_color = _conf_color(conf_val)
                pdf.set_fill_color(*dot_color)
                pdf.ellipse(cx + (cell_w - 2) / 2 - 9, cy + img_h + 5.5, 2.5, 2.5, "F")
                pdf.set_font("Helvetica", "", 6.5)
                pdf.set_text_color(*DIM)
                pdf.set_xy(cx + (cell_w - 2) / 2 - 5, cy + img_h + 5)
                pdf.cell(20, 4, f"{conf_val:.0%} avg conf")

        sec_y += n_rows * cell_h + 4

    # ── Footer ───────────────────────────────────────────────────────────────
    pdf.set_y(-16)
    pdf.set_draw_color(*LINE)
    pdf.line(12, pdf.get_y(), W - 12, pdf.get_y())
    pdf.set_font("Helvetica", "", 7)
    pdf.set_text_color(*FAINT)
    pdf.set_xy(12, pdf.get_y() + 2)
    pdf.cell((W - 24) / 3, 5, "VisionOps · Industrial AI")
    pdf.set_xy(12 + (W - 24) / 3, pdf.get_y())
    pdf.cell((W - 24) / 3, 5, f"Person Activity Report · {name}", align="C")
    pdf.set_xy(12 + 2 * (W - 24) / 3, pdf.get_y())
    pdf.cell((W - 24) / 3, 5, "Confidential", align="R")

    return bytes(pdf.output())
