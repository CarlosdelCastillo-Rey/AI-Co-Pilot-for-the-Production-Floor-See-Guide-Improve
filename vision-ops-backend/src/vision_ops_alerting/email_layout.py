from __future__ import annotations

import html
import re
from datetime import datetime

from vision_ops_alerting.schemas import IndustrialContext

# VisionOps Clarity email tokens (matches frontend + design HTML)
PRIMARY = "#0059bb"
INK = "#15181d"
MUTED = "#5d6571"
LABEL = "#9aa1ab"
BORDER = "#e6e9ee"
FOOTER_BG = "#f6f8fa"

SEVERITY_STYLES = {
    "warning": {
        "bar": "#b7791f",
        "badge_bg": "#fbf0db",
        "badge_color": "#8a5a10",
        "badge_label": "⚠ WARNING",
        "category_color": "#b7791f",
    },
    "critical": {
        "bar": "#c0362c",
        "badge_bg": "#fdecea",
        "badge_color": "#b22f25",
        "badge_label": "● CRITICAL",
        "category_color": "#c0362c",
    },
    "info": {
        "bar": PRIMARY,
        "badge_bg": "#eef3ff",
        "badge_color": PRIMARY,
        "badge_label": "INFO",
        "category_color": PRIMARY,
    },
}


def _format_idle(seconds: int | None) -> str:
    if not seconds:
        return "—"
    m, s = divmod(int(seconds), 60)
    if m:
        return f"{m}m {s}s"
    return f"{s}s"


def _format_timestamp(ts: str) -> tuple[str, str]:
    try:
        dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        display = dt.strftime("%d %b %Y · %H:%M:%S UTC")
        return display, ts
    except ValueError:
        return ts, ts


def build_template_vars(ctx: IndustrialContext, *, footer_reason: str, snapshot_url: str = "") -> dict[str, str]:
    actor = ctx.actor.name or ctx.actor.track_id or "unknown"
    display_ts, raw_ts = _format_timestamp(ctx.timestamp)
    idle = ctx.evidence.idle_seconds
    live = ctx.links.live_url or "http://localhost:3000/live"
    timeline = ctx.links.timeline_url or "http://localhost:3000/timeline"
    alerts = "http://localhost:3000/alerts"
    zone = ctx.evidence.roi or ctx.line_id

    return {
        "site_id": ctx.site_id,
        "line_id": ctx.line_id,
        "camera_id": ctx.camera_id,
        "timestamp": raw_ts,
        "timestamp_display": display_ts,
        "actor": actor,
        "live_url": live,
        "timeline_url": timeline,
        "alerts_url": alerts,
        "idle_seconds": str(idle or ""),
        "idle_duration": _format_idle(idle),
        "roi": ctx.evidence.roi or "",
        "zone": zone,
        "footer_reason": footer_reason,
        "snapshot_url": snapshot_url,
        "motion_score": "0.04",
        "idle_threshold": "3 min",
    }


def substitute(template: str, values: dict[str, str]) -> str:
    out = template
    for key, val in values.items():
        out = out.replace(f"{{{{{key}}}}}", val)
    return out


def _bold_narrative(body: str) -> str:
    """Wrap **bold** markers as HTML bold tags."""
    escaped = html.escape(body)

    def repl(match: re.Match[str]) -> str:
        return f'<b style="color:{INK};">{match.group(1)}</b>'

    return re.sub(r"\*\*(.+?)\*\*", repl, escaped)


def _meta_rows(values: dict[str, str], line_display: str | None = None) -> str:
    line_val = line_display or values["line_id"]
    rows = [
        ("Site", values["site_id"]),
        ("Line", line_val),
        ("Camera", values["camera_id"]),
        ("Actor", values["actor"]),
        ("Time", f'{values["timestamp_display"]}<span style="display:block;font-family:\'JetBrains Mono\',monospace;font-weight:500;font-size:11px;color:{LABEL};margin-top:2px;">{html.escape(values["timestamp"])}</span>'),
    ]
    parts = []
    for i, (label, val) in enumerate(rows):
        border = f"border-bottom:1px solid #eef0f3;" if i < len(rows) - 1 else ""
        parts.append(
            f'<tr><td style="padding:9px 0;{border}font-family:\'JetBrains Mono\',monospace;font-size:11px;font-weight:600;letter-spacing:.08em;text-transform:uppercase;color:{LABEL};width:118px;">{label}</td>'
            f'<td style="padding:9px 0;{border}font-weight:600;font-size:14px;color:{INK};">{val}</td></tr>'
        )
    return "".join(parts)


def _snapshot_block(values: dict[str, str], caption_extra: str = "live") -> str:
    url = values.get("snapshot_url") or ""
    if not url:
        return ""
    cap = f'{values["camera_id"]} · {values["line_id"]} · {values["timestamp_display"].split(" · ")[-1] if " · " in values["timestamp_display"] else values["timestamp_display"]}'
    src_attr = html.escape(url, quote=True)
    return f"""
  <tr><td style="padding:14px 26px 0;">
    <table role="presentation" width="100%" cellpadding="0" cellspacing="0" style="border:1px solid {BORDER};border-radius:10px;overflow:hidden;">
      <tr><td style="padding:0;font-size:0;line-height:0;"><img src="{src_attr}" width="548" alt="{html.escape(values["camera_id"])} snapshot" style="display:block;width:100%;height:auto;" /></td></tr>
      <tr><td style="background:#11161d;padding:9px 13px;font-family:'JetBrains Mono',monospace;font-size:11px;color:#aeb9c6;">{html.escape(cap)}&nbsp;&nbsp;<span style="color:#6fcf9b;">●</span> {caption_extra}</td></tr>
    </table>
  </td></tr>"""


def _har_logs_block(values: dict[str, str]) -> str:
    logs = values.get("har_logs") or ""
    if not logs.strip():
        return ""
    escaped = html.escape(logs)
    return f"""
  <tr><td style="padding:14px 26px 4px;">
    <div style="font-family:'JetBrains Mono',monospace;font-size:10px;font-weight:600;letter-spacing:.08em;text-transform:uppercase;color:{LABEL};margin-bottom:8px;">Session inference log</div>
    <pre style="margin:0;padding:14px 16px;background:#f6f8fa;border:1px solid {BORDER};border-radius:10px;font-family:'JetBrains Mono',monospace;font-size:11px;line-height:1.45;color:{INK};white-space:pre-wrap;word-break:break-word;max-height:320px;overflow:auto;">{escaped}</pre>
  </td></tr>"""


def _idle_stats_block(values: dict[str, str]) -> str:
    return f"""
  <tr><td style="padding:14px 26px 4px;">
    <table role="presentation" width="100%" cellpadding="0" cellspacing="0" style="background:#fbf9f4;border:1px solid #f0e6cf;border-radius:10px;">
      <tr>
        <td style="padding:14px 16px;text-align:center;border-right:1px solid #f0e6cf;"><div style="font-family:'Hanken Grotesk',sans-serif;font-weight:800;font-size:22px;color:#8a5a10;">{html.escape(values["idle_duration"])}</div><div style="font-family:'JetBrains Mono',monospace;font-size:10px;letter-spacing:.06em;text-transform:uppercase;color:#b09356;margin-top:3px;">idle duration</div></td>
        <td style="padding:14px 16px;text-align:center;border-right:1px solid #f0e6cf;"><div style="font-family:'Hanken Grotesk',sans-serif;font-weight:800;font-size:22px;color:#8a5a10;">{html.escape(values.get("motion_score", "—"))}</div><div style="font-family:'JetBrains Mono',monospace;font-size:10px;letter-spacing:.06em;text-transform:uppercase;color:#b09356;margin-top:3px;">motion score</div></td>
        <td style="padding:14px 16px;text-align:center;"><div style="font-family:'Hanken Grotesk',sans-serif;font-weight:800;font-size:22px;color:#8a5a10;">{html.escape(values.get("idle_threshold", "3 min"))}</div><div style="font-family:'JetBrains Mono',monospace;font-size:10px;letter-spacing:.06em;text-transform:uppercase;color:#b09356;margin-top:3px;">threshold</div></td>
      </tr>
    </table>
  </td></tr>"""


def build_email_html(
    *,
    slug: str,
    category: str,
    headline: str,
    body: str,
    severity_level: str,
    values: dict[str, str],
    layout: str = "standard",
    line_display: str | None = None,
) -> str:
    sev = SEVERITY_STYLES.get(severity_level, SEVERITY_STYLES["info"])
    narrative = _bold_narrative(substitute(body, values))
    headline_out = html.escape(substitute(headline, values))
    category_out = html.escape(category)
    footer_reason = html.escape(values.get("footer_reason", "Vision alerts"))
    site = html.escape(values["site_id"])
    live = html.escape(values["live_url"])
    timeline = html.escape(values["timeline_url"])
    alerts = html.escape(values["alerts_url"])

    extra = ""
    if layout == "idle_stats":
        snapshot_extra = _snapshot_block(values, caption_extra="idle detected") if values.get("snapshot_url") else ""
        extra = snapshot_extra + _idle_stats_block(values)
    elif layout == "har_logs":
        snapshot_extra = _snapshot_block(values, caption_extra="action detected") if values.get("snapshot_url") else ""
        extra = snapshot_extra + _har_logs_block(values)
    elif layout == "standard":
        extra = _snapshot_block(values, caption_extra="live" if severity_level != "critical" else "zone breach")

    return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"></head>
<body style="margin:0;padding:24px;background:#eef0f3;">
<table role="presentation" width="100%" cellpadding="0" cellspacing="0" style="background:#eef0f3;">
<tr><td align="center">
<table role="presentation" width="600" cellpadding="0" cellspacing="0" style="width:600px;max-width:100%;background:#ffffff;border-radius:14px;overflow:hidden;border:1px solid {BORDER};font-family:'Inter',Arial,sans-serif;">
  <tr><td style="height:4px;background:{sev["bar"]};font-size:0;line-height:0;">&nbsp;</td></tr>
  <tr><td style="padding:20px 26px 0;">
    <table role="presentation" width="100%" cellpadding="0" cellspacing="0"><tr>
      <td style="vertical-align:middle;">
        <table role="presentation" cellpadding="0" cellspacing="0"><tr>
          <td style="width:32px;height:32px;background:{PRIMARY};border-radius:8px;text-align:center;vertical-align:middle;font-family:'Hanken Grotesk',Arial,sans-serif;font-weight:800;font-size:13px;color:#ffffff;">VO</td>
          <td style="padding-left:10px;font-family:'Hanken Grotesk',Arial,sans-serif;font-weight:800;font-size:16px;color:{INK};">VisionOps<span style="display:block;font-family:'JetBrains Mono',monospace;font-weight:600;font-size:9px;letter-spacing:.16em;color:{LABEL};white-space:nowrap;">INDUSTRIAL AI</span></td>
        </tr></table>
      </td>
      <td align="right" style="font-family:'JetBrains Mono',monospace;font-size:11px;font-weight:600;color:{sev["badge_color"]};background:{sev["badge_bg"]};padding:5px 10px;border-radius:6px;white-space:nowrap;">{sev["badge_label"]}</td>
    </tr></table>
    <div style="font-family:'JetBrains Mono',monospace;font-size:10px;color:{LABEL};margin-top:10px;">{html.escape(slug)}</div>
  </td></tr>
  <tr><td style="padding:18px 26px 4px;">
    <div style="font-family:'JetBrains Mono',monospace;font-size:11px;font-weight:600;letter-spacing:.1em;text-transform:uppercase;color:{sev["category_color"]};">{category_out}</div>
    <div style="font-family:'Hanken Grotesk',Arial,sans-serif;font-weight:700;font-size:24px;line-height:1.2;letter-spacing:-.02em;color:{INK};margin-top:6px;">{headline_out}</div>
    <div style="font-size:14px;line-height:1.55;color:{MUTED};margin-top:8px;">{narrative}</div>
  </td></tr>
  {extra}
  <tr><td style="padding:18px 26px 4px;">
    <table role="presentation" width="100%" cellpadding="0" cellspacing="0">
      {_meta_rows(values, line_display)}
    </table>
  </td></tr>
  <tr><td style="padding:20px 26px 24px;">
    <table role="presentation" cellpadding="0" cellspacing="0"><tr>
      <td style="border-radius:8px;background:{PRIMARY};"><a href="{live}" style="display:inline-block;padding:12px 20px;font-family:'Inter',Arial,sans-serif;font-weight:600;font-size:14px;color:#ffffff;text-decoration:none;">View live feed →</a></td>
      <td style="width:10px;"></td>
      <td style="border-radius:8px;border:1px solid #d6dbe2;"><a href="{timeline}" style="display:inline-block;padding:11px 18px;font-family:'Inter',Arial,sans-serif;font-weight:600;font-size:14px;color:{INK};text-decoration:none;">Open timeline</a></td>
    </tr></table>
  </td></tr>
  <tr><td style="padding:16px 26px;background:{FOOTER_BG};border-top:1px solid #eef0f3;">
    <div style="font-size:12px;line-height:1.5;color:#8a929c;">You're receiving this because <b style="color:{MUTED};">{footer_reason}</b> are enabled for <b style="color:{MUTED};">{site}</b>. <a href="{alerts}" style="color:{PRIMARY};text-decoration:none;">Manage alert rules</a></div>
    <div style="font-size:11px;color:#aab2bc;margin-top:8px;">Alignity IQ Edge, LLC · Houston, TX · VisionOps autonomous vision platform</div>
  </td></tr>
</table>
</td></tr></table>
</body></html>"""


def build_email_text(
    *,
    category: str,
    headline: str,
    body: str,
    values: dict[str, str],
) -> str:
    plain_body = substitute(body, values).replace("**", "")
    snapshot_line = ""
    if values.get("snapshot_url"):
        if values["snapshot_url"].startswith("data:image"):
            snapshot_line = "Snapshot: attached as visionops-alert.jpg\n\n"
        else:
            snapshot_line = f"Snapshot: {values['snapshot_url']}\n\n"
    return (
        f"VisionOps — {substitute(headline, values)}\n"
        f"{category}\n\n"
        f"{plain_body}\n\n"
        f"{snapshot_line}"
        f"Site: {values['site_id']}\n"
        f"Line: {values['line_id']}\n"
        f"Camera: {values['camera_id']}\n"
        f"Actor: {values['actor']}\n"
        f"Time: {values['timestamp_display']}\n\n"
        f"Live: {values['live_url']}\n"
        f"Timeline: {values['timeline_url']}\n"
    )
