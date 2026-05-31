import type { ShiftAiSummaryApi, ShiftSummaryApi, TimelineEventApi } from "@/lib/api";

export type TimelineReportMeta = {
  siteName: string;
  shiftLabel: string;
  reportId: string;
  windowLabel: string;
  preparedBy?: string;
};

function escapeHtml(value: string): string {
  return value
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function severityClass(severity: TimelineEventApi["severity"]): "crit" | "warn" | "info" {
  if (severity === "critical") return "crit";
  if (severity === "warning") return "warn";
  return "info";
}

function severityLabel(severity: TimelineEventApi["severity"]): string {
  if (severity === "critical") return "Critical";
  if (severity === "warning") return "Warning";
  if (severity === "info") return "Info";
  return "Normal";
}

function resolutionPill(status: TimelineEventApi["resolutionStatus"]): { class: string; label: string } {
  if (status === "RESOLVED" || status === "FALSE_POSITIVE") {
    return { class: "resolved", label: status === "FALSE_POSITIVE" ? "False positive" : "Resolved" };
  }
  if (status === "ACKNOWLEDGED") return { class: "ackd", label: "Acknowledged" };
  return { class: "", label: "Open" };
}

function eventDetailRows(event: TimelineEventApi): { label: string; value: string }[] {
  const rows: { label: string; value: string }[] = [];
  if (event.industrialReasonCode) rows.push({ label: "Reason code", value: event.industrialReasonCode });
  if (event.downtimeCausedSeconds) {
    rows.push({ label: "Downtime", value: `${Math.round(event.downtimeCausedSeconds / 60)} min` });
  }
  if (event.scrapCausedUnits) rows.push({ label: "Scrap", value: `${event.scrapCausedUnits} units` });
  if (event.acknowledgedBy) rows.push({ label: "Ack by", value: event.acknowledgedBy });
  if (event.resolvedBy) rows.push({ label: "Closed by", value: event.resolvedBy });
  if (event.occurredAt) rows.push({ label: "Raised", value: event.occurredAt });
  if (event.closureNotes) rows.push({ label: "Notes", value: event.closureNotes });
  if (event.hiddenFromPanel) rows.push({ label: "Panel", value: "Removed from live timeline" });
  return rows.slice(0, 4);
}

function chipTags(event: TimelineEventApi): string[] {
  const tags = event.meta.map((m) => m.text);
  if (event.cameraId && !tags.some((t) => t.includes(event.cameraId!))) {
    tags.push(event.cameraId);
  }
  return tags.slice(0, 4);
}

function formatReportDate(summary: ShiftSummaryApi | null): string {
  if (!summary?.date) {
    return new Date().toLocaleDateString("en-GB", { day: "numeric", month: "long", year: "numeric" });
  }
  const parsed = new Date(summary.date);
  if (!Number.isNaN(parsed.getTime())) {
    return parsed.toLocaleDateString("en-GB", { day: "numeric", month: "long", year: "numeric" });
  }
  return summary.date;
}

function buildIncidentArticle(event: TimelineEventApi, index: number): string {
  const sev = severityClass(event.severity);
  const pill = resolutionPill(event.resolutionStatus);
  const details = eventDetailRows(event);
  const chips = chipTags(event);

  return `
    <article class="inc ${sev}">
      <div class="inc-head">
        <span class="ix">#${String(index + 1).padStart(2, "0")}</span>
        <span class="itime">${escapeHtml(event.time)}</span>
        <span class="badge ${sev}">${severityLabel(event.severity)}</span>
        <span class="pill ${pill.class}">${escapeHtml(pill.label)}</span>
        <span class="itag">${escapeHtml(event.id.slice(0, 12))}</span>
      </div>
      <h3 class="ititle">${escapeHtml(event.title)}</h3>
      <p class="idesc">${escapeHtml(event.description)}</p>
      ${
        chips.length
          ? `<div class="ichips">${chips.map((c) => `<span class="ichip">${escapeHtml(c)}</span>`).join("")}</div>`
          : ""
      }
      ${
        details.length
          ? `<div class="idetail">${details
              .map(
                (d) =>
                  `<div class="d"><div class="dl">${escapeHtml(d.label)}</div><div class="dv">${escapeHtml(d.value)}</div></div>`,
              )
              .join("")}</div>`
          : ""
      }
    </article>`;
}

function buildAiSummarySection(ai: ShiftAiSummaryApi | null): string {
  if (!ai) return "";

  const statusClass =
    ai.currentStatus === "all_clear" ? "status" : ai.currentStatus === "critical" ? "status crit-banner" : "status warn-banner";

  const listBlock = (title: string, items: string[]) =>
    items.length
      ? `<div class="ai-block"><div class="ai-bl">${escapeHtml(title)}</div><ul class="ai-list">${items
          .map((item) => `<li>${escapeHtml(item)}</li>`)
          .join("")}</ul></div>`
      : "";

  return `
    <section class="sec">
      <div class="sec-h"><span class="t">VisionOps AI Summary</span><span class="num">AUTO</span><span class="ln"></span></div>
      <div class="${statusClass}">
        <span class="d"></span>
        <span class="t">${escapeHtml(ai.statusHeadline)}</span>
        <span class="s">— ${escapeHtml(ai.statusDetail)}</span>
        <span class="meta">${escapeHtml(ai.metrics.uptime ?? "—")} · OEE ${ai.metrics.oee?.toFixed(1) ?? "—"}%</span>
      </div>
      <p class="ai-narrative">${escapeHtml(ai.narrative)}</p>
      <div class="ai-grid">
        ${listBlock("Highlights", ai.highlights)}
        ${listBlock("Suggestions", ai.suggestions)}
        ${listBlock("Recommendations", ai.recommendations)}
      </div>
    </section>`;
}

export function buildTimelineReportHtml(
  events: TimelineEventApi[],
  summary: ShiftSummaryApi | null,
  aiSummary: ShiftAiSummaryApi | null,
  meta: TimelineReportMeta,
): string {
  const total = summary?.totalEvents ?? events.length;
  const open = summary?.openCount ?? 0;
  const ack = summary?.acknowledgedCount ?? 0;
  const resolved = summary?.resolvedCount ?? 0;
  const falsePos = summary?.falsePositiveCount ?? 0;
  const uptime = summary?.uptime ?? "—";
  const allClear = summary?.allClear ?? (summary?.openCriticalCount ?? 0) === 0;

  const sev = aiSummary?.severityCounts
    ? { ...aiSummary.severityCounts }
    : { critical: 0, warning: 0, info: 0, normal: 0 };
  if (!aiSummary) {
    for (const e of events) {
      sev[e.severity] += 1;
    }
  }
  const sevTotal = Math.max(1, sev.critical + sev.warning + sev.info + sev.normal);
  const critPct = ((sev.critical / sevTotal) * 100).toFixed(1);
  const warnPct = ((sev.warning / sevTotal) * 100).toFixed(1);
  const infoPct = ((sev.info / sevTotal) * 100).toFixed(1);

  const statusBannerClass = allClear ? "status" : "status warn-banner";
  const statusTitle = allClear ? "All clear" : "Action needed";
  const statusSub = allClear
    ? "— no open critical incidents at shift close"
    : `— ${summary?.openCriticalCount ?? 0} critical open · ${open} awaiting triage`;
  const statusMeta = `${uptime} uptime · ${total} events handled`;

  const generated = new Date().toLocaleString("en-GB", {
    day: "numeric",
    month: "long",
    year: "numeric",
    hour: "2-digit",
    minute: "2-digit",
    timeZoneName: "short",
  });

  return `<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<title>VisionOps — Post-Shift Report</title>
<link rel="preconnect" href="https://fonts.googleapis.com" />
<link href="https://fonts.googleapis.com/css2?family=Hanken+Grotesk:wght@500;600;700;800&family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@500;600;700&display=swap" rel="stylesheet" />
<style>
  :root {
    --blue-700:#00489a; --blue-600:#0059bb; --blue-500:#0070ea; --blue-050:#eef3ff;
    --ok:#1f8a5b; --ok-bg:#e6f4ec; --warn:#b7791f; --warn-bg:#fbf0db; --crit:#c0362c; --crit-bg:#fbe7e4;
    --ink:#14181d; --dim:#5a626c; --faint:#949ba5; --line:#e7eaef; --divider:#eef0f3; --card:#ffffff;
    --f-d:'Hanken Grotesk',system-ui,sans-serif; --f-b:'Inter',system-ui,sans-serif; --f-m:'JetBrains Mono',ui-monospace,monospace;
  }
  * { box-sizing: border-box; }
  html, body { margin: 0; padding: 0; }
  body { font-family: var(--f-b); color: var(--ink); background: #fff; -webkit-font-smoothing: antialiased;
    -webkit-print-color-adjust: exact; print-color-adjust: exact; }
  .doc { background: #fff; width: 816px; }
  .pad { padding: 46px 54px 70px; }
  @page { size: Letter; margin: 14mm 14mm 18mm; }
  @media print { .pad { padding: 0; } .foot { position: fixed; } }

  .mast { background: linear-gradient(125deg, var(--blue-700) 0%, var(--blue-600) 58%, var(--blue-500) 130%); color: #fff;
    padding: 30px 54px 26px; position: relative; overflow: hidden; }
  .mast::after { content:''; position:absolute; right:-60px; top:-60px; width:260px; height:260px; border-radius:50%;
    background: radial-gradient(circle, rgba(255,255,255,.14), transparent 62%); }
  .mast-top { display: flex; align-items: center; gap: 13px; }
  .mk { width: 40px; height: 40px; border-radius: 11px; background: rgba(255,255,255,.16); border: 1px solid rgba(255,255,255,.28);
    display: grid; place-items: center; flex: 0 0 40px; }
  .mast-top .nm { font-family: var(--f-d); font-weight: 800; font-size: 18px; letter-spacing: -.01em; line-height: 1; }
  .mast-top .sub { font-family: var(--f-m); font-size: 9.5px; letter-spacing: .22em; text-transform: uppercase; color: rgba(255,255,255,.72); margin-top: 5px; }
  .mast-top .rk { margin-left: auto; text-align: right; font-family: var(--f-m); font-size: 10.5px; letter-spacing: .04em; color: rgba(255,255,255,.82); }
  .mast h1 { font-family: var(--f-d); font-weight: 800; font-size: 33px; letter-spacing: -.025em; margin: 20px 0 0; line-height: 1.04; }
  .mast .lede { font-size: 13.5px; color: rgba(255,255,255,.82); margin-top: 7px; }
  .metastrip { display: flex; flex-wrap: wrap; gap: 0; margin-top: 20px; border-top: 1px solid rgba(255,255,255,.2); padding-top: 14px; }
  .metastrip .m { padding-right: 30px; margin-right: 30px; border-right: 1px solid rgba(255,255,255,.2); }
  .metastrip .m:last-child { border-right: 0; }
  .metastrip .ml { font-family: var(--f-m); font-size: 9px; letter-spacing: .16em; text-transform: uppercase; color: rgba(255,255,255,.62); }
  .metastrip .mv { font-weight: 600; font-size: 13px; margin-top: 4px; }

  .sec { margin-top: 30px; }
  .sec-h { display: flex; align-items: center; gap: 11px; margin-bottom: 15px; }
  .sec-h .t { font-family: var(--f-d); font-weight: 700; font-size: 15px; letter-spacing: .01em; color: var(--ink); }
  .sec-h .num { font-family: var(--f-m); font-size: 10px; font-weight: 600; color: var(--blue-600); background: var(--blue-050);
    padding: 2px 8px; border-radius: 5px; }
  .sec-h .ln { flex: 1; height: 1px; background: var(--line); }

  .status { display: flex; align-items: center; gap: 12px; padding: 13px 18px; border-radius: 11px; border: 1px solid #cde6d8;
    background: var(--ok-bg); margin-bottom: 18px; }
  .status .d { width: 9px; height: 9px; border-radius: 50%; background: var(--ok); box-shadow: 0 0 0 4px rgba(31,138,91,.18); flex: 0 0 9px; }
  .status .t { font-family: var(--f-d); font-weight: 700; font-size: 14px; color: var(--ok); }
  .status .s { font-size: 12.5px; color: #2f7a58; margin-left: 2px; }
  .status .meta { margin-left: auto; font-family: var(--f-m); font-size: 10.5px; letter-spacing: .04em; color: #4f8a6c; text-transform: uppercase; }
  .status.warn-banner { border-color: #ecdcb5; background: var(--warn-bg); }
  .status.warn-banner .d { background: var(--warn); box-shadow: 0 0 0 4px rgba(183,121,31,.18); }
  .status.warn-banner .t { color: var(--warn); }
  .status.warn-banner .s { color: #8a6524; }
  .status.warn-banner .meta { color: #8a6524; }
  .status.crit-banner { border-color: #e8c4bf; background: var(--crit-bg); }
  .status.crit-banner .d { background: var(--crit); box-shadow: 0 0 0 4px rgba(192,54,44,.18); }
  .status.crit-banner .t { color: var(--crit); }
  .status.crit-banner .s { color: #9a3d34; }
  .status.crit-banner .meta { color: #9a3d34; }

  .kpis { display: grid; grid-template-columns: repeat(5, minmax(0, 1fr)); gap: 12px; }
  .kpi { border: 1px solid var(--line); border-radius: 11px; padding: 13px 14px; background: var(--card); }
  .kpi .l { font-family: var(--f-m); font-size: 8.5px; letter-spacing: .1em; text-transform: uppercase; color: var(--faint); }
  .kpi .v { font-family: var(--f-d); font-weight: 800; font-size: 28px; letter-spacing: -.03em; margin-top: 7px; line-height: 1; color: var(--ink); }
  .kpi .v small { font-size: 14px; font-weight: 700; color: var(--dim); }
  .kpi.good .v { color: var(--ok); }
  .kpi .foot { font-family: var(--f-m); font-size: 9.5px; color: var(--faint); margin-top: 6px; }

  .brk { display: grid; grid-template-columns: minmax(0, 1fr) 220px; gap: 26px; align-items: center; margin-top: 16px;
    border: 1px solid var(--line); border-radius: 11px; padding: 16px 18px; }
  .brk .bl { font-family: var(--f-m); font-size: 9px; letter-spacing: .12em; text-transform: uppercase; color: var(--faint); margin-bottom: 9px; }
  .bar { display: flex; height: 13px; border-radius: 7px; overflow: hidden; }
  .bar > span { display: block; }
  .bar .c { background: var(--crit); } .bar .w { background: var(--warn); } .bar .i { background: var(--blue-500); }
  .leg { display: flex; flex-direction: column; gap: 8px; }
  .leg .r { display: flex; align-items: center; gap: 9px; font-size: 12px; color: var(--dim); }
  .leg .sw { width: 9px; height: 9px; border-radius: 3px; flex: 0 0 9px; }
  .leg .r b { margin-left: auto; font-family: var(--f-m); font-size: 12px; color: var(--ink); }

  .ai-narrative { font-size: 13px; line-height: 1.55; color: var(--dim); margin: 0 0 14px; }
  .ai-grid { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 12px; }
  .ai-block { border: 1px solid var(--line); border-radius: 11px; padding: 14px 16px; background: var(--card); }
  .ai-bl { font-family: var(--f-m); font-size: 9px; letter-spacing: .12em; text-transform: uppercase; color: var(--faint); margin-bottom: 8px; }
  .ai-list { margin: 0; padding-left: 16px; font-size: 12px; line-height: 1.45; color: var(--dim); }
  .ai-list li { margin-bottom: 6px; }

  .inc { position: relative; border: 1px solid var(--line); border-radius: 12px; padding: 15px 18px 15px 20px; margin-bottom: 12px;
    background: var(--card); break-inside: avoid; overflow: hidden; }
  .inc::before { content:''; position: absolute; left: 0; top: 0; bottom: 0; width: 4px; }
  .inc.crit::before { background: var(--crit); } .inc.warn::before { background: var(--warn); } .inc.info::before { background: var(--blue-500); }
  .inc-head { display: flex; align-items: center; gap: 9px; flex-wrap: wrap; }
  .ix { font-family: var(--f-m); font-size: 11px; font-weight: 700; color: var(--faint); }
  .itime { font-family: var(--f-m); font-size: 12px; font-weight: 600; color: var(--dim); }
  .badge { font-family: var(--f-m); font-size: 9px; font-weight: 700; text-transform: uppercase; letter-spacing: .07em; padding: 3px 8px; border-radius: 5px; }
  .badge.crit { background: var(--crit-bg); color: var(--crit); } .badge.warn { background: var(--warn-bg); color: var(--warn); } .badge.info { background: var(--blue-050); color: var(--blue-600); }
  .pill { font-family: var(--f-m); font-size: 9px; font-weight: 700; text-transform: uppercase; letter-spacing: .06em; padding: 3px 8px; border-radius: 20px; border: 1px solid var(--line); color: var(--dim); }
  .pill.resolved { color: var(--ok); border-color: #bfe3cf; background: var(--ok-bg); }
  .pill.ackd { color: var(--warn); border-color: #ecdcb5; background: var(--warn-bg); }
  .itag { margin-left: auto; font-family: var(--f-m); font-size: 10px; color: var(--faint); }
  .ititle { font-family: var(--f-d); font-weight: 700; font-size: 16px; letter-spacing: -.01em; margin: 9px 0 5px; color: var(--ink); }
  .idesc { font-size: 12.5px; line-height: 1.5; color: var(--dim); margin: 0; max-width: 92%; }
  .ichips { display: flex; flex-wrap: wrap; gap: 7px; margin-top: 11px; }
  .ichip { font-family: var(--f-m); font-size: 10.5px; color: var(--dim); background: #f2f4f7; padding: 3px 9px; border-radius: 6px; }
  .idetail { display: flex; flex-wrap: wrap; gap: 0; margin-top: 12px; padding-top: 11px; border-top: 1px dashed var(--line); }
  .idetail .d { padding-right: 22px; margin-right: 22px; border-right: 1px solid var(--divider); }
  .idetail .d:last-child { border-right: 0; }
  .idetail .dl { font-family: var(--f-m); font-size: 8.5px; letter-spacing: .1em; text-transform: uppercase; color: var(--faint); }
  .idetail .dv { font-size: 12px; font-weight: 600; color: var(--ink); margin-top: 3px; }

  .foot { display: flex; align-items: center; justify-content: space-between;
    padding: 10px 54px; border-top: 1px solid var(--line); background: #fff; font-family: var(--f-m); font-size: 9.5px;
    letter-spacing: .04em; color: var(--faint); }
  .foot .c { color: var(--dim); }
  .signoff { display: flex; align-items: center; gap: 14px; margin-top: 26px; padding-top: 16px; border-top: 1px solid var(--line); }
  .signoff .blk { flex: 1; }
  .signoff .l { font-family: var(--f-m); font-size: 9px; letter-spacing: .12em; text-transform: uppercase; color: var(--faint); }
  .signoff .v { font-weight: 600; font-size: 13px; margin-top: 5px; color: var(--ink); }
  .signoff .sig { font-family: var(--f-d); font-style: italic; font-weight: 600; font-size: 18px; color: var(--blue-700); border-bottom: 1px solid var(--line); padding-bottom: 2px; }
</style>
</head>
<body>
<div class="doc">
  <header class="mast">
    <div class="mast-top">
      <div class="mk">
        <svg viewBox="0 0 24 24" width="22" height="22" fill="none" stroke="#fff" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round">
          <path d="M5 9V6.5A1.5 1.5 0 016.5 5H9"/><path d="M15 5h2.5A1.5 1.5 0 0119 6.5V9"/><path d="M19 15v2.5a1.5 1.5 0 01-1.5 1.5H15"/><path d="M9 19H6.5A1.5 1.5 0 015 17.5V15"/><circle cx="12" cy="12" r="2.5"/>
        </svg>
      </div>
      <div>
        <div class="nm">VisionOps</div>
        <div class="sub">Industrial AI · Edge</div>
      </div>
      <div class="rk">REPORT · ${escapeHtml(meta.reportId)}<br/>Confidential</div>
    </div>
    <h1>Post-Shift Log</h1>
    <div class="lede">Autonomous vision incident summary · See · Guide · Improve</div>
    <div class="metastrip">
      <div class="m"><div class="ml">Site</div><div class="mv">${escapeHtml(meta.siteName)}</div></div>
      <div class="m"><div class="ml">Shift</div><div class="mv">${escapeHtml(meta.shiftLabel)}</div></div>
      <div class="m"><div class="ml">Date</div><div class="mv">${escapeHtml(formatReportDate(summary))}</div></div>
      <div class="m"><div class="ml">Window</div><div class="mv">${escapeHtml(meta.windowLabel)}</div></div>
      <div class="m"><div class="ml">Prepared by</div><div class="mv">${escapeHtml(meta.preparedBy ?? "—")}</div></div>
    </div>
  </header>

  <div class="pad">
    <section class="sec" style="margin-top:34px;">
      <div class="sec-h"><span class="t">Shift Summary</span><span class="ln"></span></div>
      <div class="${statusBannerClass}">
        <span class="d"></span>
        <span class="t">${statusTitle}</span>
        <span class="s">${statusSub}</span>
        <span class="meta">${statusMeta}</span>
      </div>
      <div class="kpis">
        <div class="kpi"><div class="l">Total Events</div><div class="v">${total}</div><div class="foot">across ${summary?.assets?.length ?? 1} line(s)</div></div>
        <div class="kpi"><div class="l">Open</div><div class="v">${open}</div><div class="foot">${open ? "pending triage" : "none pending"}</div></div>
        <div class="kpi"><div class="l">Acknowledged</div><div class="v">${ack}</div><div class="foot">triaged on-shift</div></div>
        <div class="kpi good"><div class="l">Resolved</div><div class="v">${resolved}</div><div class="foot">${falsePos ? `${falsePos} false pos.` : "closed out"}</div></div>
        <div class="kpi good"><div class="l">Uptime</div><div class="v">${uptime.endsWith("%") ? `${escapeHtml(uptime.replace("%", ""))}<small>%</small>` : escapeHtml(uptime)}</div><div class="foot">${falsePos} false positives</div></div>
      </div>
      <div class="brk">
        <div>
          <div class="bl">Incidents by severity</div>
          <div class="bar">
            <span class="c" style="width:${critPct}%"></span>
            <span class="w" style="width:${warnPct}%"></span>
            <span class="i" style="width:${infoPct}%"></span>
          </div>
        </div>
        <div class="leg">
          <div class="r"><span class="sw" style="background:var(--crit)"></span>Critical<b>${sev.critical}</b></div>
          <div class="r"><span class="sw" style="background:var(--warn)"></span>Warning<b>${sev.warning}</b></div>
          <div class="r"><span class="sw" style="background:var(--blue-500)"></span>Info<b>${sev.info}</b></div>
        </div>
      </div>
    </section>

    ${buildAiSummarySection(aiSummary)}

    <section class="sec">
      <div class="sec-h"><span class="t">Incident Log</span><span class="num">${events.length} ENTRIES</span><span class="ln"></span></div>
      ${events.map((event, i) => buildIncidentArticle(event, i)).join("")}
    </section>

    <section class="signoff">
      <div class="blk"><div class="l">Reviewed &amp; approved</div><div class="sig">${escapeHtml((meta.preparedBy ?? "").split(" · ")[0] || "—")}</div></div>
      <div class="blk"><div class="l">Role</div><div class="v">Shift Supervisor</div></div>
      <div class="blk"><div class="l">Generated</div><div class="v">${escapeHtml(generated)}</div></div>
    </section>
  </div>

  <footer class="foot">
    <span>VisionOps · Industrial AI</span>
    <span class="c">Post-Shift Log · ${escapeHtml(meta.siteName)} · ${escapeHtml(meta.shiftLabel)}</span>
    <span>Confidential — Alignity IQ Edge</span>
  </footer>
</div>
</body>
</html>`;
}

export function buildReportMeta(
  events: TimelineEventApi[],
  aiSummary: ShiftAiSummaryApi | null,
  shiftLabel: string,
  preparedBy?: string,
): TimelineReportMeta {
  const now = new Date();
  const isoDate = now.toISOString().slice(0, 10).replace(/-/g, "");
  const letter = shiftLabel.replace(/^SHIFT\s+/i, "").split("-")[0] ?? "A";
  const reportId = `PSL-${now.getFullYear()}-${isoDate.slice(4)}-${letter}`;

  let windowLabel = "Full shift window";
  if (events.length > 0) {
    const times = events.map((e) => e.time).filter(Boolean);
    if (times.length) windowLabel = `${times[times.length - 1]} – ${times[0]} UTC`;
  }

  return {
    siteName: aiSummary?.siteName ?? "site-01",
    shiftLabel,
    reportId,
    windowLabel,
    preparedBy: preparedBy ?? shiftLabel.replace(/^SHIFT\s+/i, "Shift "),
  };
}
