"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import { AppShell } from "@/components/layout/AppShell";
import { DATA_RESET_EVENT } from "@/components/layout/Sidebar";
import { useAuth } from "@/components/auth/AuthProvider";
import { DataEmptyState } from "@/components/shared/DataEmptyState";
import { KpiLabel } from "@/components/shared/KpiHelp";
import { Icon } from "@/components/ui/Icon";
import { EventResolveModal } from "@/components/timeline/EventResolveModal";
import { EventWorkflowActions } from "@/components/timeline/EventWorkflowActions";
import { ResolutionStatusBadge } from "@/components/timeline/TimelineBadges";
import { ShiftAiSummaryPanel } from "@/components/timeline/ShiftAiSummaryPanel";
import { TimelineEventThumbnail } from "@/components/timeline/TimelineEventThumbnail";
import {
  acknowledgeTimelineEvent,
  dismissTimelineEvent,
  fetchReasonCodes,
  fetchShiftAiSummary,
  fetchShiftSummary,
  fetchTimelineEvents,
  resolveTimelineEvent,
  type ReasonCodeApi,
  type ResolutionStatus,
  type ShiftAiSummaryApi,
  type ShiftSummaryApi,
  type TimelineEventApi,
} from "@/lib/api";
import { exportTimelinePdf } from "@/lib/timeline-pdf";
import { isTimelineThumbnail } from "@/lib/images";
import { currentShiftLabel } from "@/lib/shift";
import type { Severity } from "@/lib/types";
import { cn } from "@/lib/cn";

const severityPill: Record<Severity, string> = {
  critical: "bg-[#FBE7E4] text-[#C0362C]",
  warning: "bg-[#FBF0DB] text-[#B7791F]",
  info: "bg-[#EEF3FF] text-[#0059BB]",
  normal: "bg-[#F1F3F6] text-[#687079]",
};

const severityAccent: Record<Severity, string> = {
  critical: "border-l-[#C0362C]",
  warning: "border-l-[#B7791F]",
  info: "border-l-[#0059BB]",
  normal: "border-l-[#9AA1AB]",
};

const severityDot: Record<Severity, string> = {
  critical: "bg-[#C0362C]",
  warning: "bg-[#B7791F]",
  info: "bg-[#0059BB]",
  normal: "bg-[#9AA1AB]",
};

type SeverityFilter = "" | Severity;
type StatusFilter = "" | ResolutionStatus;

function shiftCompletePct(summary: ShiftSummaryApi | null): number {
  if (!summary?.totalEvents) return 100;
  const done = (summary.resolvedCount ?? 0) + (summary.falsePositiveCount ?? 0);
  return Math.round((done / summary.totalEvents) * 100);
}

export function TimelinePageClient() {
  const { user } = useAuth();
  const [events, setEvents] = useState<TimelineEventApi[]>([]);
  const [summary, setSummary] = useState<ShiftSummaryApi | null>(null);
  const [aiSummary, setAiSummary] = useState<ShiftAiSummaryApi | null>(null);
  const [reasonCodes, setReasonCodes] = useState<ReasonCodeApi[]>([]);
  const [loading, setLoading] = useState(true);
  const [busyId, setBusyId] = useState<string | null>(null);
  const [search, setSearch] = useState("");
  const [severityFilter, setSeverityFilter] = useState<SeverityFilter>("");
  const [statusFilter, setStatusFilter] = useState<StatusFilter>("");
  const [harOnly, setHarOnly] = useState(false);
  const [filterOpen, setFilterOpen] = useState(false);
  const [resolveEvent, setResolveEvent] = useState<TimelineEventApi | null>(null);
  const [exporting, setExporting] = useState(false);

  const load = useCallback(async () => {
    const [eventData, summaryData, aiData, codes] = await Promise.all([
      fetchTimelineEvents({
        limit: 50,
        severity: severityFilter || undefined,
        resolutionStatus: statusFilter || undefined,
        harOnly: harOnly || undefined,
      }),
      fetchShiftSummary(),
      fetchShiftAiSummary(),
      fetchReasonCodes(),
    ]);
    setEvents(eventData);
    setSummary(summaryData);
    setAiSummary(aiData);
    setReasonCodes(codes);
    setLoading(false);
  }, [severityFilter, statusFilter, harOnly]);

  useEffect(() => {
    void load();
    const id = setInterval(() => void load(), 30_000);
    const onReset = () => void load();
    window.addEventListener(DATA_RESET_EVENT, onReset);
    return () => {
      clearInterval(id);
      window.removeEventListener(DATA_RESET_EVENT, onReset);
    };
  }, [load]);

  const filteredEvents = useMemo(() => {
    const q = search.trim().toLowerCase();
    if (!q) return events;
    return events.filter(
      (e) =>
        e.title.toLowerCase().includes(q) ||
        e.description.toLowerCase().includes(q) ||
        e.meta.some((m) => m.text.toLowerCase().includes(q)) ||
        (e.cameraId?.toLowerCase().includes(q) ?? false) ||
        (e.resolutionStatus?.toLowerCase().includes(q) ?? false),
    );
  }, [events, search]);

  const severityCounts = useMemo(() => {
    const counts = { critical: 0, warning: 0, info: 0, normal: 0 };
    for (const e of events) {
      counts[e.severity] += 1;
    }
    return counts;
  }, [events]);

  const handleAcknowledge = async (eventId: string) => {
    setBusyId(eventId);
    await acknowledgeTimelineEvent(eventId);
    await load();
    setBusyId(null);
  };

  const handleResolveSubmit = async (payload: {
    status: "RESOLVED" | "FALSE_POSITIVE";
    reasonCode?: string;
    downtimeMinutes: number;
    scrapUnits: number;
    notes: string;
  }) => {
    if (!resolveEvent) return;
    setBusyId(resolveEvent.id);
    await resolveTimelineEvent(resolveEvent.id, {
      status: payload.status,
      reasonCode: payload.reasonCode,
      downtimeSeconds: Math.round(payload.downtimeMinutes * 60),
      scrapUnits: payload.scrapUnits,
      notes: payload.notes || undefined,
    });
    await load();
    setBusyId(null);
  };

  const handleDismiss = async (eventId: string) => {
    const event = events.find((e) => e.id === eventId);
    if (!event) return;
    if (
      !window.confirm(
        `Remove “${event.title}” from the timeline panel?\n\nIt will remain in shift reports, PDF export, and AI summaries.`,
      )
    ) {
      return;
    }
    setBusyId(eventId);
    const ok = await dismissTimelineEvent(eventId);
    if (ok) {
      setEvents((prev) => prev.filter((e) => e.id !== eventId));
      await load();
    }
    setBusyId(null);
  };

  const handleExport = async () => {
    if (exporting) return;
    setExporting(true);
    try {
      const reportEvents = await fetchTimelineEvents({
        limit: 200,
        severity: severityFilter || undefined,
        resolutionStatus: statusFilter || undefined,
        panelOnly: false,
      });
      const q = search.trim().toLowerCase();
      const forReport = q
        ? reportEvents.filter(
            (e) =>
              e.title.toLowerCase().includes(q) ||
              e.description.toLowerCase().includes(q) ||
              e.meta.some((m) => m.text.toLowerCase().includes(q)) ||
              (e.cameraId?.toLowerCase().includes(q) ?? false) ||
              (e.resolutionStatus?.toLowerCase().includes(q) ?? false),
          )
        : reportEvents;
      if (forReport.length === 0) return;

      await exportTimelinePdf(
        forReport,
        summary,
        aiSummary,
        aiSummary?.shiftLabel ?? currentShiftLabel(),
        user ? `${user.name} · ${user.role}` : undefined,
      );
    } finally {
      setExporting(false);
    }
  };

  const allClear = summary?.allClear ?? (summary?.openCriticalCount ?? 0) === 0;
  const completePct = shiftCompletePct(summary);
  const uptimeTrendUp = (summary?.uptimePct ?? 0) >= 90;
  const filtersActive = Boolean(severityFilter || statusFilter || harOnly || search.trim());
  const totalSeverity = Math.max(
    1,
    severityCounts.critical + severityCounts.warning + severityCounts.info + severityCounts.normal,
  );

  return (
    <AppShell
      searchPlaceholder="Search events, operators, incidents…"
      searchValue={search}
      onSearchChange={setSearch}
    >
      <div className="flex min-h-[calc(100vh-4rem)] bg-[#F5F6F8]">
        {/* Main timeline column */}
        <div className="min-w-0 flex-1">
          {/* Page header */}
          <header className="border-b border-[#EBEDF1] bg-white px-8 pb-5 pt-8">
            <div className="mx-auto flex max-w-[920px] flex-wrap items-end justify-between gap-4">
              <div>
                <div className="mb-3 inline-flex items-center gap-2 rounded-full border border-[#CFE0FF] bg-[#EEF3FF] px-3 py-1 font-label text-[11px] font-bold uppercase tracking-wider text-[#0059BB]">
                  {currentShiftLabel()}
                </div>
                <h1 className="font-headline text-[28px] font-semibold leading-tight text-[#0C0F13]">
                  Post-Shift Log
                </h1>
                <p className="mt-1 text-[14px] text-[#687079]">
                  Operational workflow — triage, tag, and close incidents
                </p>
              </div>
              <div className="relative flex flex-wrap items-center gap-2">
                <button
                  type="button"
                  className="inline-flex min-h-[40px] items-center gap-2 rounded-[10px] border border-[#EBEDF1] bg-[#F5F6F8] px-3 font-label text-[13px] font-medium text-[#2B3340]"
                >
                  <Icon name="schedule" size={16} />
                  Group: Time
                </button>
                <button
                  type="button"
                  onClick={() => setFilterOpen((o) => !o)}
                  className={cn(
                    "inline-flex min-h-[40px] items-center gap-2 rounded-[10px] border px-3 font-label text-[13px] font-medium transition-colors",
                    severityFilter || statusFilter || harOnly
                      ? "border-[#0059BB] bg-[#EEF3FF] text-[#0059BB]"
                      : "border-[#EBEDF1] bg-white text-[#2B3340] hover:bg-[#F5F6F8]",
                  )}
                >
                  <Icon name="filter_list" size={16} />
                  Filter
                </button>
                {filterOpen && (
                  <>
                    <button
                      type="button"
                      className="fixed inset-0 z-[40]"
                      aria-label="Close filters"
                      onClick={() => setFilterOpen(false)}
                    />
                    <div className="absolute right-0 top-full z-[50] mt-2 w-56 rounded-[14px] border border-[#EBEDF1] bg-white p-3 shadow-[0_8px_24px_rgba(12,15,19,0.08)]">
                      <p className="mb-2 font-label text-[11px] font-bold uppercase tracking-wide text-[#9AA1AB]">
                        Severity
                      </p>
                      {(["", "critical", "warning", "info", "normal"] as const).map((s) => (
                        <button
                          key={s || "all-sev"}
                          type="button"
                          onClick={() => setSeverityFilter(s)}
                          className={cn(
                            "mb-1 block w-full rounded-lg px-3 py-2 text-left text-[13px] capitalize hover:bg-[#F5F6F8]",
                            severityFilter === s && "bg-[#EEF3FF] font-semibold text-[#0059BB]",
                          )}
                        >
                          {s || "All severities"}
                        </button>
                      ))}
                      <p className="mb-2 mt-3 font-label text-[11px] font-bold uppercase tracking-wide text-[#9AA1AB]">
                        Source
                      </p>
                      <button
                        type="button"
                        onClick={() => {
                          setHarOnly((h) => !h);
                          setFilterOpen(false);
                        }}
                        className={cn(
                          "mb-1 block w-full rounded-lg px-3 py-2 text-left text-[13px] hover:bg-[#F5F6F8]",
                          harOnly && "bg-[#EEF3FF] font-semibold text-[#0059BB]",
                        )}
                      >
                        HAR activity alerts only
                      </button>
                      <p className="mb-2 mt-3 font-label text-[11px] font-bold uppercase tracking-wide text-[#9AA1AB]">
                        Workflow Status
                      </p>
                      {(["", "OPEN", "ACKNOWLEDGED", "RESOLVED", "FALSE_POSITIVE"] as const).map((s) => (
                        <button
                          key={s || "all-status"}
                          type="button"
                          onClick={() => {
                            setStatusFilter(s);
                            setFilterOpen(false);
                          }}
                          className={cn(
                            "mb-1 block w-full rounded-lg px-3 py-2 text-left text-[13px] hover:bg-[#F5F6F8]",
                            statusFilter === s && "bg-[#EEF3FF] font-semibold text-[#0059BB]",
                          )}
                        >
                          {s ? s.replace("_", " ") : "All statuses"}
                        </button>
                      ))}
                    </div>
                  </>
                )}
                <button
                  type="button"
                  onClick={() => void handleExport()}
                  disabled={exporting || loading}
                  className="inline-flex min-h-[40px] items-center gap-2 rounded-[10px] bg-[#0059BB] px-4 font-label text-[13px] font-semibold text-white transition-colors hover:bg-[#0070EA] disabled:opacity-40"
                >
                  <Icon name="download" size={16} />
                  {exporting ? "Exporting…" : "Export PDF"}
                </button>
              </div>
            </div>
          </header>

          {/* Status strip — Clarity style */}
          <div
            className={cn(
              "border-b border-[#EBEDF1] px-8 py-3",
              allClear ? "bg-[#E6F4EC]" : "bg-[#FBE7E4]",
            )}
          >
            <div className="mx-auto flex max-w-[920px] flex-wrap items-center justify-between gap-3">
              <div className="flex items-center gap-2">
                <span
                  className={cn(
                    "inline-flex items-center rounded-full px-2.5 py-0.5 font-label text-[11px] font-bold uppercase tracking-wide",
                    allClear ? "bg-[#1F8A5B] text-white" : "bg-[#C0362C] text-white",
                  )}
                >
                  {allClear ? "All clear" : "Action needed"}
                </span>
                <p className="text-[13px] font-medium text-[#2B3340]">
                  {allClear
                    ? `No open critical incidents · shift ${completePct}% complete`
                    : `${summary?.openCriticalCount ?? 0} critical open · ${summary?.openCount ?? 0} awaiting triage`}
                </p>
              </div>
              {summary?.totalEvents != null && (
                <span className="font-label text-[12px] text-[#687079]">
                  {summary.totalEvents} events today
                </span>
              )}
            </div>
          </div>

          {/* Timeline feed */}
          <div className="relative mx-auto max-w-[920px] px-8 py-8">
            <div className="absolute bottom-8 left-[39px] top-8 w-px bg-[#DDE1E6]" />
            <div className="relative space-y-6">
              {loading ? (
                <p className="pl-10 text-[14px] text-[#687079]">Loading events…</p>
              ) : filteredEvents.length === 0 ? (
                filtersActive ? (
                  <p className="pl-10 text-[14px] text-[#687079]">No events match your filters.</p>
                ) : (
                  <DataEmptyState
                    className="ml-10"
                    icon="event_available"
                    title="No incidents logged yet"
                    description="The shift timeline is empty. Use Fresh start in the sidebar anytime to clear runtime alerts and begin again."
                  />
                )
              ) : (
                filteredEvents.map((event) => (
                  <article key={event.id} className="relative pl-10">
                    <div
                      className={cn(
                        "absolute left-[3px] top-5 h-2.5 w-2.5 rounded-full ring-4 ring-[#F5F6F8]",
                        severityDot[event.severity],
                      )}
                    />
                    <div
                      className={cn(
                        "overflow-hidden rounded-[14px] border border-[#EBEDF1] bg-white shadow-[0_1px_3px_rgba(12,15,19,0.04)] transition-shadow hover:shadow-[0_4px_12px_rgba(12,15,19,0.06)]",
                        "border-l-4",
                        severityAccent[event.severity],
                        event.resolutionStatus === "OPEN" &&
                          event.severity === "critical" &&
                          "ring-1 ring-[#F0C4BF]",
                      )}
                    >
                      <div className="flex flex-col md:flex-row md:items-stretch">
                        <div className="min-w-0 flex-1 p-5">
                          <div className="mb-3 flex flex-wrap items-center gap-2">
                            <span className="font-label text-[13px] font-medium text-[#687079]">
                              {event.time}
                            </span>
                            <span
                              className={cn(
                                "rounded-full px-2 py-0.5 font-label text-[10px] font-bold uppercase tracking-wide",
                                severityPill[event.severity],
                              )}
                            >
                              {event.severity === "info" ? "Info" : event.severity}
                            </span>
                            <ResolutionStatusBadge status={event.resolutionStatus} />
                            {event.harSource && (
                              <span className="rounded-full bg-[#E8F5E9] px-2 py-0.5 font-label text-[10px] font-bold uppercase tracking-wide text-[#2E7D32]">
                                HAR
                              </span>
                            )}
                          </div>
                          <h3 className="mb-1.5 font-headline text-[17px] font-semibold leading-snug text-[#0C0F13]">
                            {event.title}
                          </h3>
                          <p className="text-[14px] leading-relaxed text-[#5A626C]">
                            {event.description}
                          </p>
                          <div className="mt-3 flex flex-wrap gap-2">
                            {event.meta.map((m) => (
                              <span
                                key={m.text}
                                className="inline-flex items-center gap-1 rounded-full border border-[#EBEDF1] bg-[#F5F6F8] px-2.5 py-1 font-label text-[11px] text-[#687079]"
                              >
                                <Icon name={m.icon} size={13} />
                                {m.text}
                              </span>
                            ))}
                          </div>
                          <EventWorkflowActions
                            event={event}
                            busy={busyId === event.id}
                            onAcknowledge={handleAcknowledge}
                            onResolve={setResolveEvent}
                            onDismiss={handleDismiss}
                          />
                        </div>
                        {isTimelineThumbnail(event.thumbnail) ? (
                          <TimelineEventThumbnail
                            src={event.thumbnail}
                            title={event.title}
                            clipDuration={event.clipDuration}
                          />
                        ) : null}
                      </div>
                    </div>
                  </article>
                ))
              )}
            </div>
          </div>
        </div>

        {/* Shift Workflow sidebar — 280px Clarity panel */}
        <aside className="hidden w-[280px] shrink-0 flex-col border-l border-[#EBEDF1] bg-white xl:flex">
          <div className="flex-1 overflow-y-auto p-5">
            <h2 className="mb-5 font-headline text-[16px] font-semibold text-[#0C0F13]">Shift Workflow</h2>

            <div className="mb-4 rounded-[14px] border border-[#EBEDF1] bg-[#F5F6F8] p-4">
              <KpiLabel
                label="Open incidents"
                kpiId="open_critical"
                className="mb-2 font-label text-[11px] font-bold uppercase tracking-wide text-[#9AA1AB]"
              />
              <p className="font-headline text-[40px] font-bold leading-none text-[#C0362C]">
                {String(summary?.openCount ?? 0).padStart(2, "0")}
              </p>
              <div className="mt-3 flex gap-4 font-label text-[12px]">
                <span className="text-[#C0362C]">
                  <strong>{severityCounts.critical}</strong> critical
                </span>
                <span className="text-[#B7791F]">
                  <strong>{severityCounts.warning}</strong> warning
                </span>
              </div>
            </div>

            <div className="mb-4 grid grid-cols-2 gap-2">
              <div className="rounded-[10px] border border-[#EBEDF1] bg-white p-3">
                <p className="font-label text-[11px] text-[#9AA1AB]">Resolved</p>
                <p className="font-headline text-[22px] font-semibold text-[#0C0F13]">
                  {summary?.resolvedCount ?? 0}
                </p>
              </div>
              <div className="rounded-[10px] border border-[#EBEDF1] bg-white p-3">
                <p className="font-label text-[11px] text-[#9AA1AB]">False Pos.</p>
                <p className="font-headline text-[22px] font-semibold text-[#0C0F13]">
                  {summary?.falsePositiveCount ?? 0}
                </p>
              </div>
            </div>

            <div className="mb-4 rounded-[14px] border border-[#EBEDF1] bg-white p-4">
              <p className="mb-3 font-label text-[11px] font-bold uppercase tracking-wide text-[#9AA1AB]">
                Incidents by severity
              </p>
              <div className="space-y-2">
                {(
                  [
                    ["critical", "#C0362C"],
                    ["warning", "#B7791F"],
                    ["info", "#0059BB"],
                  ] as const
                ).map(([key, color]) => {
                  const count = severityCounts[key];
                  const pct = Math.round((count / totalSeverity) * 100);
                  return (
                    <div key={key}>
                      <div className="mb-1 flex justify-between font-label text-[11px] capitalize text-[#687079]">
                        <span>{key}</span>
                        <span className="font-semibold text-[#2B3340]">{count}</span>
                      </div>
                      <div className="h-1.5 overflow-hidden rounded-full bg-[#EBEDF1]">
                        <div
                          className="h-full rounded-full transition-all"
                          style={{ width: `${pct}%`, backgroundColor: color }}
                        />
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>

            {summary?.avgAckSeconds != null && (
              <div className="mb-4 rounded-[14px] border border-[#EBEDF1] bg-white p-4">
                <KpiLabel
                  label="Avg time to ack"
                  kpiId="avg_ack_time"
                  className="mb-1 font-label text-[11px] font-bold uppercase tracking-wide text-[#9AA1AB]"
                />
                <p className="font-headline text-[22px] font-semibold text-[#0C0F13]">
                  {summary.avgAckSeconds < 60
                    ? `${summary.avgAckSeconds}s`
                    : `${Math.round(summary.avgAckSeconds / 60)}m`}
                </p>
              </div>
            )}

            <div className="rounded-[14px] border border-[#EBEDF1] bg-white p-4">
              <KpiLabel
                label="Uptime rate"
                kpiId="uptime"
                className="mb-1 font-label text-[11px] font-bold uppercase tracking-wide text-[#9AA1AB]"
              />
              <div className="flex items-center justify-between">
                <span className="font-headline text-[22px] font-semibold text-[#0059BB]">
                  {summary?.uptime ?? "—"}
                </span>
                <Icon
                  name={uptimeTrendUp ? "trending_up" : "trending_down"}
                  className={uptimeTrendUp ? "text-[#1F8A5B]" : "text-[#B7791F]"}
                  size={20}
                />
              </div>
            </div>

            {summary?.topReasonCodes && summary.topReasonCodes.length > 0 && (
              <div className="mt-4 border-t border-[#EBEDF1] pt-4">
                <p className="mb-3 font-label text-[11px] font-bold uppercase tracking-wide text-[#9AA1AB]">
                  Top reason codes
                </p>
                <div className="space-y-2">
                  {summary.topReasonCodes.map((r) => (
                    <div key={r.code} className="flex items-center justify-between text-[13px]">
                      <span className="text-[#5A626C]">{r.code}</span>
                      <span className="font-label font-bold text-[#2B3340]">{r.count}</span>
                    </div>
                  ))}
                </div>
              </div>
            )}

            <div className="mt-4 border-t border-[#EBEDF1] pt-4">
              <ShiftAiSummaryPanel aiSummary={aiSummary} loading={loading} />
            </div>
          </div>

          <div className="border-t border-[#EBEDF1] p-5">
            <button
              type="button"
              onClick={() => void handleExport()}
              disabled={exporting || loading}
              className="flex min-h-[48px] w-full items-center justify-center gap-2 rounded-[10px] bg-[#161B22] font-label text-[13px] font-semibold text-white transition-colors hover:bg-[#2B3340] disabled:opacity-40"
            >
              <Icon name="check_circle" size={18} />
              {exporting ? "Exporting PDF…" : "Close out shift"}
            </button>
          </div>
        </aside>
      </div>

      {resolveEvent && (
        <EventResolveModal
          event={resolveEvent}
          reasonCodes={reasonCodes}
          open={Boolean(resolveEvent)}
          onClose={() => setResolveEvent(null)}
          onSubmit={handleResolveSubmit}
        />
      )}
    </AppShell>
  );
}
