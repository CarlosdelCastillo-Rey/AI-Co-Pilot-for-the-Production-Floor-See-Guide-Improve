"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { AppShell } from "@/components/layout/AppShell";
import { FloorStatusBanner } from "@/components/shared/FloorStatusBanner";
import { KpiHelp, KpiLabel } from "@/components/shared/KpiHelp";
import { Icon } from "@/components/ui/Icon";
import {
  fetchAnalyticsCoq,
  fetchAnalyticsHeatmap,
  fetchAnalyticsInsights,
  fetchAnalyticsOee,
  fetchAnalyticsPareto,
  fetchAnalyticsSummary,
  fetchTimelineEvents,
  getLiveCameraFeeds,
  type AnalyticsCoqApi,
  type AnalyticsHeatmapApi,
  type AnalyticsInsightsApi,
  type AnalyticsOeeApi,
  type AnalyticsParetoApi,
  type AnalyticsSummaryApi,
  type TimelineEventApi,
} from "@/lib/api";
import type { CameraFeed } from "@/lib/types";
import { cn } from "@/lib/cn";

const SHIFTS = ["morning", "evening", "night"] as const;

export function AnalyticsPageClient() {
  const [shift, setShift] = useState<(typeof SHIFTS)[number]>("morning");
  const [cameraId, setCameraId] = useState("");
  const [cameras, setCameras] = useState<CameraFeed[]>([]);
  const [summary, setSummary] = useState<AnalyticsSummaryApi | null>(null);
  const [heatmap, setHeatmap] = useState<AnalyticsHeatmapApi | null>(null);
  const [insights, setInsights] = useState<AnalyticsInsightsApi | null>(null);
  const [oee, setOee] = useState<AnalyticsOeeApi | null>(null);
  const [coq, setCoq] = useState<AnalyticsCoqApi | null>(null);
  const [pareto, setPareto] = useState<AnalyticsParetoApi | null>(null);
  const [openQueue, setOpenQueue] = useState<TimelineEventApi[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    void getLiveCameraFeeds().then((list) => {
      setCameras(list);
      if (list.length && !cameraId) setCameraId(list[0].id);
    });
  }, [cameraId]);

  useEffect(() => {
    if (!cameraId) return;
    const load = async () => {
      setLoading(true);
      const cam = cameraId;
      const [s, h, i, o, c, p, q] = await Promise.all([
        fetchAnalyticsSummary(shift, cam),
        fetchAnalyticsHeatmap(shift, cam),
        fetchAnalyticsInsights(shift, cam),
        fetchAnalyticsOee(shift, cam),
        fetchAnalyticsCoq(shift, cam),
        fetchAnalyticsPareto(shift, cam),
        fetchTimelineEvents({ limit: 10, resolutionStatus: "OPEN" }),
      ]);
      setSummary(s);
      setHeatmap(h);
      setInsights(i);
      setOee(o);
      setCoq(c);
      setPareto(p);
      setOpenQueue(q.filter((e) => e.severity === "critical").slice(0, 5));
      setLoading(false);
    };
    void load();
    const id = setInterval(() => void load(), 15_000);
    return () => clearInterval(id);
  }, [shift, cameraId]);

  const allClear = (summary?.openCriticalCount ?? 0) === 0;
  const criticalFlash = !allClear;

  return (
    <AppShell fullBleed>
      <FloorStatusBanner
        allClear={allClear}
        openCriticalCount={summary?.openCriticalCount ?? 0}
        openCount={summary?.openCriticalCount ?? openQueue.length}
      />
      <div
        className={cn(
          "flex h-[calc(100vh-4rem-4.5rem)] overflow-hidden",
          criticalFlash && "floor-page-critical",
        )}
      >
        <section className="relative flex w-[58%] flex-col border-r border-outline-variant">
          <Toolbar
            shift={shift}
            setShift={setShift}
            cameraId={cameraId}
            setCameraId={setCameraId}
            cameras={cameras}
            date={summary?.date}
          />
          <div className="blueprint-grid flex flex-grow flex-col gap-4 overflow-y-auto bg-white p-gutter">
            {loading && !oee ? (
              <p className="text-body-sm text-outline">Loading analytics…</p>
            ) : (
              <>
                <div className="grid grid-cols-2 gap-4">
                  <OeePanel oee={oee} summaryOee={summary?.oee} />
                  <CoqPanel coq={coq} />
                </div>
                <ParetoPanel pareto={pareto} />
                <HeatmapPanel heatmap={heatmap} cameraId={cameraId} />
              </>
            )}
          </div>
        </section>

        <section className="flex w-[42%] flex-col overflow-y-auto bg-surface-container-low p-6">
          <h2 className="mb-6 flex items-center gap-2 font-headline text-headline-md">
            <Icon name="insert_chart" className="text-primary" />
            Operational Insights
          </h2>
          <div className="space-y-gutter">
            <InsightCard
              title="Flow Efficiency"
              kpiId="flow_efficiency"
              value={summary?.flowEfficiency ?? insights?.flowEfficiency ?? "—"}
              trend={summary?.flowEfficiencyTrend ?? "—"}
              history={insights?.flowHistory ?? []}
            />
            <InsightCard
              title="OEE Composite"
              kpiId="oee"
              value={summary?.oee ?? (oee ? `${oee.oee}%` : "—")}
              trend={summary?.uptime ? `Uptime ${summary.uptime}` : "—"}
              history={[]}
            />
            <DowntimeCard stations={insights?.downtimeByStation ?? []} />
            <BottleneckCard items={insights?.bottlenecks ?? []} />
            <WorkflowQueueCard items={openQueue} />
            <RecommendationCard text={insights?.recommendation ?? "Loading insights…"} />
          </div>
        </section>
      </div>
    </AppShell>
  );
}

function Toolbar({
  shift,
  setShift,
  cameraId,
  setCameraId,
  cameras,
  date,
}: {
  shift: (typeof SHIFTS)[number];
  setShift: (s: (typeof SHIFTS)[number]) => void;
  cameraId: string;
  setCameraId: (id: string) => void;
  cameras: CameraFeed[];
  date?: string;
}) {
  return (
    <div className="flex h-14 shrink-0 items-center justify-between border-b border-outline-variant bg-surface px-6">
      <div className="flex items-center gap-4">
        <div className="flex items-center gap-2 rounded border border-outline-variant bg-surface-container-low px-3 py-1.5">
          <Icon name="calendar_today" size={16} />
          <span className="text-label-sm">{date ?? "Today"}</span>
        </div>
        <div className="flex gap-1 rounded border border-outline-variant bg-surface-container-high p-1">
          {SHIFTS.map((s) => (
            <button
              key={s}
              type="button"
              onClick={() => setShift(s)}
              className={cn(
                "min-h-touch rounded px-3 py-1 text-label-sm capitalize",
                shift === s ? "bg-surface font-bold text-primary shadow-sm" : "hover:bg-surface/50",
              )}
            >
              {s}
            </button>
          ))}
        </div>
      </div>
      <div className="flex items-center gap-3">
        <label className="text-label-sm text-on-surface-variant">Camera:</label>
        <select
          className="min-h-touch rounded border-outline-variant bg-surface py-1 pl-2 pr-8 text-label-sm focus:ring-primary"
          value={cameraId}
          onChange={(e) => setCameraId(e.target.value)}
        >
          {cameras.map((c) => (
            <option key={c.id} value={c.id}>
              {c.name} ({c.location})
            </option>
          ))}
        </select>
      </div>
    </div>
  );
}

function OeePanel({ oee, summaryOee }: { oee: AnalyticsOeeApi | null; summaryOee?: string }) {
  const segments = [
    { label: "Availability", kpiId: "oee_availability", value: oee?.availability ?? 0, color: "bg-primary" },
    { label: "Performance", kpiId: "oee_performance", value: oee?.performance ?? 0, color: "bg-tertiary-fixed-dim" },
    { label: "Quality", kpiId: "oee_quality", value: oee?.quality ?? 0, color: "bg-success" },
  ];

  return (
    <article className="rounded-xl border border-outline-variant bg-surface p-md">
      <KpiLabel label="OEE — Overall Equipment Effectiveness" kpiId="oee" className="mb-2 text-label-sm uppercase text-on-surface-variant" />
      <p className="mb-4 font-headline text-display-lg industrial-kpi text-primary">
        {summaryOee ?? (oee ? `${oee.oee}%` : "—")}
      </p>
      <div className="space-y-3">
        {segments.map((s) => (
          <div key={s.label}>
            <div className="mb-1 flex justify-between text-label-sm">
              <KpiLabel label={s.label} kpiId={s.kpiId} />
              <span className="font-bold">{s.value.toFixed(1)}%</span>
            </div>
            <div className="h-3 overflow-hidden rounded-full bg-surface-container-high">
              <div className={cn("h-full rounded-full", s.color)} style={{ width: `${s.value}%` }} />
            </div>
          </div>
        ))}
      </div>
    </article>
  );
}

function CoqPanel({ coq }: { coq: AnalyticsCoqApi | null }) {
  return (
    <article className="rounded-xl border border-outline-variant bg-surface p-md">
      <KpiLabel label="Cost of Quality (CoQ)" kpiId="coq" className="mb-2 text-label-sm uppercase text-on-surface-variant" />
      <p className="mb-4 font-headline text-display-lg industrial-kpi text-error">
        {coq != null ? `$${coq.totalCostUsd.toLocaleString(undefined, { minimumFractionDigits: 0 })}` : "—"}
      </p>
      <div className="space-y-2 text-body-sm">
        <div className="flex justify-between">
          <span className="text-on-surface-variant">Downtime ({coq?.downtimeMinutes.toFixed(1) ?? 0} min @ ${coq?.lineCostPerMinute ?? 0}/min)</span>
          <span className="font-bold">${coq?.downtimeCostUsd.toFixed(0) ?? "0"}</span>
        </div>
        <div className="flex justify-between">
          <span className="text-on-surface-variant">Scrap ({coq?.scrapUnits ?? 0} units @ ${coq?.materialCostPerUnit ?? 0}/unit)</span>
          <span className="font-bold">${coq?.scrapCostUsd.toFixed(0) ?? "0"}</span>
        </div>
      </div>
    </article>
  );
}

function ParetoPanel({ pareto }: { pareto: AnalyticsParetoApi | null }) {
  const items = pareto?.items ?? [];
  const max = items[0]?.count ?? 1;

  return (
    <article className="rounded-xl border border-outline-variant bg-surface p-md">
      <KpiLabel
        label={`Pareto — Root Cause Tags (${pareto?.totalTagged ?? 0} closed)`}
        kpiId="pareto"
        className="mb-4 text-label-sm uppercase text-on-surface-variant"
      />
      {items.length === 0 ? (
        <p className="text-body-sm text-outline">Resolve incidents on Timeline to populate Pareto chart.</p>
      ) : (
        <div className="space-y-3">
          {items.map((item) => (
            <div key={item.code}>
              <div className="mb-1 flex justify-between text-label-sm">
                <span>{item.label}</span>
                <span className="font-bold">
                  {item.count} ({item.pct}%)
                </span>
              </div>
              <div className="h-2 overflow-hidden rounded-full bg-surface-container-high">
                <div
                  className="h-full rounded-full bg-primary"
                  style={{ width: `${(item.count / max) * 100}%` }}
                />
              </div>
            </div>
          ))}
        </div>
      )}
    </article>
  );
}

function HeatmapPanel({ heatmap, cameraId }: { heatmap: AnalyticsHeatmapApi | null; cameraId: string }) {
  const cells = heatmap?.grid.cells ?? [];
  const flatMax = Math.max(...cells.flat(), 0.01);

  return (
    <article className="relative min-h-[280px] overflow-hidden rounded-xl border-2 border-outline-variant/30 bg-surface-container-lowest">
      {cells.length > 0 ? (
        <div
          className="absolute inset-0 grid gap-px p-2"
          style={{
            gridTemplateColumns: `repeat(${heatmap?.grid.width ?? 10}, minmax(0, 1fr))`,
            gridTemplateRows: `repeat(${heatmap?.grid.height ?? 10}, minmax(0, 1fr))`,
          }}
        >
          {cells.flat().map((v, i) => {
            const intensity = v / flatMax;
            return (
              <div
                key={i}
                className="rounded-sm"
                style={{
                  backgroundColor: `rgba(186, 26, 26, ${0.08 + intensity * 0.72})`,
                }}
              />
            );
          })}
        </div>
      ) : (
        <div className="heatmap-overlay absolute inset-0 opacity-40" />
      )}
      <div className="absolute right-6 top-6 rounded-md border border-black/10 bg-black/5 px-3 py-2 font-label text-label-sm backdrop-blur-sm">
        <p className="font-bold">GRID: {cameraId || "—"}</p>
        <p className="opacity-70">SENSORS ACTIVE: {heatmap?.sensorsActive ?? 0}</p>
        <p className="mt-1 flex items-center gap-1 font-bold text-error">
          ANOMALIES: {String(heatmap?.anomalyCount ?? 0).padStart(2, "0")}
          <KpiHelp kpiId="heatmap_anomalies" />
        </p>
        {heatmap?.source && (
          <p className="mt-1 text-[10px] opacity-60">source: {heatmap.source}</p>
        )}
      </div>
    </article>
  );
}

function WorkflowQueueCard({ items }: { items: TimelineEventApi[] }) {
  return (
    <article className="overflow-hidden rounded-xl border border-outline-variant bg-surface">
      <div className="border-b border-outline-variant bg-error/5 px-4 py-2">
        <KpiLabel label="Open Critical Queue" kpiId="open_critical" className="text-label-sm font-bold uppercase" />
      </div>
      <div className="divide-y divide-outline-variant">
        {items.length === 0 ? (
          <p className="p-4 text-body-sm text-outline">No open critical events.</p>
        ) : (
          items.map((item) => (
            <Link
              key={item.id}
              href="/timeline"
              className="flex min-h-touch items-center gap-3 p-4 transition-colors hover:bg-surface-container-low"
            >
              <Icon name="report" filled className="text-error" />
              <div className="flex-1">
                <p className="text-body-sm font-bold">{item.title}</p>
                <p className="text-label-sm text-on-surface-variant">
                  {item.time} · {item.cameraId}
                </p>
              </div>
              <Icon name="chevron_right" className="text-outline" />
            </Link>
          ))
        )}
      </div>
    </article>
  );
}

function InsightCard({
  title,
  kpiId,
  value,
  trend,
  history,
}: {
  title: string;
  kpiId: string;
  value: string;
  trend: string;
  history: { date: string; value: number }[];
}) {
  const max = Math.max(...history.map((h) => h.value), 1);
  const trendUp = trend.startsWith("+");

  return (
    <article className="rounded-xl border border-outline-variant bg-surface p-md">
      <div className="mb-4 flex items-start justify-between">
        <div>
          <KpiLabel label={title} kpiId={kpiId} className="text-label-sm uppercase text-on-surface-variant" />
          <h3 className="font-headline text-headline-md">{value}</h3>
        </div>
        <span
          className={cn(
            "flex items-center gap-1 text-sm font-bold",
            trendUp ? "text-green-600" : trend.startsWith("-") ? "text-error" : "text-outline",
          )}
        >
          <Icon name={trendUp ? "trending_up" : trend.startsWith("-") ? "trending_down" : "remove"} size={16} />
          {trend}
        </span>
      </div>
      {history.length > 0 ? (
        <div className="flex h-24 items-end gap-1 px-2">
          {history.map((h, i) => (
            <div
              key={h.date}
              title={`${h.date}: ${h.value}%`}
              className={cn(
                "w-full rounded-t-sm",
                i === history.length - 1 ? "border-t-2 border-primary bg-primary/30" : "bg-primary/10",
              )}
              style={{ height: `${(h.value / max) * 100}%`, minHeight: "4px" }}
            />
          ))}
        </div>
      ) : (
        <p className="text-body-sm text-outline">No 7-day history yet.</p>
      )}
    </article>
  );
}

function DowntimeCard({
  stations,
}: {
  stations: { name: string; minutes: number; widthPct: string; critical: boolean }[];
}) {
  return (
    <article className="rounded-xl border border-outline-variant bg-surface p-md">
      <KpiLabel label="Est. Downtime per Station (min)" kpiId="downtime_by_station" className="mb-4 text-label-sm uppercase text-on-surface-variant" />
      {stations.length === 0 ? (
        <p className="text-body-sm text-outline">No critical or warning events for this scope today.</p>
      ) : (
        <div className="space-y-4">
          {stations.map((s, i) => (
            <div key={`${s.name}-${i}`} className="space-y-1">
              <div className="flex justify-between text-label-sm">
                <span>{s.name}</span>
                <span className={s.critical ? "font-bold text-error" : "font-bold"}>{s.minutes.toFixed(1)}m</span>
              </div>
              <div className="h-2 w-full overflow-hidden rounded-full bg-surface-container-high">
                <div
                  className={cn("h-full rounded-full", s.critical ? "bg-error" : "bg-primary")}
                  style={{ width: s.widthPct }}
                />
              </div>
            </div>
          ))}
        </div>
      )}
    </article>
  );
}

function BottleneckCard({
  items,
}: {
  items: { id: string; title: string; severity: string; description: string; critical: boolean }[];
}) {
  return (
    <article className="overflow-hidden rounded-xl border border-outline-variant bg-surface">
      <div className="border-b border-outline-variant bg-secondary-container/50 px-4 py-2">
        <p className="text-label-sm font-bold uppercase">Recent Bottleneck Events</p>
      </div>
      <div className="divide-y divide-outline-variant">
        {items.length === 0 ? (
          <p className="p-4 text-body-sm text-outline">No bottlenecks detected today.</p>
        ) : (
          items.map((item) => (
            <div key={item.id} className={cn("flex gap-3 p-4", item.critical && "bg-error/5")}>
              <Icon
                name={item.critical ? "report" : "warning"}
                filled={item.critical}
                className={item.critical ? "text-error" : "text-tertiary"}
              />
              <div>
                <p className="text-body-sm font-bold">{item.title}</p>
                <p className="text-label-sm text-on-surface-variant">{item.description}</p>
              </div>
            </div>
          ))
        )}
      </div>
    </article>
  );
}

function RecommendationCard({ text }: { text: string }) {
  return (
    <article className="flex items-center gap-4 rounded-xl bg-inverse-surface p-md">
      <div className="flex h-12 w-12 shrink-0 items-center justify-center rounded-lg bg-primary-fixed-dim/20">
        <Icon name="auto_awesome" className="text-primary-fixed-dim" />
      </div>
      <div>
        <p className="text-body-sm font-bold text-surface-bright">Operational Recommendation</p>
        <p className="text-body-sm text-on-surface-variant">{text}</p>
      </div>
    </article>
  );
}
