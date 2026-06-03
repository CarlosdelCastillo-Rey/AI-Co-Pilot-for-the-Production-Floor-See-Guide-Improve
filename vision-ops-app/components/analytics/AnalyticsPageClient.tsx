"use client";

import { useEffect, useMemo, useState } from "react";
import Link from "next/link";
import { AppShell } from "@/components/layout/AppShell";
import { DATA_RESET_EVENT } from "@/components/layout/Sidebar";
import { FloorStatusBanner } from "@/components/shared/FloorStatusBanner";
import { KpiLabel } from "@/components/shared/KpiHelp";
import { Icon } from "@/components/ui/Icon";
import {
  fetchAnalyticsCoq,
  fetchAnalyticsHeatmap,
  fetchAnalyticsInsights,
  fetchAnalyticsOee,
  fetchAnalyticsPareto,
  fetchAnalyticsSummary,
  fetchHarAnalyticsDaily,
  fetchHarAnalyticsPlant,
  fetchHarAnalyticsRealtime,
  fetchTimelineEvents,
  getLiveCameraFeeds,
  type HarAnalyticsDailyApi,
  type HarPlantAnalyticsApi,
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
import {
  OEE_TARGET_PCT,
  coqBudgetTone,
  coqBudgetUsd,
  formatShiftLabel,
  heatmapToFloorZones,
  oeeTargetMarkerLeft,
  paretoTopThreePct,
  parseTrendDirection,
  type FloorZone,
} from "./analytics-utils";
import "./analytics-page.css";

const SHIFTS = ["morning", "evening", "night"] as const;

type AttentionItem = {
  id: string;
  title: string;
  description: string;
  meta: string;
  critical: boolean;
};

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
  const [harDaily, setHarDaily] = useState<HarAnalyticsDailyApi | null>(null);
  const [harRealtimeCount, setHarRealtimeCount] = useState(0);
  const [harPlant, setHarPlant] = useState<HarPlantAnalyticsApi | null>(null);

  useEffect(() => {
    void getLiveCameraFeeds().then((list) => {
      setCameras(list);
      if (list.length && !cameraId) {
        const harFirst = list.find((c) => c.id.startsWith("cam-har"));
        setCameraId(harFirst?.id ?? list[0].id);
      }
    });
  }, [cameraId]);

  useEffect(() => {
    if (!cameraId) return;
    const load = async () => {
      setLoading(true);
      const cam = cameraId;
      const isHarCam = cam.startsWith("cam-har");
      const harScope = isHarCam ? cam : undefined;
      const [s, h, i, o, c, p, q, harD, harRt, harP] = await Promise.all([
        fetchAnalyticsSummary(shift, cam),
        fetchAnalyticsHeatmap(shift, cam),
        fetchAnalyticsInsights(shift, cam),
        fetchAnalyticsOee(shift, cam),
        fetchAnalyticsCoq(shift, cam),
        fetchAnalyticsPareto(shift, cam),
        fetchTimelineEvents({ limit: 12, resolutionStatus: "OPEN" }),
        isHarCam ? fetchHarAnalyticsDaily(cam) : Promise.resolve(null),
        isHarCam ? fetchHarAnalyticsRealtime(cam) : Promise.resolve(null),
        fetchHarAnalyticsPlant(harScope),
      ]);
      setSummary(s);
      setHeatmap(h);
      setInsights(i);
      setOee(o);
      setCoq(c);
      setPareto(p);
      setOpenQueue(q.filter((e) => e.severity === "critical"));
      setHarDaily(harD);
      setHarRealtimeCount(harRt?.inferenceCount ?? 0);
      setHarPlant(harP ?? i?.harActions ?? null);
      setLoading(false);
    };
    void load();
    const id = setInterval(() => void load(), 15_000);
    const onReset = () => void load();
    window.addEventListener(DATA_RESET_EVENT, onReset);
    return () => {
      clearInterval(id);
      window.removeEventListener(DATA_RESET_EVENT, onReset);
    };
  }, [shift, cameraId]);

  const allClear = (summary?.openCriticalCount ?? 0) === 0;
  const criticalFlash = !allClear;
  const selectedCamera = cameras.find((c) => c.id === cameraId);
  const floorZones = useMemo(() => heatmapToFloorZones(heatmap), [heatmap]);
  const attentionItems = useMemo(
    () => buildAttentionFeed(openQueue, insights?.bottlenecks ?? []),
    [openQueue, insights?.bottlenecks],
  );

  const oeeNumeric =
    oee?.oee ??
    (summary?.oee ? parseFloat(summary.oee.replace("%", "")) : NaN);
  const oeeDisplay = Number.isFinite(oeeNumeric) ? `${oeeNumeric}%` : summary?.oee ?? "—";
  const oeeValue = Number.isFinite(oeeNumeric) ? oeeNumeric : 0;
  const flowTrend = parseTrendDirection(summary?.flowEfficiencyTrend ?? "—");
  const coqBudget = coq ? coqBudgetUsd(coq.totalCostUsd) : 8000;
  const coqPct = coq ? Math.min(100, Math.round((coq.totalCostUsd / coqBudget) * 100)) : 0;
  const paretoTop3 = pareto ? paretoTopThreePct(pareto.items) : null;
  const severityTags =
    insights?.severityTags ??
    heatmap?.severityTags ??
    harPlant?.severityTags ?? { critical: 0, warning: 0, info: 0 };
  const productivityScore = harPlant?.productivityScore;

  return (
    <AppShell fullBleed>
      <div className="an-page flex h-[calc(100dvh-4rem)] max-h-[calc(100dvh-4rem)] flex-col overflow-hidden">
        <FloorStatusBanner
          className="shrink-0"
          allClear={allClear}
          openCriticalCount={summary?.openCriticalCount ?? 0}
          openCount={summary?.openCriticalCount ?? openQueue.length}
          actionHref="/timeline"
          actionLabel="Go to Timeline"
        />

        <div
          className={cn(
            "flex min-h-0 flex-1 flex-col overflow-hidden",
            criticalFlash && "floor-page-critical",
          )}
        >
          <Toolbar
            shift={shift}
            setShift={setShift}
            cameraId={cameraId}
            setCameraId={setCameraId}
            cameras={cameras}
            date={summary?.date}
            cameraLabel={
              selectedCamera
                ? `${selectedCamera.name} (${selectedCamera.location})`
                : cameraId || "—"
            }
          />

          <div className="an-body">
            <div className="an-left">
              {loading && !oee ? (
                <p className="text-body-sm text-outline">Loading analytics…</p>
              ) : (
                <>
                  <div className="an-score">
                    <OeeCard oee={oee} display={oeeDisplay} numeric={oeeValue} shift={shift} />
                    <CoqCard coq={coq} budgetUsd={coqBudget} budgetPct={coqPct} />
                  </div>
                  <HarPlant360Card plant={harPlant} />
                  {cameraId.startsWith("cam-har") && (
                    <HarActivityMetricsCard
                      daily={harDaily}
                      realtimeCount={harRealtimeCount}
                    />
                  )}
                  {harPlant?.hasData && (harPlant.actionPareto?.length ?? 0) > 0 ? (
                    <ActionParetoCard items={harPlant.actionPareto} />
                  ) : null}
                  <ParetoCard pareto={pareto} topThreePct={paretoTop3} />
                  <FloorHeatmapCard
                    heatmap={heatmap}
                    cameraId={cameraId}
                    zones={floorZones}
                    anomalyCount={heatmap?.anomalyCount ?? 0}
                    severityTags={severityTags}
                  />
                </>
              )}
            </div>

            <aside className="an-right">
              <div className="an-sech">
                <span className="ico">
                  <Icon name="insert_chart" size={16} />
                </span>
                <h2>Operational Insights</h2>
              </div>

              <div className="an-itiles">
                <InsightTile
                  label="Flow Efficiency"
                  kpiId="flow_efficiency"
                  value={summary?.flowEfficiency ?? insights?.flowEfficiency ?? "—"}
                  trend={summary?.flowEfficiencyTrend ?? "—"}
                  trendDir={flowTrend}
                  history={insights?.flowHistory ?? []}
                />
                <InsightTile
                  label={productivityScore != null ? "Action Productivity" : "OEE Composite"}
                  kpiId={productivityScore != null ? "har_productivity" : "oee"}
                  value={
                    productivityScore != null
                      ? `${productivityScore}%`
                      : oeeDisplay
                  }
                  trend={
                    productivityScore != null
                      ? `${harPlant?.assembleSharePct ?? 0}% ${harPlant?.primaryActionLabel ?? "primary"}`
                      : summary?.uptime
                        ? `Uptime ${summary.uptime}`
                        : "—"
                  }
                  trendDir={productivityScore != null && productivityScore >= 70 ? "up" : productivityScore != null ? "down" : "flat"}
                  history={[]}
                />
              </div>

              <SeverityTagsCard tags={severityTags} actionCostUsd={harPlant?.actionCostUsd ?? coq?.actionDeviationCostUsd} />

              <DowntimeCard stations={insights?.downtimeByStation ?? []} />
              <AttentionFeed items={attentionItems} />
              <AiRecommendation text={insights?.recommendation ?? "Loading insights…"} />
            </aside>
          </div>
        </div>
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
  cameraLabel,
}: {
  shift: (typeof SHIFTS)[number];
  setShift: (s: (typeof SHIFTS)[number]) => void;
  cameraId: string;
  setCameraId: (id: string) => void;
  cameras: CameraFeed[];
  date?: string;
  cameraLabel: string;
}) {
  return (
    <div className="an-toolbar">
      <span className="an-date">
        <Icon name="calendar_today" size={15} className="text-outline" />
        {date ?? "Today"}
      </span>
      <div className="an-seg">
        {SHIFTS.map((s) => (
          <button key={s} type="button" className={shift === s ? "on" : ""} onClick={() => setShift(s)}>
            {s}
          </button>
        ))}
      </div>
      <span className="flex-1" />
      <label className="an-cam-select">
        <span className="lab">Camera</span>
        <span className="dot" aria-hidden />
        <select value={cameraId} onChange={(e) => setCameraId(e.target.value)}>
          {cameras.length === 0 ? (
            <option value="">No cameras</option>
          ) : (
            cameras.map((c) => (
              <option key={c.id} value={c.id}>
                {c.name} — {c.location}
              </option>
            ))
          )}
        </select>
        <Icon name="expand_more" size={15} className="text-outline" />
      </label>
      <span className="sr-only">Selected: {cameraLabel}</span>
    </div>
  );
}

function CardHeader({ label, kpiId, trailing }: { label: string; kpiId: string; trailing?: React.ReactNode }) {
  return (
    <div className="an-ac-h">
      <KpiLabel label={label} kpiId={kpiId} className="an-ac-lab" />
      {trailing ? <span className="ml-auto text-label-sm text-outline">{trailing}</span> : null}
    </div>
  );
}

function OeeCard({
  oee,
  display,
  numeric,
  shift,
}: {
  oee: AnalyticsOeeApi | null;
  display: string;
  numeric: number;
  shift: string;
}) {
  const segments = [
    { label: "Availability", kpiId: "oee_availability", value: oee?.availability ?? 0, cls: "avail", tgt: 90 },
    { label: "Performance", kpiId: "oee_performance", value: oee?.performance ?? 0, cls: "perf", tgt: 95 },
    { label: "Quality", kpiId: "oee_quality", value: oee?.quality ?? 0, cls: "qual", tgt: 99 },
  ] as const;

  const gap = numeric - OEE_TARGET_PCT;
  const trendLabel =
    gap > 0 ? `+${gap.toFixed(1)} pts` : gap < 0 ? `${gap.toFixed(1)} pts` : "at target";
  const trendDir = gap > 0.5 ? "up" : gap < -0.5 ? "down" : "flat";

  return (
    <article className="an-ac an-oee-card">
      <CardHeader label="OEE — Overall Equipment Effectiveness" kpiId="oee" />
      <div className="an-oee-top">
        <span className="an-oee-big">
          {display.endsWith("%") ? (
            <>
              {display.slice(0, -1)}
              <span style={{ fontSize: "26px" }}>%</span>
            </>
          ) : (
            display
          )}
        </span>
        <span className={cn("an-oee-trend", trendDir)}>
          <Icon
            name={trendDir === "up" ? "trending_up" : trendDir === "down" ? "trending_down" : "remove"}
            size={14}
          />
          {trendLabel}
        </span>
      </div>
      <p className="an-oee-sub">
        {formatShiftLabel(shift)} · target {OEE_TARGET_PCT}.0% · world-class ≥ {OEE_TARGET_PCT}%
      </p>
      {segments.map((s) => (
        <div key={s.label} className="an-oee-seg">
          <div className="r">
            <KpiLabel label={s.label} kpiId={s.kpiId} className="nm" />
            <span className="vv">{s.value.toFixed(1)}%</span>
          </div>
          <div className="an-oee-bar">
            <i className={s.cls} style={{ width: `${s.value}%` }} />
            <span className="tgt" style={{ left: oeeTargetMarkerLeft(s.tgt) }} />
          </div>
        </div>
      ))}
    </article>
  );
}

function CoqCard({
  coq,
  budgetUsd,
  budgetPct,
}: {
  coq: AnalyticsCoqApi | null;
  budgetUsd: number;
  budgetPct: number;
}) {
  const tone = coqBudgetTone(budgetPct);
  const budgetLabel =
    budgetUsd >= 1000 ? `$${(budgetUsd / 1000).toFixed(1)}k` : `$${budgetUsd}`;

  return (
    <article className="an-ac an-coq-card">
      <CardHeader label="Cost of Quality (CoQ)" kpiId="coq" />
      <div className={cn("an-coq-big", tone)}>{coq != null ? `$${coq.totalCostUsd.toLocaleString()}` : "—"}</div>
      <div className="an-coq-budget">
        <span className="meter">
          <i style={{ width: `${budgetPct}%` }} />
        </span>
        <span className="pct">{budgetPct}% of {budgetLabel} budget</span>
      </div>
      <div className="an-coq-rows">
        <div className="an-coq-row">
          <span className="sw dt" />
          <span className="lbl">
            <b>Downtime</b> · {coq?.downtimeMinutes.toFixed(1) ?? 0} min @ ${coq?.lineCostPerMinute ?? 0}/min
          </span>
          <span className="amt">${coq?.downtimeCostUsd.toFixed(0) ?? "0"}</span>
        </div>
        <div className="an-coq-row">
          <span className="sw sc" />
          <span className="lbl">
            <b>Scrap</b> · {coq?.scrapUnits ?? 0} units @ ${coq?.materialCostPerUnit ?? 0}/unit
          </span>
          <span className="amt">${coq?.scrapCostUsd.toFixed(0) ?? "0"}</span>
        </div>
        {(coq?.actionDeviationCostUsd ?? 0) > 0 ? (
          <div className="an-coq-row">
            <span className="sw act" />
            <span className="lbl">
              <b>Action deviations</b> · {coq?.actionDowntimeMinutes?.toFixed(1) ?? 0} min est. rework
            </span>
            <span className="amt">${coq?.actionDeviationCostUsd?.toFixed(0) ?? "0"}</span>
          </div>
        ) : null}
      </div>
    </article>
  );
}

function ParetoCard({
  pareto,
  topThreePct,
}: {
  pareto: AnalyticsParetoApi | null;
  topThreePct: number | null;
}) {
  const items = pareto?.items ?? [];
  const max = items[0]?.count ?? 1;

  return (
    <article className="an-ac an-pareto-card">
      <CardHeader
        label={`Pareto — Root Cause Tags · ${pareto?.totalTagged ?? 0} closed`}
        kpiId="pareto"
      />
      {items.length === 0 ? (
        <p className="px-4 pb-4 text-body-sm text-outline">
          Resolve incidents on Timeline to populate the Pareto chart.
        </p>
      ) : (
        <>
          <div className="an-pareto-rows">
            {items.map((item, idx) => (
              <div key={item.code} className="an-prow">
                <span className="rk">{idx + 1}</span>
                <div>
                  <div className="nm">{item.label}</div>
                  <div className="track">
                    <i style={{ width: `${(item.count / max) * 100}%` }} />
                  </div>
                </div>
                <div className="nums">
                  <span className="c">{item.count}</span> <span className="p">{item.pct}%</span>
                </div>
              </div>
            ))}
          </div>
          {topThreePct != null ? (
            <div className="an-pareto-cum">
              <span className="k">Vital few</span>
              <span className="flex-1" />
              <span className="k">
                Top 3 causes drive <b>{topThreePct}%</b> of stoppages
              </span>
            </div>
          ) : null}
        </>
      )}
    </article>
  );
}

function FloorHeatmapCard({
  heatmap,
  cameraId,
  zones,
  anomalyCount,
  severityTags,
}: {
  heatmap: AnalyticsHeatmapApi | null;
  cameraId: string;
  zones: FloorZone[];
  anomalyCount: number;
  severityTags: { critical: number; warning: number; info: number };
}) {
  const bottleneck = zones.find((z) => z.isBottleneck);
  const source = heatmap?.source ?? "unknown";
  const sourceLabel =
    heatmap?.sourceLabel ??
    (source === "stored"
      ? "demo seed"
      : source === "har_actions"
        ? "HAR actions"
        : source === "events_fallback"
          ? "event density"
          : source);

  return (
    <article className="an-ac an-floor-card">
      <CardHeader
        label="Spatial Activity Heatmap · Line floor"
        kpiId="heatmap_anomalies"
        trailing={source === "har_actions" ? "HAR action density" : "Dwell density · last 8h"}
      />
      <div className="an-heatmap-source">
        <span className={cn("an-src-badge", source === "har_actions" ? "live" : source === "stored" ? "demo" : "fallback")}>
          {sourceLabel}
        </span>
        {source === "stored" ? (
          <span className="an-src-hint">Demo grid from seed data — select a HAR camera for live action heatmap.</span>
        ) : source === "events_fallback" ? (
          <span className="an-src-hint">Synthetic grid from today&apos;s event severity counts.</span>
        ) : null}
      </div>
      <div className="an-floor">
        <div className="flow-line" />
        {zones.map((z) => (
          <div
            key={z.id}
            className={cn("an-zone", z.heatClass)}
            style={{ left: z.left, top: z.id === "IN" ? "30%" : "18%", width: z.width, height: z.id === "IN" ? "40%" : "64%" }}
          >
            {z.isBottleneck ? <span className="pulse" /> : null}
            <span className="zn">{z.id}</span>
            <span className="zt">{z.title}</span>
            <span className="zv">{z.label}</span>
          </div>
        ))}
        {anomalyCount > 0 ? (
          <div className="anomaly-badge" style={{ left: bottleneck ? "57.5%" : "50%", top: "9%" }}>
            <Icon name="warning" size={11} />
            {anomalyCount} anomal{anomalyCount === 1 ? "y" : "ies"}
          </div>
        ) : null}
        <div className="an-floor-meta">
          <span className="dt" />
          {heatmap?.sensorsActive ?? 0} sensors active · grid {cameraId}
          {severityTags.critical + severityTags.warning + severityTags.info > 0
            ? ` · ${severityTags.critical} crit / ${severityTags.warning} warn / ${severityTags.info} info`
            : ""}
        </div>
      </div>
      <div className="an-heat-legend">
        <span className="k">Idle</span>
        <span className="ramp" />
        <span className="k">Congested</span>
      </div>
    </article>
  );
}

function InsightTile({
  label,
  kpiId,
  value,
  trend,
  trendDir,
  history,
}: {
  label: string;
  kpiId: string;
  value: string;
  trend: string;
  trendDir: "up" | "down" | "flat";
  history: { date: string; value: number }[];
}) {
  const max = Math.max(...history.map((h) => h.value), 1);

  return (
    <article className="an-ac an-itile">
      <KpiLabel label={label} kpiId={kpiId} className="lab" />
      <div className="val">{value}</div>
      <span className={cn("trend", trendDir)}>
        <Icon
          name={trendDir === "up" ? "trending_up" : trendDir === "down" ? "trending_down" : "remove"}
          size={13}
        />
        {trend}
      </span>
      {history.length > 0 ? (
        <div className="an-spark">
          {history.map((h, i) => (
            <i
              key={h.date}
              title={`${h.date}: ${h.value}%`}
              className={i === history.length - 1 ? "cur" : undefined}
              style={{ height: `${(h.value / max) * 100}%`, minHeight: "4px" }}
            />
          ))}
        </div>
      ) : (
        <p className="mt-2 text-label-sm text-outline">No 7-day history yet.</p>
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
    <article className="an-ac an-dt-card">
      <CardHeader label="Est. Downtime per Station (min)" kpiId="downtime_by_station" />
      {stations.length === 0 ? (
        <p className="px-4 pb-4 text-body-sm text-outline">No critical or warning events for this scope today.</p>
      ) : (
        <div className="an-dt-rows">
          {stations.map((s, i) => (
            <div key={`${s.name}-${i}`} className="an-dt-row">
              <div className="r">
                <span className="nm">{s.name}</span>
                <span className={cn("mn", s.critical && "crit")}>{s.minutes.toFixed(1)}m</span>
              </div>
              <div className="an-dt-track">
                <i className={s.critical ? "crit" : undefined} style={{ width: s.widthPct }} />
              </div>
            </div>
          ))}
        </div>
      )}
    </article>
  );
}

function AttentionFeed({ items }: { items: AttentionItem[] }) {
  return (
    <article className="an-ac an-feed-card">
      <div className="an-feed-head shrink-0">
        <span className="an-ac-lab">Needs Attention</span>
        {items.length > 0 ? <span className="cnt">{items.length}</span> : null}
      </div>
      {items.length === 0 ? (
        <div className="an-clear-state">
          <span className="ic">
            <Icon name="check_circle" />
          </span>
          <div>
            <p className="text-body-sm font-semibold text-on-surface">All clear for this shift</p>
            <p className="text-label-sm text-outline">No open critical items or bottlenecks flagged.</p>
          </div>
        </div>
      ) : (
        <div className="an-feed-scroll">
          {items.map((item) => (
            <Link key={item.id} href="/timeline" className="an-feed-item">
              <span className={cn("ic", item.critical ? "crit" : "warn")}>
                <Icon name={item.critical ? "report" : "warning"} filled={item.critical} size={16} />
              </span>
              <div className="bd">
                <p className="ti">{item.title}</p>
                <p className="de">{item.description}</p>
                <p className="mt">{item.meta}</p>
              </div>
              <Icon name="chevron_right" className="text-outline" size={16} />
            </Link>
          ))}
        </div>
      )}
    </article>
  );
}

function AiRecommendation({ text }: { text: string }) {
  const parts = text.split(/(?<=[.!?])\s+/);
  const lead = parts[0] ?? text;
  const rest = parts.slice(1).join(" ");

  return (
    <article className="an-airec">
      <div className="an-airec-h">
        <span className="ic">
          <Icon name="auto_awesome" size={17} className="text-[#bcd4ff]" />
        </span>
        <span className="t">Operational Recommendation</span>
        <span className="badge-ai">VisionOps AI</span>
      </div>
      <p>
        <b>{lead}</b>
        {rest ? ` ${rest}` : ""}
      </p>
      <div className="an-airec-act">
        <Link href="/timeline" className="b go">
          View timeline
        </Link>
        <Link href="/alerts" className="b">
          Alert rules
        </Link>
      </div>
    </article>
  );
}

function HarActivityMetricsCard({
  daily,
  realtimeCount,
}: {
  daily: HarAnalyticsDailyApi | null;
  realtimeCount: number;
}) {
  const hasData = daily?.hasData;
  const maxHour = Math.max(1, ...(daily?.hourlyCounts?.map((h) => h.count) ?? [1]));

  return (
    <article className="an-card col-span-full">
      <div className="an-sech">
        <span className="ico">
          <Icon name="precision_manufacturing" size={16} />
        </span>
        <h2>HAR activity (live video)</h2>
      </div>
      {!hasData ? (
        <p className="text-body-sm text-outline">
          No HAR logs yet today — open Live with playback on to populate metrics.
        </p>
      ) : (
        <div className="grid gap-4 sm:grid-cols-3">
          <div className="rounded-lg border border-outline-variant/50 bg-surface-container-low p-4">
            <p className="font-label text-[10px] uppercase text-outline">Inferences today</p>
            <p className="font-mono text-2xl font-bold text-on-surface">{daily?.totalInferences ?? 0}</p>
            <p className="mt-1 text-body-sm text-outline">{realtimeCount} in last 30 min</p>
          </div>
          <div className="rounded-lg border border-outline-variant/50 bg-surface-container-low p-4">
            <p className="font-label text-[10px] uppercase text-outline">Non-assembly rate</p>
            <p className="font-mono text-2xl font-bold text-on-surface">
              {daily?.nonAssemblyRatePct ?? 0}%
            </p>
            <p className="mt-1 text-body-sm text-outline">
              Assemble system {daily?.assembleSharePct ?? 0}%
            </p>
          </div>
          <div className="rounded-lg border border-outline-variant/50 bg-surface-container-low p-4">
            <p className="font-label text-[10px] uppercase text-outline">Top deviations</p>
            <ul className="mt-2 space-y-1 text-body-sm text-on-surface">
              {(daily?.topDeviations ?? []).slice(0, 3).map((d) => (
                <li key={d.label} className="flex justify-between gap-2">
                  <span className="truncate">{d.label}</span>
                  <span className="font-mono text-outline">{d.count}</span>
                </li>
              ))}
            </ul>
          </div>
          <div className="col-span-full">
            <p className="mb-2 font-label text-[10px] uppercase text-outline">Inferences by hour</p>
            <div className="flex h-16 items-end gap-0.5">
              {(daily?.hourlyCounts ?? []).map((h) => (
                <div
                  key={h.hour}
                  className="flex-1 rounded-t bg-primary/70"
                  style={{ height: `${Math.max(4, (h.count / maxHour) * 100)}%` }}
                  title={`${h.hour}:00 — ${h.count}`}
                />
              ))}
            </div>
          </div>
        </div>
      )}
    </article>
  );
}

function HarPlant360Card({ plant }: { plant: HarPlantAnalyticsApi | null }) {
  if (!plant) return null;

  const maxHour = Math.max(1, ...(plant.hourlyCounts?.map((h) => h.count) ?? [1]));

  return (
    <article className="an-ac an-har-360">
      <div className="an-sech">
        <span className="ico">
          <Icon name="all_inclusive" size={16} />
        </span>
        <h2>Action analytics · 360° plant view</h2>
        <span className="ml-auto text-label-sm text-outline">
          {plant.cameraCount} HAR feed{plant.cameraCount === 1 ? "" : "s"}
        </span>
      </div>
      {!plant.hasData ? (
        <p className="text-body-sm text-outline px-4 pb-4">
          No HAR action logs today — run Live playback on cam-har feeds to populate the 360° dashboard.
        </p>
      ) : (
        <>
          <div className="an-har-360-grid">
            <div className="an-har-stat">
              <p className="k">Productivity score</p>
              <p className="v">{plant.productivityScore}%</p>
              <p className="s">{plant.assembleSharePct}% {plant.primaryActionLabel}</p>
            </div>
            <div className="an-har-stat">
              <p className="k">Inferences today</p>
              <p className="v">{plant.totalInferences}</p>
              <p className="s">{plant.nonAssemblyRatePct}% non-assembly</p>
            </div>
            <div className="an-har-stat">
              <p className="k">Est. action cost</p>
              <p className="v">${plant.actionCostUsd.toLocaleString()}</p>
              <p className="s">{plant.actionDowntimeMinutes} min rework est.</p>
            </div>
          </div>
          <div className="an-har-cams">
            <p className="an-har-cams-title">Per-camera productivity</p>
            <ul>
              {plant.byCamera.map((cam) => (
                <li key={cam.cameraId}>
                  <span className="nm">{cam.name}</span>
                  <span className="sc">{cam.productivityScore}%</span>
                  <span className="act">{cam.topAction ?? "—"}</span>
                </li>
              ))}
            </ul>
          </div>
          <div className="px-4 pb-4">
            <p className="mb-2 font-label text-[10px] uppercase text-outline">Plant inferences by hour</p>
            <div className="flex h-12 items-end gap-0.5">
              {plant.hourlyCounts.map((h) => (
                <div
                  key={h.hour}
                  className="flex-1 rounded-t bg-secondary/80"
                  style={{ height: `${Math.max(4, (h.count / maxHour) * 100)}%` }}
                  title={`${h.hour}:00 — ${h.count}`}
                />
              ))}
            </div>
          </div>
        </>
      )}
    </article>
  );
}

function ActionParetoCard({ items }: { items: { label: string; count: number; pct: number }[] }) {
  const max = items[0]?.count ?? 1;
  return (
    <article className="an-ac an-pareto-card">
      <CardHeader label="Pareto — HAR action deviations" kpiId="har_action_pareto" />
      <div className="an-pareto-rows">
        {items.map((item, idx) => (
          <div key={item.label} className="an-prow">
            <span className="rk">{idx + 1}</span>
            <div>
              <div className="nm">{item.label}</div>
              <div className="track">
                <i style={{ width: `${(item.count / max) * 100}%` }} />
              </div>
            </div>
            <div className="nums">
              <span className="c">{item.count}</span> <span className="p">{item.pct}%</span>
            </div>
          </div>
        ))}
      </div>
    </article>
  );
}

function SeverityTagsCard({
  tags,
  actionCostUsd,
}: {
  tags: { critical: number; warning: number; info: number };
  actionCostUsd?: number;
}) {
  return (
    <article className="an-severity-card">
      <div className="an-severity-h">
        <Icon name="sell" size={15} className="text-outline" />
        <span>Severity tags · today</span>
      </div>
      <div className="an-severity-tags">
        <span className="an-tag crit">
          critical <b>{tags.critical}</b>
        </span>
        <span className="an-tag warn">
          warning <b>{tags.warning}</b>
        </span>
        <span className="an-tag info">
          info <b>{tags.info}</b>
        </span>
      </div>
      {actionCostUsd != null && actionCostUsd > 0 ? (
        <p className="an-severity-cost">
          Action deviation cost estimate: <b>${actionCostUsd.toLocaleString()}</b>
        </p>
      ) : null}
    </article>
  );
}

function buildAttentionFeed(
  queue: TimelineEventApi[],
  bottlenecks: NonNullable<AnalyticsInsightsApi["bottlenecks"]>,
): AttentionItem[] {
  const seen = new Set<string>();
  const items: AttentionItem[] = [];

  for (const e of queue.slice(0, 5)) {
    seen.add(e.id);
    items.push({
      id: e.id,
      title: e.title,
      description: e.description ?? "Open critical event requires acknowledgment.",
      meta: `${e.time} · ${e.cameraId ?? "floor"} · OPEN CRITICAL`,
      critical: true,
    });
  }

  for (const b of bottlenecks) {
    if (seen.has(b.id) || items.length >= 6) continue;
    seen.add(b.id);
    items.push({
      id: b.id,
      title: b.title,
      description: b.description,
      meta: `${b.severity.toUpperCase()} · bottleneck signal`,
      critical: b.critical,
    });
  }

  return items;
}
