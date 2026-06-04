"use client";

import { useCallback, useEffect, useState } from "react";
import Link from "next/link";
import { AppShell } from "@/components/layout/AppShell";
import { DATA_RESET_EVENT } from "@/components/layout/Sidebar";
import { Icon } from "@/components/ui/Icon";
import {
  fetchHarModelPerformance,
  type HarActivityLogApi,
  type HarModelPerformanceApi,
  type HarModelPerformanceGroup,
} from "@/lib/api";
import { cn } from "@/lib/cn";

const TABS = [
  { id: "model", label: "By model" },
  { id: "hyper", label: "By hyperparameters" },
  { id: "combo", label: "Model × video × params" },
  { id: "logs", label: "Recent logs" },
] as const;

type TabId = (typeof TABS)[number]["id"];

type LogFilter = {
  label: string;
  modelId?: string;
  hyperparamKey?: string;
  comboKey?: string;
};

const MODEL_COLORS: Record<string, string> = {
  "dinov2-puro": "#4FC3F7",
  "dinov2-mcjepa": "#81C784",
  "vjepa2-puro": "#FFB74D",
  "vjepa2-mcjepa-frozen": "#CE93D8",
  "vjepa2-mcjepa-partial": "#F06292",
};

function todayIso(): string {
  return new Date().toISOString().slice(0, 10);
}

function pct(v: number | null | undefined): string {
  if (v == null) return "—";
  return `${v.toFixed(1)}%`;
}

function conf(v: number | null | undefined): string {
  if (v == null) return "—";
  return `${Math.round(v * 100)}%`;
}

function MetricCell({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <p className="font-label text-[10px] uppercase tracking-wide text-outline">{label}</p>
      <p className="font-mono text-body-sm text-on-surface">{value}</p>
    </div>
  );
}

function GroupCard({
  title,
  subtitle,
  accent,
  group,
  showVideo,
  showHyper,
  onSelect,
}: {
  title: string;
  subtitle?: string;
  accent?: string;
  group: HarModelPerformanceGroup;
  showVideo?: boolean;
  showHyper?: boolean;
  onSelect?: () => void;
}) {
  const preview = group.previewUrl || group.clipUrl;
  const Wrapper = onSelect ? "button" : "article";
  return (
    <Wrapper
      type={onSelect ? "button" : undefined}
      onClick={onSelect}
      className={cn(
        "w-full rounded-lg border border-outline-variant/60 bg-surface-container-lowest p-4 text-left",
        onSelect &&
          "cursor-pointer transition hover:border-primary/40 hover:shadow-[0_4px_12px_rgba(12,15,19,0.06)] focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-primary",
      )}
    >
      <div className="flex gap-4">
        {preview ? (
          <span className="relative block h-20 w-28 shrink-0 overflow-hidden rounded-md bg-on-surface-variant/10">
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img
              src={preview.startsWith("/api/") ? `/vision-api${preview}` : preview}
              alt=""
              className="h-full w-full object-cover"
            />
          </span>
        ) : (
          <div className="flex h-20 w-28 shrink-0 items-center justify-center rounded-md bg-on-surface-variant/10">
            <Icon name="videocam" className="text-outline" size={28} />
          </div>
        )}
        <div className="min-w-0 flex-1">
          <div className="flex items-start justify-between gap-2">
            <h3 className="truncate font-semibold text-on-surface" style={{ color: accent ?? undefined }}>
              {title}
            </h3>
            {onSelect ? (
              <span className="shrink-0 font-label text-[10px] uppercase tracking-wide text-primary">
                View logs
              </span>
            ) : null}
          </div>
          {subtitle ? <p className="truncate text-body-sm text-outline">{subtitle}</p> : null}
          {showVideo && group.videoName ? (
            <p className="mt-0.5 truncate text-[11px] text-outline" title={group.videoName}>
              Video: {group.videoName}
            </p>
          ) : null}
          {showHyper && group.hyperparamLabel ? (
            <p className="mt-0.5 text-[11px] text-outline">{group.hyperparamLabel}</p>
          ) : null}
          <div className="mt-3 grid grid-cols-2 gap-3 sm:grid-cols-4">
            <MetricCell label="Inferences" value={String(group.totalInferences)} />
            <MetricCell label="Avg conf" value={conf(group.avgConfidence)} />
            <MetricCell label="Primary rate" value={pct(group.primaryActionRatePct)} />
            <MetricCell label="Avg infer" value={group.avgInferMs != null ? `${group.avgInferMs} ms` : "—"} />
          </div>
          {group.topLabel ? (
            <p className="mt-2 text-body-sm text-on-surface">
              Top action: <strong>{group.topLabel}</strong> ({pct(group.topLabelPct)})
            </p>
          ) : null}
          {group.byLabel && Object.keys(group.byLabel).length > 0 ? (
            <div className="mt-2 flex flex-wrap gap-1">
              {Object.entries(group.byLabel)
                .sort((a, b) => b[1] - a[1])
                .slice(0, 4)
                .map(([label, count]) => (
                  <span
                    key={label}
                    className="rounded-full bg-surface-container-high px-2 py-0.5 font-label text-[10px] text-outline"
                  >
                    {label} · {count}
                  </span>
                ))}
            </div>
          ) : null}
        </div>
      </div>
    </Wrapper>
  );
}

function HarLogsTable({ logs }: { logs: HarActivityLogApi[] }) {
  return (
    <div className="overflow-x-auto rounded-lg border border-outline-variant/60">
      <table className="w-full min-w-[56rem] border-collapse text-left text-body-sm">
        <thead>
          <tr className="border-b border-outline-variant bg-surface-container-low text-label-sm text-outline">
            <th className="px-3 py-2">Time</th>
            <th className="px-3 py-2">Preview</th>
            <th className="px-3 py-2">Model</th>
            <th className="px-3 py-2">Action</th>
            <th className="px-3 py-2">Conf</th>
            <th className="px-3 py-2">Video</th>
            <th className="px-3 py-2">Hyperparams</th>
            <th className="px-3 py-2">Source</th>
          </tr>
        </thead>
        <tbody>
          {logs.map((log) => {
            const preview = log.previewUrl || log.snapshotUrl || log.clipUrl;
            const previewSrc = preview?.startsWith("/api/") ? `/vision-api${preview}` : preview;
            return (
              <tr key={log.id} className="border-b border-outline-variant/40">
                <td className="whitespace-nowrap px-3 py-2 font-mono text-[11px] text-outline">
                  {log.occurredAt ? new Date(log.occurredAt).toLocaleTimeString() : "—"}
                </td>
                <td className="px-3 py-2">
                  {previewSrc ? (
                    <a href={previewSrc} target="_blank" rel="noopener noreferrer">
                      {/* eslint-disable-next-line @next/next/no-img-element */}
                      <img src={previewSrc} alt="" className="h-10 w-14 rounded object-cover" />
                    </a>
                  ) : (
                    "—"
                  )}
                </td>
                <td className="px-3 py-2">
                  <span
                    className="font-mono text-[11px]"
                    style={{ color: MODEL_COLORS[log.modelId] ?? undefined }}
                  >
                    {log.modelLabel ?? log.modelId}
                  </span>
                </td>
                <td className="px-3 py-2 text-on-surface">{log.predictedLabel ?? "—"}</td>
                <td className="px-3 py-2 font-mono">
                  {log.confidence != null ? `${Math.round(log.confidence * 100)}%` : "—"}
                </td>
                <td className="max-w-[10rem] truncate px-3 py-2 text-[11px] text-outline">
                  {log.videoName ? (
                    log.clipUrl ? (
                      <a href={log.clipUrl} className="text-primary hover:underline">
                        {log.videoName}
                      </a>
                    ) : (
                      log.videoName
                    )
                  ) : (
                    "—"
                  )}
                </td>
                <td className="max-w-[12rem] truncate px-3 py-2 text-[10px] text-outline">
                  {log.hyperparamLabel ?? "—"}
                </td>
                <td className="px-3 py-2 text-[11px] text-outline">{log.source ?? "—"}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

export function HarAnalysisPageClient() {
  const [tab, setTab] = useState<TabId>("model");
  const [date, setDate] = useState(todayIso());
  const [source, setSource] = useState("");
  const [data, setData] = useState<HarModelPerformanceApi | null>(null);
  const [loading, setLoading] = useState(true);
  const [logFilter, setLogFilter] = useState<LogFilter | null>(null);
  const [logEntries, setLogEntries] = useState<HarActivityLogApi[]>([]);
  const [logEntriesTotal, setLogEntriesTotal] = useState(0);
  const [logsLoading, setLogsLoading] = useState(false);

  const load = useCallback(async () => {
    setLoading(true);
    const res = await fetchHarModelPerformance({
      date,
      source: source || undefined,
    });
    setData(res);
    setLoading(false);
  }, [date, source]);

  const loadLogs = useCallback(
    async (filter: LogFilter | null) => {
      setLogsLoading(true);
      const res = await fetchHarModelPerformance({
        date,
        source: source || undefined,
        modelId: filter?.modelId,
        hyperparamKey: filter?.hyperparamKey,
        comboKey: filter?.comboKey,
        logsLimit: 500,
      });
      setLogEntries(res?.recentLogs ?? []);
      setLogEntriesTotal(filter ? (res?.filteredCount ?? res?.totalLogs ?? 0) : (res?.totalLogs ?? 0));
      setLogsLoading(false);
    },
    [date, source],
  );

  const openLogFilter = useCallback(
    async (filter: LogFilter) => {
      setLogFilter(filter);
      setTab("logs");
      await loadLogs(filter);
    },
    [loadLogs],
  );

  const clearLogFilter = useCallback(async () => {
    setLogFilter(null);
    await loadLogs(null);
  }, [loadLogs]);

  const openLogsTab = useCallback(async () => {
    setTab("logs");
    if (!logFilter) await loadLogs(null);
  }, [loadLogs, logFilter]);

  useEffect(() => {
    void load();
    setLogFilter(null);
    const id = setInterval(() => void load(), 20_000);
    const onReset = () => void load();
    window.addEventListener(DATA_RESET_EVENT, onReset);
    return () => {
      clearInterval(id);
      window.removeEventListener(DATA_RESET_EVENT, onReset);
    };
  }, [load]);

  useEffect(() => {
    if (tab === "logs" && !logFilter) void loadLogs(null);
  }, [date, source, tab, logFilter, loadLogs]);

  return (
    <AppShell fullBleed>
      <div className="min-h-[calc(100vh-4rem)] overflow-y-auto p-6 lg:p-8">
        <div className="mx-auto max-w-[1400px] space-y-6">
          <header className="flex flex-col gap-4 lg:flex-row lg:items-start lg:justify-between">
            <div>
              <h2 className="font-headline text-headline-lg text-on-surface">Model Analysis</h2>
              <p className="mt-1 max-w-2xl text-body-md text-outline">
                Compare HAR model performance by checkpoint, hyperparameter preset, mock video, and
                inference log — with preview links and confidence breakdowns.
              </p>
            </div>
            <div className="flex flex-wrap items-center gap-2">
              <input
                type="date"
                value={date}
                onChange={(e) => setDate(e.target.value)}
                className="rounded-md border border-outline-variant/60 bg-surface-container-lowest px-3 py-2 text-body-sm"
              />
              <select
                value={source}
                onChange={(e) => setSource(e.target.value)}
                className="rounded-md border border-outline-variant/60 bg-surface-container-lowest px-3 py-2 text-body-sm"
              >
                <option value="">All sources</option>
                <option value="live">Live</option>
                <option value="bench">Bench (Model Lab)</option>
                <option value="probe">Probe</option>
              </select>
              <Link
                href="/live-individual"
                className="rounded-md border border-primary/40 px-3 py-2 text-body-sm text-primary hover:bg-primary/10"
              >
                Open Model Lab
              </Link>
            </div>
          </header>

          <div className="flex flex-wrap gap-2 border-b border-outline-variant/50 pb-2">
            {TABS.map((t) => (
              <button
                key={t.id}
                type="button"
                onClick={() => {
                  if (t.id === "logs") void openLogsTab();
                  else setTab(t.id);
                }}
                className={cn(
                  "rounded-md px-3 py-1.5 font-label text-label-sm",
                  tab === t.id
                    ? "bg-primary/15 font-semibold text-primary"
                    : "text-outline hover:bg-surface-container-high",
                )}
              >
                {t.label}
              </button>
            ))}
          </div>

          {loading ? (
            <div className="h-64 animate-pulse rounded-card bg-surface-container-low" />
          ) : !data?.hasData ? (
            <div className="rounded-card border border-outline-variant/60 bg-surface-container-low p-10 text-center">
              <Icon name="analytics" className="mx-auto text-outline" size={48} />
              <p className="mt-4 text-body-md text-on-surface">No HAR logs for {date}</p>
              <p className="mt-1 text-body-sm text-outline">
                Run inference on <Link href="/live-individual" className="text-primary">Model Lab</Link> or{" "}
                <Link href="/live" className="text-primary">Live Streams</Link> — logs persist with model,
                video, hyperparameters, and preview snapshots.
              </p>
            </div>
          ) : (
            <>
              <p className="text-body-sm text-outline">
                {data.totalLogs} log entries on {data.date}
                {source ? ` · source: ${source}` : ""}
              </p>

              {tab === "model" && (
                <div className="grid gap-4 lg:grid-cols-2">
                  {data.byModel.map((g) => (
                    <GroupCard
                      key={g.modelId}
                      title={g.modelLabel ?? g.modelId ?? "—"}
                      subtitle={g.modelId}
                      accent={MODEL_COLORS[g.modelId ?? ""]}
                      group={g}
                      onSelect={() =>
                        void openLogFilter({
                          label: g.modelLabel ?? g.modelId ?? "model",
                          modelId: g.modelId,
                        })
                      }
                    />
                  ))}
                </div>
              )}

              {tab === "hyper" && (
                <div className="grid gap-4 lg:grid-cols-2">
                  {data.byHyperparams.map((g) => (
                    <GroupCard
                      key={g.hyperparamKey}
                      title={g.hyperparamLabel ?? g.hyperparamKey ?? "—"}
                      subtitle={`${g.modelCount ?? 0} models · ${g.videoCount ?? 0} videos`}
                      group={g}
                      showHyper
                      onSelect={() =>
                        void openLogFilter({
                          label: g.hyperparamLabel ?? g.hyperparamKey ?? "preset",
                          hyperparamKey: g.hyperparamKey,
                        })
                      }
                    />
                  ))}
                </div>
              )}

              {tab === "combo" && (
                <div className="space-y-4">
                  {data.byCombo.map((g) => (
                    <GroupCard
                      key={g.comboKey ?? `${g.modelId}-${g.videoName}`}
                      title={g.modelLabel ?? g.modelId ?? "—"}
                      subtitle={[g.source, g.cameraId].filter(Boolean).join(" · ")}
                      accent={MODEL_COLORS[g.modelId ?? ""]}
                      group={g}
                      showVideo
                      showHyper
                      onSelect={() =>
                        void openLogFilter({
                          label: [g.modelLabel ?? g.modelId, g.videoName, g.hyperparamLabel]
                            .filter(Boolean)
                            .join(" · "),
                          modelId: g.modelId,
                          hyperparamKey: g.hyperparamKey,
                          comboKey: g.comboKey,
                        })
                      }
                    />
                  ))}
                </div>
              )}

              {tab === "logs" && (
                <div className="space-y-3">
                  {logFilter ? (
                    <div className="flex flex-wrap items-center justify-between gap-2 rounded-lg border border-primary/30 bg-primary/5 px-4 py-3">
                      <p className="text-body-sm text-on-surface">
                        Showing <strong>{logEntries.length}</strong>
                        {logEntriesTotal > logEntries.length
                          ? ` of ${logEntriesTotal}`
                          : ""}{" "}
                        entries for <strong>{logFilter.label}</strong>
                      </p>
                      <button
                        type="button"
                        onClick={() => void clearLogFilter()}
                        className="rounded-md border border-outline-variant/60 px-3 py-1.5 font-label text-label-sm text-primary hover:bg-primary/10"
                      >
                        Clear filter
                      </button>
                    </div>
                  ) : (
                    <p className="text-body-sm text-outline">
                      Showing {logEntries.length}
                      {logEntriesTotal > logEntries.length ? ` of ${logEntriesTotal}` : ""} log entries
                      on {data.date}. Click a model, preset, or combo card to drill down.
                    </p>
                  )}
                  {logsLoading ? (
                    <div className="h-48 animate-pulse rounded-lg bg-surface-container-low" />
                  ) : logEntries.length === 0 ? (
                    <p className="text-body-sm text-outline">No log entries match this filter.</p>
                  ) : (
                    <HarLogsTable logs={logEntries} />
                  )}
                </div>
              )}
            </>
          )}
        </div>
      </div>
    </AppShell>
  );
}
