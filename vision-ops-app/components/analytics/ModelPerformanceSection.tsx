"use client";

import { useCallback, useEffect, useState } from "react";
import Link from "next/link";
import { DATA_RESET_EVENT } from "@/components/layout/Sidebar";
import { Icon } from "@/components/ui/Icon";
import {
  fetchHarModelPerformance,
  resolveHarPreviewUrl,
  type HarActivityLogApi,
  type HarModelPerformanceApi,
  type HarModelPerformanceGroup,
} from "@/lib/api";
import { cn } from "@/lib/cn";

const TABS = [
  { id: "model", label: "By model" },
  { id: "hyper", label: "Hyperparameters" },
  { id: "combo", label: "Model × video" },
  { id: "logs", label: "Inference logs" },
] as const;

type TabId = (typeof TABS)[number]["id"];

type LogFilter = {
  label: string;
  modelId?: string;
  hyperparamKey?: string;
  comboKey?: string;
};

const MODEL_COLORS: Record<string, string> = {
  "v2-vjepa-powermean7": "#A5D6A7",
  "v2-vjepa": "#FFB74D",
  "v2-dinov2": "#4FC3F7",
};

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
  const preview = resolveHarPreviewUrl(group.previewUrl || group.snapshotUrl);
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
            <img src={preview} alt="" className="h-full w-full object-cover" />
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
            <th className="px-3 py-2">Source</th>
          </tr>
        </thead>
        <tbody>
          {logs.map((log) => {
            const previewSrc = resolveHarPreviewUrl(log.previewUrl || log.snapshotUrl);
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
                  {log.videoName ?? "—"}
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

export function AiModelSummaryStrip({ data }: { data: HarModelPerformanceApi | null }) {
  if (!data?.hasData || data.byModel.length === 0) {
    return (
      <article className="an-ac an-ai-strip">
        <div className="an-sech">
          <span className="ico">
            <Icon name="psychology" size={16} />
          </span>
          <h2>AI model performance</h2>
        </div>
        <p className="px-4 pb-4 text-body-sm text-outline">
          No HAR inference logs yet — run mock videos on{" "}
          <Link href="/live" className="text-primary hover:underline">
            Live Streams
          </Link>{" "}
          to populate model metrics.
        </p>
      </article>
    );
  }

  return (
    <article className="an-ac an-ai-strip">
      <div className="an-sech">
        <span className="ico">
          <Icon name="psychology" size={16} />
        </span>
        <h2>AI model performance · {data.date}</h2>
        <span className="ml-auto text-label-sm text-outline">{data.totalLogs} inferences logged</span>
      </div>
      <div className="an-ai-model-grid">
        {data.byModel.map((m) => (
          <div key={m.modelId} className="an-ai-model-card">
            <p className="k" style={{ color: MODEL_COLORS[m.modelId ?? ""] ?? undefined }}>
              {m.modelLabel ?? m.modelId}
            </p>
            <p className="v">{conf(m.avgConfidence)}</p>
            <p className="s">avg confidence · {m.totalInferences} runs</p>
            <p className="s">
              Primary {pct(m.primaryActionRatePct)} · {m.avgInferMs != null ? `${m.avgInferMs} ms` : "—"} infer
            </p>
          </div>
        ))}
      </div>
    </article>
  );
}

type ModelPerformanceSectionProps = {
  date: string;
  source: string;
  onDateChange?: (d: string) => void;
  onSourceChange?: (s: string) => void;
};

export function ModelPerformanceSection({
  date,
  source,
  onDateChange,
  onSourceChange,
}: ModelPerformanceSectionProps) {
  const [tab, setTab] = useState<TabId>("model");
  const [data, setData] = useState<HarModelPerformanceApi | null>(null);
  const [loading, setLoading] = useState(true);
  const [logFilter, setLogFilter] = useState<LogFilter | null>(null);
  const [logEntries, setLogEntries] = useState<HarActivityLogApi[]>([]);
  const [logEntriesTotal, setLogEntriesTotal] = useState(0);
  const [logsLoading, setLogsLoading] = useState(false);

  const load = useCallback(async () => {
    setLoading(true);
    const res = await fetchHarModelPerformance({ date, source: source || undefined });
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
    <section className="an-model-section">
      <div className="an-sech">
        <span className="ico">
          <Icon name="monitoring" size={16} />
        </span>
        <h2>Model analysis</h2>
        <div className="ml-auto flex flex-wrap items-center gap-2">
          <input
            type="date"
            value={date}
            onChange={(e) => onDateChange?.(e.target.value)}
            className="rounded-md border border-outline-variant/60 bg-surface-container-lowest px-2 py-1 text-body-sm"
          />
          <select
            value={source}
            onChange={(e) => onSourceChange?.(e.target.value)}
            className="rounded-md border border-outline-variant/60 bg-surface-container-lowest px-2 py-1 text-body-sm"
          >
            <option value="">All sources</option>
            <option value="live">Live</option>
            <option value="bench">Bench</option>
            <option value="probe">Probe</option>
          </select>
          <Link href="/live" className="text-body-sm text-primary hover:underline">
            Open Live
          </Link>
        </div>
      </div>

      <div className="an-model-tabs">
        {TABS.map((t) => (
          <button
            key={t.id}
            type="button"
            onClick={() => {
              setTab(t.id);
              if (t.id === "logs" && !logFilter) void loadLogs(null);
            }}
            className={cn(tab === t.id && "on")}
          >
            {t.label}
          </button>
        ))}
      </div>

      {loading ? (
        <div className="an-model-loading" />
      ) : !data?.hasData ? (
        <p className="px-4 py-6 text-body-sm text-outline">
          No inference logs for {date}. Start playback on Live to record model runs.
        </p>
      ) : (
        <>
          {tab === "model" && (
            <div className="an-model-grid">
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
            <div className="an-model-grid">
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
            <div className="an-model-stack">
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
            <div className="space-y-3 px-4 pb-4">
              {logFilter ? (
                <div className="flex flex-wrap items-center justify-between gap-2 rounded-lg border border-primary/30 bg-primary/5 px-4 py-3">
                  <p className="text-body-sm text-on-surface">
                    <strong>{logEntries.length}</strong>
                    {logEntriesTotal > logEntries.length ? ` of ${logEntriesTotal}` : ""} · {logFilter.label}
                  </p>
                  <button
                    type="button"
                    onClick={() => {
                      setLogFilter(null);
                      void loadLogs(null);
                    }}
                    className="rounded-md border border-outline-variant/60 px-3 py-1.5 text-label-sm text-primary"
                  >
                    Clear filter
                  </button>
                </div>
              ) : null}
              {logsLoading ? (
                <div className="an-model-loading" />
              ) : logEntries.length === 0 ? (
                <p className="text-body-sm text-outline">No log entries match this filter.</p>
              ) : (
                <HarLogsTable logs={logEntries} />
              )}
            </div>
          )}
        </>
      )}
    </section>
  );
}
