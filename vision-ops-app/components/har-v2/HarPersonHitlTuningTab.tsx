"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { HarV2Empty, HarV2Message, HarV2Panel } from "@/components/har-v2/har-v2-ui";
import {
  type RetrainActionLabel,
  type RetrainDatasetStats,
  type RetrainIdentity,
  type RetrainJobState,
  fetchRetrainDatasetStats,
  fetchRetrainStatus,
  triggerRetrain,
} from "@/lib/har-v2-api";
import { cn } from "@/lib/cn";

// ── Small helpers ─────────────────────────────────────────────────────────────

function Chip({ label, count, tone = "default" }: { label: string; count: number; tone?: "new" | "known" | "default" }) {
  const cls =
    tone === "new"
      ? "border-primary/40 bg-primary/5 text-primary"
      : tone === "known"
        ? "border-tertiary/40 bg-tertiary/5 text-tertiary"
        : "border-outline-variant/40 bg-surface-container-low text-on-surface-variant";
  return (
    <span className={`inline-flex items-center gap-1.5 rounded-full border px-2.5 py-1 font-label text-[11px] ${cls}`}>
      {label}
      <span className="rounded-full bg-black/10 px-1.5 py-0.5 text-[10px] font-semibold">{count}</span>
    </span>
  );
}

function StatCard({ label, value, sub }: { label: string; value: number | string; sub?: string }) {
  return (
    <div className="flex flex-col gap-0.5 rounded-lg border border-outline-variant/40 bg-surface-container-low px-4 py-3">
      <span className="font-label text-[10px] uppercase tracking-wide text-outline">{label}</span>
      <span className="font-headline text-title-lg text-on-surface">{value}</span>
      {sub && <span className="text-[11px] text-outline">{sub}</span>}
    </div>
  );
}

function StatusBadge({ status }: { status: RetrainJobState["status"] }) {
  const map: Record<string, { label: string; cls: string }> = {
    idle:    { label: "Idle",    cls: "bg-surface-container text-outline" },
    running: { label: "Running", cls: "bg-primary/10 text-primary animate-pulse" },
    done:    { label: "Done",    cls: "bg-tertiary/10 text-tertiary" },
    failed:  { label: "Failed",  cls: "bg-error/10 text-error" },
  };
  const { label, cls } = map[status] ?? map.idle;
  return (
    <span className={`inline-flex items-center rounded-full px-2.5 py-0.5 font-label text-[11px] ${cls}`}>
      {label}
    </span>
  );
}

function LogConsole({ lines }: { lines: string[] }) {
  const endRef = useRef<HTMLDivElement>(null);
  useEffect(() => { endRef.current?.scrollIntoView({ behavior: "smooth" }); }, [lines.length]);
  if (!lines.length) return null;
  return (
    <div className="mt-3 max-h-52 overflow-y-auto rounded-md border border-outline-variant/30 bg-surface-container-lowest p-3 font-mono text-[11px] text-on-surface-variant">
      {lines.map((l, i) => <div key={i} className="leading-5">{l}</div>)}
      <div ref={endRef} />
    </div>
  );
}

// ── Identity list ─────────────────────────────────────────────────────────────

function IdentityList({ identities }: { identities: RetrainIdentity[] }) {
  if (!identities.length) {
    return (
      <p className="text-[11px] text-outline">
        No persons registered yet. Tag tracks in the <strong>Sessions</strong> tab to build the Re-ID registry.
      </p>
    );
  }
  return (
    <div className="divide-y divide-outline-variant/20">
      {identities.map((p) => (
        <div key={p.global_person_id} className="flex items-center justify-between py-2">
          <div>
            <p className="font-label text-sm text-on-surface">{p.display_name}</p>
            <p className="text-[10px] text-outline font-mono">{p.global_person_id.slice(0, 14)}…</p>
          </div>
          <div className="text-right">
            <p className="text-[11px] text-on-surface-variant">{p.n_appearances} appearances</p>
            {p.last_seen_at && (
              <p className="text-[10px] text-outline">
                {new Date(p.last_seen_at).toLocaleDateString()}
              </p>
            )}
          </div>
        </div>
      ))}
    </div>
  );
}

// ── Action label list ─────────────────────────────────────────────────────────

function ActionLabelSection({
  known,
  newActions,
  baseClasses,
}: {
  known: RetrainActionLabel[];
  newActions: RetrainActionLabel[];
  baseClasses: string[];
}) {
  return (
    <div className="space-y-4">
      {newActions.length > 0 && (
        <div>
          <p className="mb-2 font-label text-[10px] uppercase tracking-wide text-primary">
            New actions (will expand the model)
          </p>
          <div className="flex flex-wrap gap-2">
            {newActions.map((a) => (
              <Chip key={a.label} label={a.label} count={a.samples} tone="new" />
            ))}
          </div>
        </div>
      )}

      {known.length > 0 && (
        <div>
          <p className="mb-2 font-label text-[10px] uppercase tracking-wide text-tertiary">
            Known actions (reinforcing existing classes)
          </p>
          <div className="flex flex-wrap gap-2">
            {known.map((a) => (
              <Chip key={a.label} label={a.label} count={a.samples} tone="known" />
            ))}
          </div>
        </div>
      )}

      {known.length === 0 && newActions.length === 0 && (
        <p className="text-[11px] text-outline">
          No confirmed action labels yet. Open a session, pick a track, and set its correct action label.
        </p>
      )}

      <details className="group">
        <summary className="cursor-pointer text-[11px] text-outline hover:text-on-surface-variant">
          {baseClasses.length} base classes (InHARD)
        </summary>
        <div className="mt-2 flex flex-wrap gap-1.5">
          {baseClasses.map((c) => (
            <span key={c} className="rounded border border-outline-variant/30 px-2 py-0.5 text-[10px] text-outline">
              {c}
            </span>
          ))}
        </div>
      </details>
    </div>
  );
}

// ── Anomaly rules ─────────────────────────────────────────────────────────────

const STORAGE_KEY = "visionops_anomaly_rules";

type AnomalyRules = {
  idleStreakEnabled: boolean;
  idleStreakThreshold: number;
  lowConfEnabled: boolean;
  lowConfThreshold: number;
  pickingAnomalyEnabled: boolean;
  pickingRatioThreshold: number;
  amrProximityEnabled: boolean;
};

const DEFAULT_RULES: AnomalyRules = {
  idleStreakEnabled: true,
  idleStreakThreshold: 5,
  lowConfEnabled: true,
  lowConfThreshold: 30,
  pickingAnomalyEnabled: true,
  pickingRatioThreshold: 55,
  amrProximityEnabled: false,
};

function loadRules(): AnomalyRules {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    return raw ? { ...DEFAULT_RULES, ...(JSON.parse(raw) as Partial<AnomalyRules>) } : DEFAULT_RULES;
  } catch {
    return DEFAULT_RULES;
  }
}

function RuleToggle({
  label,
  description,
  enabled,
  onToggle,
  children,
}: {
  label: string;
  description: string;
  enabled: boolean;
  onToggle: () => void;
  children?: React.ReactNode;
}) {
  return (
    <div className={cn(
      "rounded-lg border p-4 transition-colors",
      enabled ? "border-primary/30 bg-primary/5" : "border-outline-variant/40 bg-surface-container-low",
    )}>
      <div className="flex items-start justify-between gap-3">
        <div className="min-w-0 flex-1">
          <p className={cn("font-label text-[11px] font-semibold", enabled ? "text-primary" : "text-on-surface")}>{label}</p>
          <p className="mt-0.5 text-[11px] text-outline">{description}</p>
        </div>
        <button
          type="button"
          onClick={onToggle}
          className={cn(
            "relative inline-flex h-5 w-9 shrink-0 rounded-full border-2 border-transparent transition-colors focus:outline-none",
            enabled ? "bg-primary" : "bg-outline-variant/50",
          )}
          aria-pressed={enabled}
        >
          <span
            className={cn(
              "inline-block h-4 w-4 transform rounded-full bg-white shadow transition-transform",
              enabled ? "translate-x-4" : "translate-x-0",
            )}
          />
        </button>
      </div>
      {enabled && children ? <div className="mt-3 border-t border-outline-variant/20 pt-3">{children}</div> : null}
    </div>
  );
}

function AnomalyRulesPanel() {
  const [rules, setRules] = useState<AnomalyRules>(DEFAULT_RULES);
  const [saved, setSaved] = useState(false);

  useEffect(() => {
    setRules(loadRules());
  }, []);

  function update(patch: Partial<AnomalyRules>) {
    setRules((prev) => {
      const next = { ...prev, ...patch };
      try { localStorage.setItem(STORAGE_KEY, JSON.stringify(next)); } catch { /* ignore */ }
      return next;
    });
    setSaved(true);
    setTimeout(() => setSaved(false), 1500);
  }

  return (
    <HarV2Panel title="Anomaly detection rules" className="lg:col-span-2">
      <p className="mb-4 text-body-sm text-on-surface-variant">
        Rule-based anomaly flags applied in the Registry tab. No model retraining needed — computed
        from existing inference logs. Toggle rules to match your floor&apos;s expected workflow.
      </p>
      {saved ? (
        <p className="mb-3 rounded-md border border-primary/25 bg-primary/5 px-3 py-1.5 text-[11px] text-primary">
          ✓ Rules saved to browser storage
        </p>
      ) : null}
      <div className="grid gap-3 sm:grid-cols-2">
        <RuleToggle
          label="Idle streak alert"
          description="Flag workers with consecutive 'No action' windows above threshold"
          enabled={rules.idleStreakEnabled}
          onToggle={() => update({ idleStreakEnabled: !rules.idleStreakEnabled })}
        >
          <label className="mb-1 block font-label text-[10px] uppercase tracking-wide text-outline">
            Threshold — {rules.idleStreakThreshold} consecutive windows
          </label>
          <input
            type="range" min={2} max={20} step={1}
            value={rules.idleStreakThreshold}
            onChange={(e) => update({ idleStreakThreshold: Number(e.target.value) })}
            className="w-full accent-primary"
          />
        </RuleToggle>

        <RuleToggle
          label="Low confidence burst"
          description="Flag when model confidence drops below threshold for multiple events"
          enabled={rules.lowConfEnabled}
          onToggle={() => update({ lowConfEnabled: !rules.lowConfEnabled })}
        >
          <label className="mb-1 block font-label text-[10px] uppercase tracking-wide text-outline">
            Confidence threshold — {rules.lowConfThreshold}%
          </label>
          <input
            type="range" min={10} max={50} step={5}
            value={rules.lowConfThreshold}
            onChange={(e) => update({ lowConfThreshold: Number(e.target.value) })}
            className="w-full accent-primary"
          />
        </RuleToggle>

        <RuleToggle
          label="Robo hormiga — theft pattern"
          description="Flag workers with unusually high picking-action rate, a behavioral proxy for systematic small theft"
          enabled={rules.pickingAnomalyEnabled}
          onToggle={() => update({ pickingAnomalyEnabled: !rules.pickingAnomalyEnabled })}
        >
          <label className="mb-1 block font-label text-[10px] uppercase tracking-wide text-outline">
            Picking ratio trigger — {rules.pickingRatioThreshold}% of tracked events
          </label>
          <input
            type="range" min={30} max={80} step={5}
            value={rules.pickingRatioThreshold}
            onChange={(e) => update({ pickingRatioThreshold: Number(e.target.value) })}
            className="w-full accent-primary"
          />
        </RuleToggle>

        <RuleToggle
          label="Robo hormiga — AMR proximity"
          description="Flag sessions where non-person YOLO detections appear within safety distance of workers (requires live detections_json in har_activity_logs)"
          enabled={rules.amrProximityEnabled}
          onToggle={() => update({ amrProximityEnabled: !rules.amrProximityEnabled })}
        >
          <p className="text-[11px] text-outline">
            AMR proximity flags are generated during live inference. Check{" "}
            <a href="/live" className="text-primary hover:underline">Live Streams</a> for real-time alerts.
            Historical flags are surfaced in the Timeline page.
          </p>
        </RuleToggle>
      </div>

      <div className="mt-4 rounded-md border border-outline-variant/30 bg-surface-container-low p-3">
        <p className="font-label text-[10px] uppercase tracking-wide text-outline">PPE detection status</p>
        <p className="mt-1 text-body-sm text-on-surface-variant">
          Helmet / vest / glove detection requires a dedicated YOLOv8-PPE checkpoint.
          The current <code className="font-mono text-[11px]">yolov8n.pt</code> detects persons only.
          Use the HITL custom labels below to manually tag &quot;PPE missing&quot; events and build a labeled dataset.
        </p>
      </div>
    </HarV2Panel>
  );
}

// ── Custom anomaly labels ─────────────────────────────────────────────────────

const SUGGESTED_ANOMALY_LABELS = ["Eating", "Phone use", "PPE missing", "Unauthorized handling"];

function CustomLabelManager({ stats }: { stats: RetrainDatasetStats | null }) {
  const allCustom = stats?.action_labels.new.map((a) => a.label) ?? [];
  const knownAll = new Set([
    ...(stats?.base_class_names ?? []),
    ...(stats?.action_labels.known.map((a) => a.label) ?? []),
  ]);

  return (
    <HarV2Panel title="Anomaly action labels">
      <p className="mb-3 text-body-sm text-on-surface-variant">
        Add new action labels here conceptually — then tag actual crops in the{" "}
        <strong>Sessions</strong> tab by selecting &quot;No — wrong action&quot; and typing the label.
        Once you have ≥10 samples for a label, retrain to activate detection.
      </p>

      <div className="mb-4">
        <p className="mb-2 font-label text-[10px] uppercase tracking-wide text-outline">Suggested anomaly labels</p>
        <div className="flex flex-wrap gap-2">
          {SUGGESTED_ANOMALY_LABELS.map((label) => {
            const active = allCustom.includes(label);
            const isKnown = knownAll.has(label);
            return (
              <span
                key={label}
                className={cn(
                  "rounded-full border px-3 py-1 font-label text-[11px]",
                  active
                    ? "border-primary/40 bg-primary/10 text-primary"
                    : isKnown
                      ? "border-tertiary/40 bg-tertiary/10 text-tertiary"
                      : "border-outline-variant/40 bg-surface-container-low text-on-surface-variant",
                )}
              >
                {label}
                {active ? " ✓" : isKnown ? " (base)" : ""}
              </span>
            );
          })}
        </div>
      </div>

      {allCustom.length > 0 ? (
        <div>
          <p className="mb-2 font-label text-[10px] uppercase tracking-wide text-outline">
            Custom labels in your dataset
          </p>
          <div className="flex flex-wrap gap-2">
            {(stats?.action_labels.new ?? []).map((a) => (
              <span
                key={a.label}
                className="inline-flex items-center gap-1.5 rounded-full border border-primary/40 bg-primary/5 px-2.5 py-1 font-label text-[11px] text-primary"
              >
                {a.label}
                <span className="rounded-full bg-primary/10 px-1.5 py-0.5 text-[10px] font-semibold">
                  {a.samples} samples
                </span>
                {a.samples < 10 ? (
                  <span className="text-[9px] text-outline">need {10 - a.samples} more</span>
                ) : (
                  <span className="text-[9px] text-primary">ready to train</span>
                )}
              </span>
            ))}
          </div>
        </div>
      ) : (
        <p className="text-[11px] text-outline">
          No custom labels yet. Go to Sessions → select a track → set &quot;No — wrong action&quot; → type an anomaly label to create your first sample.
        </p>
      )}
    </HarV2Panel>
  );
}

// ── Data quality ──────────────────────────────────────────────────────────────

function DataQualityPanel({ stats }: { stats: RetrainDatasetStats | null }) {
  if (!stats) return null;

  const allLabels = [
    ...stats.action_labels.known.map((a) => ({ label: a.label, count: a.samples, type: "known" as const })),
    ...stats.action_labels.new.map((a) => ({ label: a.label, count: a.samples, type: "new" as const })),
  ];
  const maxCount = Math.max(...allLabels.map((a) => a.count), 1);
  const coverageScore =
    allLabels.length > 0
      ? Math.round((allLabels.filter((a) => a.count >= 10).length / allLabels.length) * 100)
      : 0;

  return (
    <HarV2Panel title="Data quality" className="lg:col-span-2">
      <div className="mb-4 flex flex-wrap items-center gap-4 text-body-sm">
        <div className="rounded-md border border-outline-variant/40 bg-surface-container-low px-3 py-2">
          <p className="font-label text-[10px] uppercase tracking-wide text-outline">Coverage score</p>
          <p className={cn(
            "mt-0.5 font-mono text-xl font-semibold",
            coverageScore >= 80 ? "text-green-600" : coverageScore >= 50 ? "text-amber-600" : "text-red-600",
          )}>
            {coverageScore}%
          </p>
          <p className="mt-0.5 text-[10px] text-outline">labels with ≥10 samples</p>
        </div>
        <div className="rounded-md border border-outline-variant/40 bg-surface-container-low px-3 py-2">
          <p className="font-label text-[10px] uppercase tracking-wide text-outline">Total HITL samples</p>
          <p className="mt-0.5 font-mono text-xl font-semibold text-on-surface">{stats.total_hitl_samples}</p>
          <p className="mt-0.5 text-[10px] text-outline">{stats.hitl_human_labels} crops · {stats.hitl_track_labels} tracks</p>
        </div>
      </div>

      {allLabels.length > 0 ? (
        <div className="space-y-2">
          <p className="font-label text-[10px] uppercase tracking-wide text-outline">Class coverage</p>
          {allLabels.map((a) => {
            const isReady = a.count >= 10;
            const barColor = a.type === "new" ? "#9C27B0" : isReady ? "#4CAF50" : "#FF9800";
            return (
              <div key={a.label} className="grid grid-cols-[minmax(120px,1fr)_minmax(80px,140px)_auto] items-center gap-3 text-body-sm">
                <span className="truncate text-on-surface">{a.label}</span>
                <div className="h-2 overflow-hidden rounded-full bg-surface-container-high">
                  <div
                    className="h-full rounded-full transition-all"
                    style={{
                      width: `${Math.max((a.count / maxCount) * 100, 2)}%`,
                      backgroundColor: barColor,
                    }}
                  />
                </div>
                <div className="flex items-center gap-1.5">
                  <span className="font-mono text-[11px] text-on-surface">{a.count}</span>
                  {!isReady ? (
                    <span className="text-[10px] text-amber-600">need {10 - a.count} more</span>
                  ) : (
                    <span className="text-[10px] text-green-600">✓</span>
                  )}
                </div>
              </div>
            );
          })}
        </div>
      ) : (
        <p className="text-[11px] text-outline">No labeled data yet.</p>
      )}
    </HarV2Panel>
  );
}

// ── Main component ────────────────────────────────────────────────────────────

export function HarPersonHitlTuningTab() {
  const [stats, setStats] = useState<RetrainDatasetStats | null>(null);
  const [job, setJob] = useState<RetrainJobState | null>(null);
  const [backbone, setBackbone] = useState<"vjepa" | "dinov2">("vjepa");
  const [epochs, setEpochs] = useState(30);
  const [err, setErr] = useState("");
  const [launching, setLaunching] = useState(false);
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const loadStats = useCallback(() => {
    fetchRetrainDatasetStats().then(setStats).catch((e) => setErr(String(e)));
  }, []);

  const loadJob = useCallback(() => {
    fetchRetrainStatus().then(setJob).catch(() => {});
  }, []);

  const stopPoll = useCallback(() => {
    if (pollRef.current) { clearInterval(pollRef.current); pollRef.current = null; }
  }, []);

  const startPoll = useCallback(() => {
    stopPoll();
    pollRef.current = setInterval(() => {
      fetchRetrainStatus().then((s) => {
        setJob(s);
        if (s.status !== "running") { stopPoll(); loadStats(); }
      });
    }, 2000);
  }, [loadStats, stopPoll]);

  useEffect(() => {
    loadStats();
    loadJob();
    return stopPoll;
  }, [loadStats, loadJob, stopPoll]);

  useEffect(() => {
    if (job?.status === "running" && !pollRef.current) startPoll();
  }, [job?.status, startPoll]);

  const handleRetrain = async () => {
    setLaunching(true);
    setErr("");
    try {
      const res = await triggerRetrain(backbone, epochs);
      if (res.status === "already_running") {
        setErr("A retrain job is already running.");
      } else {
        await loadJob();
        startPoll();
      }
    } catch (e) {
      setErr(String(e));
    } finally {
      setLaunching(false);
    }
  };

  const isRunning = job?.status === "running";
  const hasHitl = (stats?.total_hitl_samples ?? 0) > 0;

  return (
    <div className="grid gap-6 lg:grid-cols-2">

      {/* ── Anomaly detection rules ── */}
      <AnomalyRulesPanel />

      {/* ── Custom anomaly label manager ── */}
      <CustomLabelManager stats={stats} />

      {/* ── Data quality ── */}
      <DataQualityPanel stats={stats} />

      {/* ── Summary counters ── */}
      <HarV2Panel title="Dataset snapshot" className="lg:col-span-2">
        {err && <div className="mb-4"><HarV2Message tone="error">{err}</HarV2Message></div>}
        {stats ? (
          <>
            <div className="grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-5">
              <StatCard label="Sessions" value={stats.sessions} />
              <StatCard label="Identities" value={stats.persons} sub="Re-ID registry" />
              <StatCard label="Crop labels" value={stats.hitl_human_labels} sub="HarHumanLabel" />
              <StatCard label="Track labels" value={stats.hitl_track_labels} sub="dominant action" />
              <StatCard
                label="HITL total"
                value={stats.total_hitl_samples}
                sub={hasHitl ? "ready to train" : "label tracks first"}
              />
            </div>
            <div className="mt-3 flex gap-4 text-[11px] text-outline">
              <span>V-JEPA2 base: <span className={stats.base_embeddings_available.vjepa ? "text-tertiary" : "text-error"}>{stats.base_embeddings_available.vjepa ? "✓" : "✗ missing"}</span></span>
              <span>DINOv2 base: <span className={stats.base_embeddings_available.dinov2 ? "text-tertiary" : "text-error"}>{stats.base_embeddings_available.dinov2 ? "✓" : "✗ missing"}</span></span>
            </div>
          </>
        ) : (
          <div className="grid grid-cols-5 gap-3">
            {Array.from({ length: 5 }).map((_, i) => (
              <div key={i} className="h-16 animate-pulse rounded-lg bg-surface-container-low" />
            ))}
          </div>
        )}
      </HarV2Panel>

      {/* ── Identities ── */}
      <HarV2Panel title="Identities in next retrain">
        <p className="mb-3 text-[11px] text-outline">
          Persons whose Re-ID embeddings are registered. Their names appear on <code>/live</code> overlays via <code>REID_AUTO_LINK</code>.
        </p>
        {stats ? (
          <IdentityList identities={stats.identities} />
        ) : (
          <div className="space-y-2">
            {[1,2,3].map(i => <div key={i} className="h-10 animate-pulse rounded bg-surface-container-low" />)}
          </div>
        )}
      </HarV2Panel>

      {/* ── Action labels ── */}
      <HarV2Panel title="Actions in next retrain">
        <p className="mb-3 text-[11px] text-outline">
          <span className="text-primary font-semibold">New actions</span> expand the model beyond the 12 InHARD classes.{" "}
          <span className="text-tertiary font-semibold">Known actions</span> reinforce existing classes with your data.
        </p>
        {stats ? (
          <ActionLabelSection
            known={stats.action_labels.known}
            newActions={stats.action_labels.new}
            baseClasses={stats.base_class_names}
          />
        ) : (
          <div className="h-24 animate-pulse rounded bg-surface-container-low" />
        )}
      </HarV2Panel>

      {/* ── Training controls ── */}
      <HarV2Panel title="Retrain MLP head">
        <p className="mb-4 text-body-sm text-on-surface-variant">
          Merges your HITL crops with the base InHARD embeddings. Only the MLP classifier
          is updated — the backbone stays frozen. Completes in seconds on CPU.
        </p>
        <div className="space-y-4">
          <div>
            <label className="mb-1 block font-label text-[10px] uppercase tracking-wide text-outline">Backbone</label>
            <div className="flex gap-2">
              {(["vjepa", "dinov2"] as const).map((b) => (
                <button key={b} type="button" disabled={isRunning}
                  onClick={() => setBackbone(b)}
                  className={`rounded-md border px-3 py-1.5 font-label text-[11px] transition-colors disabled:opacity-50 ${backbone === b ? "border-primary bg-primary/10 text-primary" : "border-outline-variant/60 text-on-surface-variant hover:border-outline"}`}>
                  {b === "vjepa" ? "V-JEPA 2" : "DINOv2"}
                </button>
              ))}
            </div>
          </div>

          <div>
            <label className="mb-1 block font-label text-[10px] uppercase tracking-wide text-outline">
              Epochs — {epochs}
            </label>
            <input type="range" min={5} max={100} step={5} value={epochs} disabled={isRunning}
              onChange={(e) => setEpochs(Number(e.target.value))}
              className="w-full accent-primary disabled:opacity-50" />
            <div className="flex justify-between text-[10px] text-outline">
              <span>5 (fast)</span><span>100 (thorough)</span>
            </div>
          </div>

          <button type="button" disabled={isRunning || launching || !hasHitl}
            onClick={handleRetrain}
            className="w-full rounded-lg bg-primary px-4 py-2.5 font-label text-sm text-on-primary shadow-sm transition-opacity hover:opacity-90 disabled:cursor-not-allowed disabled:opacity-40">
            {isRunning ? "Training…" : launching ? "Launching…" : "Retrain model"}
          </button>
          {!hasHitl && (
            <p className="text-center text-[11px] text-outline">
              Confirm at least one track label in Sessions or Review Queue first.
            </p>
          )}
        </div>
      </HarV2Panel>

      {/* ── Job status ── */}
      <HarV2Panel title="Training status">
        {job ? (
          <div className="space-y-3">
            <div className="flex items-center justify-between">
              <StatusBadge status={job.status} />
              {job.started_at && (
                <span className="text-[11px] text-outline">{new Date(job.started_at).toLocaleTimeString()}</span>
              )}
            </div>

            {job.error && <HarV2Message tone="error">{job.error}</HarV2Message>}

            {job.status === "done" && Object.keys(job.result).length > 0 && (
              <div className="grid grid-cols-2 gap-2 rounded-md bg-tertiary/5 p-3 text-[11px]">
                <div>
                  <span className="text-outline">Val accuracy</span>
                  <p className="font-mono font-semibold text-tertiary">
                    {((job.result.val_accuracy ?? 0) * 100).toFixed(1)}%
                  </p>
                </div>
                <div>
                  <span className="text-outline">Samples</span>
                  <p className="font-mono font-semibold">{job.result.n_samples ?? "—"}</p>
                </div>
                <div>
                  <span className="text-outline">Classes</span>
                  <p className="font-mono font-semibold">{job.result.n_classes ?? "—"}</p>
                </div>
                <div>
                  <span className="text-outline">HITL added</span>
                  <p className="font-mono font-semibold">{job.result.n_hitl_samples ?? 0}</p>
                </div>
                {job.result.class_names && (
                  <div className="col-span-2">
                    <span className="text-outline">Active classes after retrain</span>
                    <div className="mt-1 flex flex-wrap gap-1">
                      {(job.result.class_names as string[]).map((c) => (
                        <span key={c} className="rounded border border-outline-variant/30 px-1.5 py-0.5 font-mono text-[10px] text-on-surface-variant">{c}</span>
                      ))}
                    </div>
                  </div>
                )}
                {job.result.output_dir && (
                  <div className="col-span-2">
                    <span className="text-outline">Archive</span>
                    <p className="mt-0.5 font-mono text-[10px] text-on-surface-variant break-all">
                      {String(job.result.output_dir).replace(/.*outputs\/retrain\//, "outputs/retrain/")}
                    </p>
                  </div>
                )}
              </div>
            )}

            {job.status === "idle" && (
              <HarV2Empty>No retrain triggered this session.</HarV2Empty>
            )}

            <LogConsole lines={job.logs} />
          </div>
        ) : (
          <div className="h-32 animate-pulse rounded-md bg-surface-container-low" />
        )}
      </HarV2Panel>

    </div>
  );
}
