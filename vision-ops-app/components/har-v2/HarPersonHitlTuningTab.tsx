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

      {/* ── Summary counters ── */}
      <HarV2Panel title="Dataset snapshot" className="lg:col-span-2">
        {err && <HarV2Message tone="error" className="mb-4">{err}</HarV2Message>}
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
