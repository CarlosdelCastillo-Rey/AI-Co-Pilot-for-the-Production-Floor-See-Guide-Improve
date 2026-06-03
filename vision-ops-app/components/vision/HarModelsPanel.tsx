"use client";

import { useCallback, useEffect, useState } from "react";
import { Button } from "@/components/ui/Button";
import {
  fetchHarModels,
  fetchHarRuns,
  fetchHarStatus,
  runAllHarProbes,
  runHarProbe,
  type HarModelId,
  type HarModelInfo,
  type HarRunSummary,
  type HarStatus,
} from "@/lib/api";

const HAR_MODEL_ORDER: HarModelId[] = [
  "dinov2-puro",
  "dinov2-mcjepa",
  "vjepa2-puro",
  "vjepa2-mcjepa-frozen",
  "vjepa2-mcjepa-partial",
];

export function HarModelsPanel() {
  const [registry, setRegistry] = useState<HarModelInfo[]>([]);
  const [status, setStatus] = useState<HarStatus | null>(null);
  const [loading, setLoading] = useState<string | null>(null);
  const [message, setMessage] = useState<string | null>(null);
  const [torchOk, setTorchOk] = useState(false);
  const [ckptDirOk, setCkptDirOk] = useState(false);
  const [recentRuns, setRecentRuns] = useState<HarRunSummary[]>([]);

  const refresh = useCallback(async () => {
    const [modelsRes, statusRes, runsRes] = await Promise.all([
      fetchHarModels(),
      fetchHarStatus(),
      fetchHarRuns(8),
    ]);
    if (modelsRes) {
      setRegistry(modelsRes.models);
      setTorchOk(modelsRes.torch_available);
      setCkptDirOk(modelsRes.checkpoint_dir_exists);
    }
    setStatus(statusRes);
    setRecentRuns(runsRes?.runs ?? []);
  }, []);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  async function probeOne(modelId: HarModelId) {
    setLoading(modelId);
    setMessage(null);
    const result = await runHarProbe(modelId);
    setLoading(null);
    setMessage(result.message);
    if (result.ok) await refresh();
  }

  async function probeAll() {
    setLoading("all");
    setMessage("Starting batch probe…");
    const result = await runAllHarProbes({
      onProgress: (msg) => setMessage(msg),
    });
    setLoading(null);
    setMessage(result.message);
    if (result.ok) await refresh();
  }

  const shared = status?.shared_clip;

  return (
    <section className="rounded-lg border border-outline-variant bg-surface-container-lowest p-lg">
      <h3 className="text-label-md text-on-surface">HAR — Avance 4 activity models</h3>
      <p className="mt-1 text-body-sm text-outline">
        Five classifiers, each on a <strong className="text-on-surface">different mock video</strong>. Probe results are
        stored per mock camera in the backend registry.
      </p>

      {message ? (
        <p className="mt-md rounded-lg border border-outline-variant bg-surface-container-low px-md py-sm text-body-sm text-on-surface">
          {message}
        </p>
      ) : null}

      <div className="mt-md flex flex-wrap gap-2 text-body-sm text-outline">
        <span>torch: {torchOk ? "yes" : "no (uv sync --extra har)"}</span>
        <span>·</span>
        <span>checkpoints: {ckptDirOk ? "found" : "missing"}</span>
        {shared ? (
          <>
            <span>·</span>
            <span>
              clip: {shared.resolved_path ?? shared.source} ({shared.frame_count_sampled} frames)
            </span>
          </>
        ) : null}
      </div>

      <div className="mt-md flex flex-wrap gap-2">
        <Button variant="primary" disabled={loading !== null} onClick={() => void probeAll()}>
            {loading === "all" ? "Running all…" : "Run all 5 (random mock videos)"}
        </Button>
      </div>

      {recentRuns.length > 0 ? (
        <div className="mt-md rounded-lg border border-outline-variant/60 bg-surface-container-low px-md py-sm">
          <p className="text-label-sm text-outline">Recent runs (SQLite)</p>
          <ul className="mt-2 space-y-1 text-body-sm text-on-surface">
            {recentRuns.map((run) => (
              <li key={run.id}>
                <span className="font-mono text-label-sm">{run.id}</span> — {run.status} —{" "}
                {run.clipSource?.slice(0, 48)}
                {run.createdAt ? ` · ${new Date(run.createdAt).toLocaleString()}` : ""}
              </li>
            ))}
          </ul>
        </div>
      ) : null}

      <div className="mt-lg overflow-x-auto">
        <table className="w-full min-w-[32rem] border-collapse text-left text-body-sm">
          <thead>
            <tr className="border-b border-outline-variant text-label-sm text-outline">
              <th className="py-2 pr-4">Model</th>
              <th className="py-2 pr-4">Camera</th>
              <th className="py-2 pr-4">Ready</th>
              <th className="py-2 pr-4">Prediction</th>
              <th className="py-2">Action</th>
            </tr>
          </thead>
          <tbody>
            {HAR_MODEL_ORDER.map((modelId) => {
              const reg = registry.find((m) => m.model_id === modelId);
              const st = status?.models?.[modelId];
              const pred = st?.prediction;
              return (
                <tr key={modelId} className="border-b border-outline-variant/50">
                  <td className="py-2 pr-4 text-on-surface">{reg?.label ?? modelId}</td>
                  <td className="py-2 pr-4 text-outline">{reg?.camera_id ?? "—"}</td>
                  <td className="py-2 pr-4">
                    {reg?.ready ? (
                      <span className="text-primary">yes</span>
                    ) : (
                      <span className="text-outline" title={reg?.reason}>
                        no
                      </span>
                    )}
                  </td>
                  <td className="py-2 pr-4 text-on-surface">
                    {pred ? `${pred.label} (${Math.round(pred.confidence * 100)}%)` : "—"}
                  </td>
                  <td className="py-2">
                    <Button
                      variant="ghost"
                      className="!px-2 !py-1 text-body-sm"
                      disabled={loading !== null}
                      onClick={() => void probeOne(modelId)}
                    >
                      {loading === modelId ? "…" : "Run"}
                    </Button>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </section>
  );
}
