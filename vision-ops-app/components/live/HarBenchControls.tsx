"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import {
  fetchHarBench,
  patchHarBenchConfig,
  resetHarBenchSession,
  setHarBenchModel,
  setHarBenchVideo,
  type HarBenchConfig,
  type HarModelId,
} from "@/lib/api";
import { cn } from "@/lib/cn";

const MODEL_ORDER: HarModelId[] = [
  "dinov2-puro",
  "dinov2-mcjepa",
  "vjepa2-puro",
  "vjepa2-mcjepa-frozen",
  "vjepa2-mcjepa-partial",
];

type Props = {
  modelId: string;
  videoName: string;
  config: HarBenchConfig;
  models: { model_id: HarModelId; label: string; ready: boolean }[];
  videos: { name: string; url: string }[];
  onModelChange: (id: HarModelId) => void;
  onVideoChange: (name: string) => void;
  onConfigChange: (cfg: HarBenchConfig) => void;
  busy?: boolean;
};

function SliderRow({
  label,
  hint,
  value,
  min,
  max,
  step = 1,
  onChange,
  onCommit,
}: {
  label: string;
  hint?: string;
  value: number;
  min: number;
  max: number;
  step?: number;
  onChange: (v: number) => void;
  onCommit: (v: number) => void;
}) {
  return (
    <div>
      <div className="mb-1 flex items-baseline justify-between gap-2">
        <label className="font-label text-[10px] font-bold uppercase tracking-wider text-outline">
          {label}
        </label>
        <span className="font-mono text-body-sm text-on-surface">{value}</span>
      </div>
      {hint ? <p className="mb-1.5 text-[10px] text-outline">{hint}</p> : null}
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
        onMouseUp={(e) => onCommit(Number((e.target as HTMLInputElement).value))}
        onTouchEnd={(e) => onCommit(Number((e.target as HTMLInputElement).value))}
        className="h-1.5 w-full cursor-pointer accent-primary"
      />
    </div>
  );
}

export function HarBenchControls({
  modelId,
  videoName,
  config,
  models,
  videos,
  onModelChange,
  onVideoChange,
  onConfigChange,
  busy = false,
}: Props) {
  const [localConfig, setLocalConfig] = useState(config);
  const [switching, setSwitching] = useState<string | null>(null);
  const debounceRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  useEffect(() => {
    setLocalConfig(config);
  }, [config]);

  const commitConfig = useCallback(
    async (patch: Partial<HarBenchConfig>) => {
      const next = { ...localConfig, ...patch };
      setLocalConfig(next);
      const saved = await patchHarBenchConfig(patch);
      if (saved) onConfigChange(saved);
    },
    [localConfig, onConfigChange],
  );

  const debouncedCommit = useCallback(
    (key: keyof HarBenchConfig, value: number) => {
      const next = { ...localConfig, [key]: value };
      setLocalConfig(next);
      if (debounceRef.current) clearTimeout(debounceRef.current);
      debounceRef.current = setTimeout(() => {
        void patchHarBenchConfig({ [key]: value }).then((saved) => {
          if (saved) onConfigChange(saved);
        });
      }, 350);
    },
    [localConfig, onConfigChange],
  );

  const handleModel = async (id: HarModelId) => {
    setSwitching(id);
    const ok = await setHarBenchModel(id);
    setSwitching(null);
    if (ok) onModelChange(id);
  };

  const handleVideo = async (name: string) => {
    setSwitching(name);
    const ok = await setHarBenchVideo(name);
    setSwitching(null);
    if (ok) onVideoChange(name);
  };

  const handleReset = async () => {
    setSwitching("reset");
    await resetHarBenchSession();
    setSwitching(null);
  };

  const disabled = busy || switching !== null;

  return (
    <section className="space-y-4 rounded-lg border border-outline-variant/60 bg-surface-container-low p-4">
      <div>
        <h3 className="font-label text-[10px] font-bold uppercase tracking-wider text-outline">
          Model
        </h3>
        <div className="mt-2 flex flex-col gap-1.5">
          {MODEL_ORDER.map((id) => {
            const meta = models.find((m) => m.model_id === id);
            const active = modelId === id;
            return (
              <button
                key={id}
                type="button"
                disabled={disabled || !meta?.ready}
                title={meta?.ready ? meta.label : "Checkpoint not ready"}
                onClick={() => void handleModel(id)}
                className={cn(
                  "rounded-md border px-3 py-2 text-left text-body-sm transition-colors",
                  active
                    ? "border-primary bg-primary/10 font-semibold text-primary"
                    : "border-outline-variant/50 text-on-surface hover:border-primary/50",
                  !meta?.ready && "cursor-not-allowed opacity-40",
                )}
              >
                {meta?.label ?? id}
                {switching === id ? " …" : null}
              </button>
            );
          })}
        </div>
      </div>

      <div>
        <h3 className="font-label text-[10px] font-bold uppercase tracking-wider text-outline">
          Mock video
        </h3>
        <select
          value={videoName}
          disabled={disabled}
          onChange={(e) => void handleVideo(e.target.value)}
          className="mt-2 w-full rounded-md border border-outline-variant/60 bg-surface-container-lowest px-3 py-2 text-body-sm text-on-surface"
        >
          {videos.map((v) => (
            <option key={v.name} value={v.name}>
              {v.name}
            </option>
          ))}
        </select>
      </div>

      <div className="space-y-4 border-t border-outline-variant/40 pt-4">
        <p className="font-label text-[10px] font-bold uppercase tracking-wider text-outline">
          Live hyperparameters
        </p>

        <SliderRow
          label="Infer every N frames"
          hint="Lower = more frequent predictions (more GPU load)"
          value={localConfig.infer_every}
          min={1}
          max={60}
          onChange={(v) => debouncedCommit("infer_every", v)}
          onCommit={(v) => void commitConfig({ infer_every: v })}
        />

        <SliderRow
          label="Sliding window (frames)"
          hint="Temporal context length for the classifier"
          value={localConfig.buffer_frames}
          min={8}
          max={96}
          step={4}
          onChange={(v) => debouncedCommit("buffer_frames", v)}
          onCommit={(v) => void commitConfig({ buffer_frames: v })}
        />

        <SliderRow
          label="Playback FPS cap"
          hint="Max decode rate for the mock clip"
          value={localConfig.stream_fps}
          min={1}
          max={30}
          onChange={(v) => debouncedCommit("stream_fps", v)}
          onCommit={(v) => void commitConfig({ stream_fps: v })}
        />

        <SliderRow
          label="Top-K classes"
          hint="How many ranked predictions to return"
          value={localConfig.top_k}
          min={1}
          max={10}
          onChange={(v) => debouncedCommit("top_k", v)}
          onCommit={(v) => void commitConfig({ top_k: v })}
        />

        <SliderRow
          label="Action dwell (windows)"
          hint="Consecutive inferences before label changes (per track)"
          value={localConfig.dwell_windows ?? 2}
          min={1}
          max={6}
          onChange={(v) => debouncedCommit("dwell_windows", v)}
          onCommit={(v) => void commitConfig({ dwell_windows: v })}
        />

        <div className="flex flex-wrap gap-2">
          {(
            [
              ["per_person_mode", "Per-person HAR"],
              ["show_heatmap", "Motion heatmap"],
              ["show_yolo_boxes", "YOLO person boxes"],
              ["ingest_logs", "Persist to SQLite"],
            ] as const
          ).map(([key, label]) => (
            <button
              key={key}
              type="button"
              disabled={disabled}
              onClick={() => void commitConfig({ [key]: !localConfig[key] })}
              className={cn(
                "rounded-md border px-2.5 py-1 font-label text-[10px]",
                localConfig[key]
                  ? "border-primary bg-primary/10 text-primary"
                  : "border-outline-variant text-outline hover:border-primary/40",
              )}
            >
              {label}: {localConfig[key] ? "on" : "off"}
            </button>
          ))}
        </div>
      </div>

      <button
        type="button"
        disabled={disabled}
        onClick={() => void handleReset()}
        className="w-full rounded-md border border-outline-variant/60 py-2 text-body-sm text-on-surface hover:border-primary hover:text-primary"
      >
        {switching === "reset" ? "Resetting…" : "Reset session & buffer"}
      </button>
    </section>
  );
}

export function useHarBenchBootstrap() {
  const [snapshot, setSnapshot] = useState<Awaited<ReturnType<typeof fetchHarBench>>>(null);
  const [loading, setLoading] = useState(true);

  const refresh = useCallback(async () => {
    const data = await fetchHarBench();
    setSnapshot(data);
    setLoading(false);
    return data;
  }, []);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  return { snapshot, loading, refresh };
}
