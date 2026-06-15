"use client";

import { useEffect, useState } from "react";
import { fetchHarLiveCamera } from "@/lib/api";
import { cn } from "@/lib/cn";

const MODEL_COLORS: Record<string, string> = {
  "v2-vjepa": "#FFB74D",
  "v2-dinov2": "#4FC3F7",
};

type TopKItem = { label: string; prob: number };
type HarPrediction = {
  label: string;
  confidence: number;
  top_k?: TopKItem[];
};

type LiveState = {
  inferring?: boolean;
  prediction?: HarPrediction | null;
  error?: string | null;
  model_id?: string;
};

export function HarLivePredictionPanel({
  cameraId,
  modelId,
  initialPrediction,
}: {
  cameraId: string;
  modelId?: string;
  initialPrediction?: HarPrediction | null;
}) {
  const [live, setLive] = useState<LiveState>({
    prediction: initialPrediction ?? null,
    inferring: !initialPrediction,
  });

  useEffect(() => {
    let cancelled = false;
    const poll = async () => {
      const state = await fetchHarLiveCamera(cameraId);
      if (!cancelled && state) {
        setLive({
          inferring: Boolean(state.inferring),
          prediction: (state.prediction as HarPrediction | null) ?? null,
          error: state.error ?? null,
          model_id: state.model_id,
        });
      }
    };
    void poll();
    const id = setInterval(() => void poll(), 1500);
    return () => {
      cancelled = true;
      clearInterval(id);
    };
  }, [cameraId]);

  const mid = modelId ?? live.model_id ?? "";
  const accent = MODEL_COLORS[mid] ?? "#81C784";
  const pred = live.prediction;
  const topK = pred?.top_k ?? [];

  return (
    <div className="pointer-events-none absolute bottom-12 right-3 z-10 w-[min(100%,220px)] rounded-lg border border-white/10 bg-on-surface/80 p-2.5 shadow-lg backdrop-blur-md">
      {live.inferring && !pred && (
        <p className="font-label text-[10px] text-white/70">Analizando…</p>
      )}
      {live.error && (
        <p className="font-label text-[10px] text-error">{live.error}</p>
      )}
      {pred && (
        <>
          <p className="font-label text-[9px] font-bold uppercase tracking-wider text-white/60">
            Actividad
          </p>
          <p className="mt-0.5 truncate font-mono text-sm font-bold" style={{ color: accent }}>
            {pred.label}{" "}
            <span className="text-white">{Math.round(pred.confidence * 100)}%</span>
          </p>
          <div className="mt-2 h-2 overflow-hidden rounded-full bg-white/10">
            <div
              className="h-full rounded-full transition-all duration-300"
              style={{ width: `${pred.confidence * 100}%`, backgroundColor: accent }}
            />
          </div>
          {topK.length > 0 && (
            <ul className="mt-2 space-y-1 border-t border-white/10 pt-2">
              {topK.map((item) => (
                <li key={item.label} className="flex items-center gap-2">
                  <div className="h-1 flex-1 overflow-hidden rounded-full bg-white/10">
                    <div
                      className={cn("h-full rounded-full")}
                      style={{
                        width: `${item.prob * 100}%`,
                        backgroundColor: accent,
                        opacity: item.label === pred.label ? 1 : 0.45,
                      }}
                    />
                  </div>
                  <span className="w-8 shrink-0 text-right font-mono text-[9px] text-white/80">
                    {(item.prob * 100).toFixed(0)}%
                  </span>
                  <span className="min-w-0 flex-1 truncate font-label text-[9px] text-white/70">
                    {item.label}
                  </span>
                </li>
              ))}
            </ul>
          )}
        </>
      )}
    </div>
  );
}
