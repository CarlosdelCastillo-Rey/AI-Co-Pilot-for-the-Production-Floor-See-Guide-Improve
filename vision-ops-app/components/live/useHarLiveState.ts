"use client";

import { useEffect, useRef, useState } from "react";
import { fetchHarActivityLogs, fetchHarLiveCamera, type HarLiveCameraState, type HarTrackPrediction } from "@/lib/api";

export type HarPrediction = {
  label: string;
  confidence: number;
  top_k?: { label: string; prob: number }[];
  track_id?: number;
};

export type HarLogEntry = {
  id: string;
  at: string;
  kind: "info" | "prediction" | "error" | "pause";
  message: string;
};

export const HAR_MODEL_COLORS: Record<string, string> = {
  "v2-vjepa": "#FFB74D",
  "v2-dinov2": "#4FC3F7",
};

function formatTime(d: Date): string {
  return d.toLocaleTimeString(undefined, { hour: "2-digit", minute: "2-digit", second: "2-digit" });
}

function formatTimeFromIso(iso: string | null): string {
  if (!iso) return formatTime(new Date());
  try {
    return formatTime(new Date(iso));
  } catch {
    return formatTime(new Date());
  }
}

function apiLogToEntry(row: {
  id: string;
  occurredAt: string | null;
  predictedLabel: string | null;
  confidence: number | null;
  actorName?: string | null;
}): HarLogEntry {
  const pct =
    row.confidence != null ? Math.round(row.confidence * 100) : 0;
  const who = row.actorName ? ` · ${row.actorName}` : "";
  return {
    id: row.id,
    at: formatTimeFromIso(row.occurredAt),
    kind: "prediction",
    message: `Action detected: ${row.predictedLabel ?? "—"} (${pct}%)${who}`,
  };
}

export function useHarLiveState(
  cameraId: string,
  initialPrediction?: HarPrediction | null,
  playing = true,
) {
  const [state, setState] = useState<{
    inferring: boolean;
    prediction: HarPrediction | null;
    error: string | null;
    modelId?: string;
    video?: string;
    videoUrl?: string;
    sessionId?: string;
    trackPredictions: HarTrackPrediction[];
    perPersonMode: boolean;
    personCount: number;
  }>({
    inferring: false,
    prediction: initialPrediction ?? null,
    error: null,
    trackPredictions: [],
    perPersonMode: true,
    personCount: 0,
  });
  const [logs, setLogs] = useState<HarLogEntry[]>(() => [
    {
      id: "boot",
      at: formatTime(new Date()),
      kind: "info",
      message: "Live inference stream connected",
    },
  ]);
  const lastPredRef = useRef<string | null>(null);
  const wasInferringRef = useRef(false);
  const lastErrorRef = useRef<string | null>(null);
  const wasPlayingRef = useRef(playing);
  const lastSessionRef = useRef<string | null>(null);
  const [sessionTracks, setSessionTracks] = useState<HarTrackPrediction[]>([]);

  const mergeSessionTracks = (incoming: HarTrackPrediction[]) => {
    if (!incoming.length) return;
    setSessionTracks((prev) => {
      const byId = new Map(prev.map((t) => [t.track_id, t]));
      for (const tr of incoming) {
        byId.set(tr.track_id, { ...byId.get(tr.track_id), ...tr });
      }
      return [...byId.values()].sort((a, b) => a.track_id - b.track_id);
    });
  };

  const pushLog = (entry: Omit<HarLogEntry, "id">) => {
    setLogs((prev) =>
      [
        { ...entry, id: `${Date.now()}-${Math.random().toString(36).slice(2, 7)}` },
        ...prev,
      ].slice(0, 80),
    );
  };

  useEffect(() => {
    void fetchHarActivityLogs(cameraId, 80).then((rows) => {
      if (!rows.length) return;
      const fromApi = rows.map(apiLogToEntry);
      setLogs((prev) => {
        const localOnly = prev.filter((e) => e.kind === "pause" || e.id === "boot");
        const merged = [...localOnly, ...fromApi];
        const seen = new Set<string>();
        const deduped: HarLogEntry[] = [];
        for (const e of merged) {
          const key = `${e.at}:${e.message}`;
          if (seen.has(key)) continue;
          seen.add(key);
          deduped.push(e);
        }
        return deduped.slice(0, 80);
      });
    });
  }, [cameraId]);

  useEffect(() => {
    if (playing === wasPlayingRef.current) return;
    wasPlayingRef.current = playing;
    pushLog({
      at: formatTime(new Date()),
      kind: "pause",
      message: playing
        ? "Playback resumed — recording session + per-person tracks"
        : "Playback paused — session finalized",
    });
    if (!playing) {
      setState((s) => ({ ...s, inferring: false }));
      wasInferringRef.current = false;
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [playing]);

  useEffect(() => {
    if (!playing) return;

    let cancelled = false;
    const poll = async () => {
      const raw: HarLiveCameraState | null = await fetchHarLiveCamera(cameraId);
      if (cancelled || !raw) return;

      const pred = (raw.prediction as HarPrediction | null) ?? null;
      const inferring = Boolean(raw.inferring);
      const predKey = pred ? `${pred.label}:${pred.confidence}` : null;

      setLogs((prev) => {
        const next = [...prev];
        const push = (entry: Omit<HarLogEntry, "id">) => {
          next.unshift({
            ...entry,
            id: `${Date.now()}-${Math.random().toString(36).slice(2, 7)}`,
          });
        };

        if (inferring && !wasInferringRef.current) {
          push({ at: formatTime(new Date()), kind: "info", message: "Analyzing window…" });
        }
        wasInferringRef.current = inferring;

        if (predKey && predKey !== lastPredRef.current && pred) {
          lastPredRef.current = predKey;
          const pct = Math.round(pred.confidence * 100);
          const who =
            raw.per_person_mode && pred.track_id != null ? ` · track #${pred.track_id}` : "";
          push({
            at: formatTime(new Date()),
            kind: "prediction",
            message: `Action detected: ${pred.label} (${pct}%)${who}`,
          });
        }

        if (raw.error && raw.error !== lastErrorRef.current) {
          lastErrorRef.current = raw.error;
          push({ at: formatTime(new Date()), kind: "error", message: raw.error });
        }

        return next.slice(0, 80);
      });

      setState({
        inferring,
        prediction: pred,
        error: raw.error ?? null,
        modelId: raw.model_id,
        video: raw.video,
        videoUrl: raw.videoUrl,
        sessionId: raw.session_id as string | undefined,
        trackPredictions: raw.track_predictions ?? [],
        perPersonMode: Boolean(raw.per_person_mode),
        personCount: raw.person_count ?? 0,
      });

      const sid = (raw.session_id as string | undefined) ?? null;
      if (sid !== lastSessionRef.current) {
        lastSessionRef.current = sid;
        setSessionTracks([]);
      }
      mergeSessionTracks(raw.track_predictions ?? []);
    };
    void poll();
    const id = setInterval(() => void poll(), 1500);
    return () => {
      cancelled = true;
      clearInterval(id);
    };
  }, [cameraId, playing]);

  useEffect(() => {
    lastSessionRef.current = null;
    setSessionTracks([]);
    lastPredRef.current = null;
  }, [cameraId]);

  return { ...state, logs, pushLog, sessionTracks };
}
