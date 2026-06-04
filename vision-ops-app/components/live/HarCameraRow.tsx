"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { CameraStream } from "@/components/live/CameraStream";
import { VideoFeedControls } from "@/components/live/VideoFeedControls";
import { CameraHarChat } from "@/components/live/CameraHarChat";
import {
  HAR_MODEL_COLORS,
  useHarLiveState,
  type HarPrediction,
} from "@/components/live/useHarLiveState";
import { Icon } from "@/components/ui/Icon";
import { setHarLivePlayback } from "@/lib/api";
import { useLivePlaybackSync } from "@/components/live/useLivePlaybackSync";
import type { CameraFeed } from "@/lib/types";
import { cn } from "@/lib/cn";

function predictionFromFeed(feed: CameraFeed): HarPrediction | null {
  const probe = feed.visionProbe as { prediction?: HarPrediction } | null | undefined;
  return probe?.prediction ?? null;
}

function modelIdFromFeed(feed: CameraFeed): string {
  return feed.inferenceModel?.replace(/_/g, "-") ?? "";
}

export function HarCameraRow({ feed }: { feed: CameraFeed }) {
  const cameraId = feed.backendCameraId ?? feed.id;
  const modelId = modelIdFromFeed(feed);
  const accent = HAR_MODEL_COLORS[modelId] ?? "#81C784";
  const initialPrediction = predictionFromFeed(feed);

  const [playing, setPlaying] = useState(false);
  const skipBackendSyncRef = useLivePlaybackSync(setPlaying);
  /** Server MJPEG with heatmap + boxes + label (notebook-style). */
  const [showOverlays, setShowOverlays] = useState(true);
  const { inferring, prediction, error, logs, video, videoUrl, sessionId } = useHarLiveState(
    cameraId,
    initialPrediction,
    playing,
  );

  const videoRef = useRef<HTMLVideoElement>(null);
  const displayVideoUrl = videoUrl ?? feed.videoUrl;
  const showMjpeg = showOverlays && feed.status === "live" && Boolean(feed.streamUrl);

  useEffect(() => {
    if (skipBackendSyncRef.current) {
      skipBackendSyncRef.current = false;
      return;
    }
    void setHarLivePlayback(cameraId, playing);
  }, [cameraId, playing, skipBackendSyncRef]);

  useEffect(() => {
    const v = videoRef.current;
    if (!v || !displayVideoUrl) return;
    if (playing) void v.play().catch(() => setPlaying(false));
    else v.pause();
  }, [playing, displayVideoUrl]);

  const togglePlayPause = useCallback(() => {
    setPlaying((p) => !p);
  }, []);

  const pred = prediction;
  const topK = pred?.top_k ?? [];
  const probe = feed.visionProbe as Record<string, unknown> | null | undefined;
  const badge = feed.modelBadge;

  return (
    <article className="max-w-full overflow-hidden rounded-card border border-outline-variant/60 bg-surface-container-lowest shadow-sm">
      <div className="flex flex-col border-b border-outline-variant/50 px-4 py-3 sm:flex-row sm:items-center sm:justify-between">
        <div>
          <h3 className="text-body-md font-semibold text-on-surface">{feed.name}</h3>
          <p className="text-body-sm text-outline">{feed.location}</p>
        </div>
        <div className="mt-2 flex flex-wrap items-center gap-2 sm:mt-0">
          {badge && (
            <span className="rounded-md bg-surface-container-high px-2 py-1 font-label text-[10px] font-medium uppercase text-on-surface">
              {badge.model} · {badge.task}
            </span>
          )}
          <span
            className={cn(
              "flex items-center gap-1.5 font-label text-[10px]",
              playing ? "text-success" : "text-outline",
            )}
          >
            <span
              className={cn(
                "h-1.5 w-1.5 rounded-full",
                playing ? "animate-pulse bg-success" : "bg-outline",
              )}
            />
            {playing ? "LIVE" : "PAUSED"}
          </span>
          {feed.streamUrl && (
            <button
              type="button"
              className={cn(
                "rounded border px-2 py-0.5 font-label text-[10px]",
                showOverlays
                  ? "border-primary bg-primary/10 text-primary"
                  : "border-outline-variant text-outline hover:border-primary",
              )}
              onClick={() => setShowOverlays((v) => !v)}
            >
              {showOverlays ? "Overlays on" : "Plain video"}
            </button>
          )}
        </div>
      </div>

      {/* Video + model summary + chat */}
      <div className="grid grid-cols-1 items-stretch gap-4 p-4 lg:grid-cols-[minmax(0,1.35fr)_minmax(300px,420px)]">
        <div className="relative aspect-video w-full max-h-[min(45vh,400px)] overflow-hidden rounded-lg bg-on-surface-variant/10 lg:max-h-none lg:min-h-[280px]">
          {showMjpeg && feed.streamUrl ? (
            <CameraStream
              streamUrl={feed.streamUrl}
              alt={feed.name}
              offlineMessage={feed.error}
              paused={!playing}
            />
          ) : displayVideoUrl ? (
            <video
              ref={videoRef}
              key={displayVideoUrl}
              src={displayVideoUrl}
              className="absolute inset-0 h-full w-full object-cover"
              autoPlay={playing}
              muted
              loop
              playsInline
              onPlay={() => setPlaying(true)}
              onPause={() => setPlaying(false)}
            />
          ) : (
            <div className="absolute inset-0 flex items-center justify-center">
              <Icon name="videocam_off" className="text-outline" size={48} />
            </div>
          )}
          <div className="absolute bottom-3 left-3 z-10">
            <VideoFeedControls playing={playing} onPlayPause={togglePlayPause} />
          </div>
        </div>

        <div className="flex min-h-0 flex-col gap-4 lg:min-h-[280px]">
          <div className="shrink-0 rounded-lg border border-outline-variant/50 bg-surface-container-low p-4">
          <p className="font-label text-[10px] font-bold uppercase tracking-wider text-outline">
            Model
          </p>
          <p className="mt-1 font-mono text-body-sm font-semibold" style={{ color: accent }}>
            {badge?.model ?? modelId}
          </p>
          {video && (
            <p className="mt-1 truncate text-body-sm text-outline" title={video}>
              Clip: {video}
            </p>
          )}
          {typeof probe?.backend === "string" && (
            <p className="mt-0.5 truncate text-[10px] text-outline">{probe.backend as string}</p>
          )}

          <div className="mt-4 border-t border-outline-variant/40 pt-4">
            <p className="font-label text-[10px] font-bold uppercase tracking-wider text-outline">
              Current prediction
            </p>
            {!playing && (
              <p className="mt-2 text-body-sm text-outline">Paused — no new inference</p>
            )}
            {playing && inferring && !pred && (
              <p className="mt-2 text-body-sm text-outline">Detecting activity on person…</p>
            )}
            {error && <p className="mt-2 text-body-sm text-error">{error}</p>}
            {pred ? (
              <>
                  <p className="mt-1 font-label text-[10px] uppercase tracking-wide text-outline">
                    Detected action
                  </p>
                  <p className="mt-1 font-mono text-lg font-bold text-on-surface">
                    {pred.label}{" "}
                    <span style={{ color: accent }}>{Math.round(pred.confidence * 100)}%</span>
                  </p>
                <div className="mt-2 h-2 overflow-hidden rounded-full bg-outline-variant/30">
                  <div
                    className="h-full rounded-full transition-all duration-300"
                    style={{ width: `${pred.confidence * 100}%`, backgroundColor: accent }}
                  />
                </div>
                {topK.length > 0 && (
                  <ul className="mt-3 max-h-32 space-y-2 overflow-y-auto">
                    {topK.map((item) => (
                      <li key={item.label}>
                        <div className="mb-0.5 flex justify-between font-label text-[10px] text-outline">
                          <span className="truncate pr-2">{item.label}</span>
                          <span className="shrink-0 font-mono">{(item.prob * 100).toFixed(1)}%</span>
                        </div>
                        <div className="h-1.5 overflow-hidden rounded-full bg-outline-variant/25">
                          <div
                            className="h-full rounded-full"
                            style={{
                              width: `${item.prob * 100}%`,
                              backgroundColor: accent,
                              opacity: item.label === pred.label ? 1 : 0.5,
                            }}
                          />
                        </div>
                      </li>
                    ))}
                  </ul>
                )}
              </>
            ) : playing && !inferring && !error ? (
              <p className="mt-2 text-body-sm text-outline">Waiting for first inference…</p>
            ) : null}
          </div>
          </div>

          <div className="min-h-[280px] min-w-0 flex-1">
            <CameraHarChat cameraId={cameraId} sessionId={sessionId} accent={accent} embedded />
          </div>
        </div>
      </div>

      {/* Inference log — full width under video, fixed tall scroll area */}
      <div className="border-t border-outline-variant/50 bg-surface-container-low px-4 py-3">
        <p className="font-label text-[10px] font-bold uppercase tracking-wider text-outline">
          Inference log
        </p>
        <div className="mt-2 h-56 overflow-hidden rounded-lg border border-outline-variant/40 bg-surface-container-lowest sm:h-64">
          <ul className="h-full space-y-1.5 overflow-y-auto overscroll-y-contain p-3 text-body-sm">
            {logs.length === 0 ? (
              <li className="text-body-sm text-outline">No log entries yet.</li>
            ) : (
              logs.map((entry) => (
                <li
                  key={entry.id}
                  className={cn(
                    "rounded px-2 py-1.5 font-mono text-[11px] leading-snug",
                    entry.kind === "prediction" && "bg-primary/10 text-on-surface",
                    entry.kind === "info" && "text-outline",
                    entry.kind === "pause" && "bg-surface-container-high text-on-surface",
                    entry.kind === "error" && "bg-error/10 text-error",
                  )}
                >
                  <span className="text-outline">{entry.at}</span> {entry.message}
                </li>
              ))
            )}
          </ul>
        </div>
      </div>
    </article>
  );
}
