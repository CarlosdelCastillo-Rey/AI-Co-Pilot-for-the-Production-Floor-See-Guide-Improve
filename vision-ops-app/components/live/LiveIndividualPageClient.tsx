"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { AppShell } from "@/components/layout/AppShell";
import { CameraHarChat } from "@/components/live/CameraHarChat";
import { CameraStream } from "@/components/live/CameraStream";
import {
  HarBenchControls,
  useHarBenchBootstrap,
} from "@/components/live/HarBenchControls";
import { VideoFeedControls } from "@/components/live/VideoFeedControls";
import {
  HAR_MODEL_COLORS,
  useHarLiveState,
} from "@/components/live/useHarLiveState";
import { Icon } from "@/components/ui/Icon";
import {
  HAR_BENCH_CAMERA_ID,
  harBenchStreamUrl,
  setHarLivePlayback,
  type HarBenchConfig,
  type HarModelId,
} from "@/lib/api";
import { cn } from "@/lib/cn";

const DEFAULT_CONFIG: HarBenchConfig = {
  infer_every: 16,
  buffer_frames: 32,
  stream_fps: 12,
  show_heatmap: true,
  show_yolo_boxes: true,
  top_k: 5,
  ingest_logs: true,
};

export function LiveIndividualPageClient() {
  const { snapshot, loading } = useHarBenchBootstrap();
  const [modelId, setModelId] = useState<HarModelId>("dinov2-puro");
  const [videoName, setVideoName] = useState("");
  const [config, setConfig] = useState<HarBenchConfig>(DEFAULT_CONFIG);
  const [playing, setPlaying] = useState(false);

  useEffect(() => {
    void setHarLivePlayback(HAR_BENCH_CAMERA_ID, false);
  }, []);
  const [showOverlays, setShowOverlays] = useState(true);

  useEffect(() => {
    if (!snapshot?.state) return;
    if (snapshot.state.model_id) setModelId(snapshot.state.model_id as HarModelId);
    if (snapshot.state.video) setVideoName(snapshot.state.video);
    if (snapshot.state.config) setConfig(snapshot.state.config);
  }, [snapshot]);

  const { inferring, prediction, error, logs, sessionId, modelId: liveModel, videoUrl } =
    useHarLiveState(HAR_BENCH_CAMERA_ID, null, playing);

  const videoRef = useRef<HTMLVideoElement>(null);
  const accent = HAR_MODEL_COLORS[liveModel ?? modelId] ?? "#81C784";
  const pred = prediction;
  const topK = pred?.top_k ?? [];
  const streamUrl = harBenchStreamUrl();
  const displayVideoUrl = videoUrl ?? snapshot?.state?.videoUrl;
  const showMjpeg = showOverlays && Boolean(snapshot?.enabled);

  useEffect(() => {
    void setHarLivePlayback(HAR_BENCH_CAMERA_ID, playing);
  }, [playing]);

  useEffect(() => {
    const v = videoRef.current;
    if (!v || !displayVideoUrl || showMjpeg) return;
    if (playing) void v.play().catch(() => setPlaying(false));
    else v.pause();
  }, [playing, displayVideoUrl, showMjpeg]);

  const togglePlayPause = useCallback(() => {
    setPlaying((p) => !p);
  }, []);

  const benchState = snapshot?.state;
  const lastInferMs = benchState?.last_infer_ms;
  const backend = benchState?.backend;
  const device = benchState?.device;

  return (
    <AppShell fullBleed>
      <div className="min-h-[calc(100vh-4rem)] overflow-y-auto p-6 lg:p-8">
        <div className="mx-auto max-w-[1500px] space-y-6">
          <header>
            <h2 className="font-headline text-headline-lg text-on-surface">HAR Model Lab</h2>
            <p className="mt-1 max-w-3xl text-body-md text-outline">
              Single-camera sandbox — swap models and mock videos, tune inference hyperparameters
              live, and compare predictions with chat and logs.
            </p>
          </header>

          {loading ? (
            <div className="h-[480px] animate-pulse rounded-card bg-surface-container-low" />
          ) : !snapshot?.enabled ? (
            <div className="rounded-card border border-outline-variant/60 bg-surface-container-low p-8 text-center">
              <Icon name="science" className="mx-auto text-outline" size={48} />
              <p className="mt-4 text-body-md text-on-surface">HAR bench is not running</p>
              <p className="mt-1 text-body-sm text-outline">
                Start vision-ops-backend with HAR_ENABLED and HAR_LIVE_ENABLED.
              </p>
            </div>
          ) : (
            <div className="grid grid-cols-1 gap-6 xl:grid-cols-[minmax(0,1fr)_minmax(320px,400px)]">
              <div className="space-y-4">
                <article className="overflow-hidden rounded-card border border-outline-variant/60 bg-surface-container-lowest shadow-sm">
                  <div className="flex flex-wrap items-center justify-between gap-2 border-b border-outline-variant/50 px-4 py-3">
                    <div>
                      <h3 className="text-body-md font-semibold text-on-surface">
                        {snapshot.models.find((m) => m.model_id === modelId)?.label ?? modelId}
                      </h3>
                      <p className="truncate text-body-sm text-outline" title={videoName}>
                        {videoName || "—"}
                      </p>
                    </div>
                    <div className="flex flex-wrap items-center gap-2">
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
                    </div>
                  </div>

                  <div className="relative aspect-video w-full overflow-hidden bg-on-surface-variant/10">
                    {showMjpeg ? (
                      <CameraStream
                        streamUrl={streamUrl}
                        alt="HAR bench"
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

                  <div className="border-t border-outline-variant/50 bg-surface-container-low px-4 py-3">
                    <p className="font-label text-[10px] font-bold uppercase tracking-wider text-outline">
                      Inference log
                    </p>
                    <div className="mt-2 h-52 overflow-hidden rounded-lg border border-outline-variant/40 bg-surface-container-lowest sm:h-60">
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
              </div>

              <div className="flex min-h-0 flex-col gap-4">
                <HarBenchControls
                  modelId={modelId}
                  videoName={videoName}
                  config={config}
                  models={snapshot.models}
                  videos={snapshot.videos}
                  onModelChange={setModelId}
                  onVideoChange={setVideoName}
                  onConfigChange={setConfig}
                />

                <div className="rounded-lg border border-outline-variant/50 bg-surface-container-low p-4">
                  <p className="font-label text-[10px] font-bold uppercase tracking-wider text-outline">
                    Current prediction
                  </p>
                  {!playing && (
                    <p className="mt-2 text-body-sm text-outline">Paused — no new inference</p>
                  )}
                  {playing && inferring && !pred && (
                    <p className="mt-2 text-body-sm text-outline">Analyzing window…</p>
                  )}
                  {error && <p className="mt-2 text-body-sm text-error">{error}</p>}
                  {pred ? (
                    <>
                      <p className="mt-2 font-mono text-lg font-bold text-on-surface">
                        {pred.label}{" "}
                        <span style={{ color: accent }}>{Math.round(pred.confidence * 100)}%</span>
                      </p>
                      <div className="mt-2 h-2 overflow-hidden rounded-full bg-outline-variant/30">
                        <div
                          className="h-full rounded-full transition-all duration-300"
                          style={{
                            width: `${pred.confidence * 100}%`,
                            backgroundColor: accent,
                          }}
                        />
                      </div>
                      {topK.length > 0 && (
                        <ul className="mt-3 max-h-36 space-y-2 overflow-y-auto">
                          {topK.map((item) => (
                            <li key={item.label}>
                              <div className="mb-0.5 flex justify-between font-label text-[10px] text-outline">
                                <span className="truncate pr-2">{item.label}</span>
                                <span className="shrink-0 font-mono">
                                  {(item.prob * 100).toFixed(1)}%
                                </span>
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
                  {(backend || lastInferMs != null) && (
                    <p className="mt-3 border-t border-outline-variant/30 pt-2 font-mono text-[10px] text-outline">
                      {backend}
                      {device ? ` · ${device}` : ""}
                      {lastInferMs != null ? ` · ${lastInferMs} ms` : ""}
                    </p>
                  )}
                </div>

                <div className="min-h-[260px] flex-1">
                  <CameraHarChat
                    cameraId={HAR_BENCH_CAMERA_ID}
                    sessionId={sessionId}
                    accent={accent}
                    embedded
                  />
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </AppShell>
  );
}
