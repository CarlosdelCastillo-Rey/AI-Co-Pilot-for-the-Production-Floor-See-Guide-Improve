"use client";

import Link from "next/link";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { AppShell } from "@/components/layout/AppShell";
import { DATA_RESET_EVENT } from "@/components/layout/Sidebar";
import { AddCameraModal } from "@/components/live/AddCameraModal";
import { CameraHarChat } from "@/components/live/CameraHarChat";
import { LiveStatsBar } from "@/components/live/LiveStatsBar";
import { MockVideoQuadGrid, type LiveViewLayout } from "@/components/live/MockVideoQuadGrid";
import { LiveCollapsiblePanel, LiveColumnToggle } from "@/components/live/LiveCollapsiblePanel";
import { LiveSessionTracksPanel } from "@/components/live/LiveSessionTracksPanel";
import {
  HarBenchControls,
  useHarBenchBootstrap,
} from "@/components/live/HarBenchControls";
import {
  HAR_MODEL_COLORS,
  useHarLiveState,
} from "@/components/live/useHarLiveState";
import { HarModelsPanel } from "@/components/vision/HarModelsPanel";
import { Icon } from "@/components/ui/Icon";
import {
  captureHarSnapshot,
  createCamera,
  fetchLiveStats,
  mockSlotCameraId,
  syncHarMockWall,
  type HarBenchConfig,
  type HarModelId,
  type HarSnapshotResult,
} from "@/lib/api";
import type { CameraCreateInput, LiveStats } from "@/lib/types";
import { cn } from "@/lib/cn";

const DEFAULT_CONFIG: HarBenchConfig = {
  infer_every: 16,
  buffer_frames: 32,
  stream_fps: 12,
  show_heatmap: false,
  show_yolo_boxes: true,
  top_k: 5,
  ingest_logs: true,
  per_person_mode: true,
  dwell_windows: 2,
  bbox_padding: 0.15,
};

type LiveColumns = {
  video: boolean;
  side: boolean;
};

const DEFAULT_COLUMNS: LiveColumns = { video: true, side: true };

function loadColumns(): LiveColumns {
  if (typeof window === "undefined") return DEFAULT_COLUMNS;
  try {
    const raw = window.localStorage.getItem("visionops:live-columns");
    if (!raw) return DEFAULT_COLUMNS;
    const parsed = JSON.parse(raw) as Partial<LiveColumns> & {
      session?: boolean;
      controls?: boolean;
    };
    if ("side" in parsed) {
      return { ...DEFAULT_COLUMNS, ...parsed };
    }
    // Migrate older 3-column preference
    const side = parsed.session !== false || parsed.controls !== false;
    return { video: parsed.video !== false, side };
  } catch {
    return DEFAULT_COLUMNS;
  }
}

function saveColumns(cols: LiveColumns) {
  if (typeof window === "undefined") return;
  window.localStorage.setItem("visionops:live-columns", JSON.stringify(cols));
}

export function LivePageClient() {
  const { snapshot, loading } = useHarBenchBootstrap();
  const [stats, setStats] = useState<LiveStats | null>(null);
  const [statsLoading, setStatsLoading] = useState(true);
  const [addCameraOpen, setAddCameraOpen] = useState(false);
  const [modelId, setModelId] = useState<HarModelId>("v2-vjepa-powermean7");
  const [videoName, setVideoName] = useState("");
  const [config, setConfig] = useState<HarBenchConfig>(DEFAULT_CONFIG);
  const [playing, setPlaying] = useState(false);
  const [showOverlays, setShowOverlays] = useState(true);
  const [viewLayout, setViewLayout] = useState<LiveViewLayout>("dual");
  const [fullViewIndex, setFullViewIndex] = useState(0);
  const wallReady = useRef(false);
  const bootstrappedRef = useRef(false);
  const [columns, setColumns] = useState<LiveColumns>(DEFAULT_COLUMNS);
  const [snapshotResult, setSnapshotResult] = useState<HarSnapshotResult | null>(null);
  const [snapshotLoading, setSnapshotLoading] = useState(false);

  useEffect(() => {
    setColumns(loadColumns());
  }, []);

  const toggleColumn = useCallback((key: keyof LiveColumns) => {
    setColumns((prev) => {
      const next = { ...prev, [key]: !prev[key] };
      if (!next.video && !next.side) {
        return prev;
      }
      saveColumns(next);
      return next;
    });
  }, []);

  const twoColumn = columns.video && columns.side;

  const loadStats = useCallback(async () => {
    try {
      const statsData = await fetchLiveStats();
      setStats(statsData);
    } finally {
      setStatsLoading(false);
    }
  }, []);

  useEffect(() => {
    void loadStats();
    const id = setInterval(() => void loadStats(), 8_000);
    const onReset = () => void loadStats();
    window.addEventListener(DATA_RESET_EVENT, onReset);
    return () => {
      clearInterval(id);
      window.removeEventListener(DATA_RESET_EVENT, onReset);
    };
  }, [loadStats]);

  useEffect(() => {
    if (!snapshot?.enabled || bootstrappedRef.current) return;
    bootstrappedRef.current = true;
    if (snapshot.videos[0]?.name) {
      setVideoName(snapshot.videos[0].name);
    }
    if (snapshot.state?.model_id) setModelId(snapshot.state.model_id as HarModelId);
    if (snapshot.state?.video) setVideoName(snapshot.state.video);
    if (snapshot.state?.config) setConfig({ ...DEFAULT_CONFIG, ...snapshot.state.config });
    if (typeof snapshot.primary_slot === "number") setFullViewIndex(snapshot.primary_slot);
    if (snapshot.layout === "full" || snapshot.layout === "dual" || snapshot.layout === "quad") {
      setViewLayout(snapshot.layout);
    }
    wallReady.current = true;
  }, [snapshot]);

  const primarySlotIndex = useMemo(() => {
    if (!snapshot?.videos.length) return 0;
    const idx = snapshot.videos.findIndex((v) => v.name === videoName);
    return idx >= 0 ? idx : 0;
  }, [snapshot?.videos, videoName]);

  const pollCameraId = mockSlotCameraId(primarySlotIndex);

  const {
    inferring,
    prediction,
    error,
    logs,
    sessionId,
    modelId: liveModel,
    trackPredictions,
    perPersonMode,
    personCount,
    sessionTracks,
    latestSnapshotUrl,
  } = useHarLiveState(pollCameraId, null, playing);

  const accent = HAR_MODEL_COLORS[liveModel ?? modelId] ?? "#81C784";
  const pred = prediction;
  const topK = pred?.top_k ?? [];
  const showMjpeg = showOverlays && Boolean(snapshot?.enabled);
  const displayTracks = sessionTracks.length ? sessionTracks : trackPredictions;

  useEffect(() => {
    if (!wallReady.current || !videoName) return;
    void syncHarMockWall({
      layout: viewLayout,
      playing,
      model_id: modelId,
      active_video: videoName,
      full_view_index: fullViewIndex,
    });
  }, [viewLayout, playing, modelId, videoName, fullViewIndex]);

  const togglePlayPause = useCallback(() => {
    setPlaying((p) => !p);
  }, []);

  const handleAnalyzeSnapshot = useCallback(async () => {
    if (snapshotLoading) return;
    setSnapshotResult(null);
    setSnapshotLoading(true);
    try {
      const result = await captureHarSnapshot(pollCameraId);
      if (result) setSnapshotResult(result);
    } finally {
      setSnapshotLoading(false);
    }
  }, [pollCameraId, snapshotLoading]);

  const handleSelectVideo = useCallback(
    (name: string) => {
      if (name === videoName) return;
      const idx = snapshot?.videos.findIndex((v) => v.name === name) ?? -1;
      setVideoName(name);
      if (idx >= 0) setFullViewIndex(idx);
      void syncHarMockWall({
        layout: viewLayout,
        playing,
        model_id: modelId,
        active_video: name,
        full_view_index: idx >= 0 ? idx : 0,
      });
    },
    [videoName, snapshot?.videos, viewLayout, playing, modelId],
  );

  const handleFullViewChange = useCallback(
    (index: number) => {
      setFullViewIndex(index);
      const video = snapshot?.videos[index];
      if (video) setVideoName(video.name);
    },
    [snapshot?.videos],
  );

  const handleViewLayoutChange = useCallback(
    (layout: LiveViewLayout) => {
      setViewLayout(layout);
      if (layout === "full") {
        const idx = snapshot?.videos.findIndex((v) => v.name === videoName) ?? 0;
        setFullViewIndex(idx >= 0 ? idx : 0);
      }
    },
    [snapshot?.videos, videoName],
  );

  const benchState = snapshot?.state;
  const lastInferMs = benchState?.last_infer_ms;
  const backend = benchState?.backend;
  const device = benchState?.device;
  const modelLabel =
    snapshot?.models.find((m) => m.model_id === modelId)?.label ?? modelId;

  return (
    <AppShell fullBleed>
      <div className="flex w-full flex-col gap-4 p-4 pb-10 sm:p-6 sm:pb-12 lg:p-8 lg:pb-14">
        <header className="shrink-0">
          <div className="flex flex-col gap-3 lg:flex-row lg:items-start lg:justify-between">
            <div>
              <h2 className="font-headline text-headline-lg text-on-surface">Live Streams</h2>
              <p className="mt-1 max-w-2xl text-body-sm text-outline">
                Mock camera wall with live HAR, batch model probes, per-person ByteTrack, and session
                audit while playback runs.
              </p>
            </div>
            <div className="flex flex-wrap items-center gap-2">
              <button
                type="button"
                onClick={() => setAddCameraOpen(true)}
                className="rounded-md border border-outline-variant/60 px-3 py-1.5 text-body-sm text-on-surface hover:bg-surface-container-high"
              >
                Add camera
              </button>
              <Link
                href="/har-hitl"
                className="rounded-md border border-primary/40 px-3 py-1.5 text-body-sm text-primary hover:bg-primary/10"
              >
                Person HITL
              </Link>
            </div>
          </div>
        </header>

        <div className="shrink-0">
          <LiveStatsBar stats={stats} loading={statsLoading} />
        </div>

        {loading ? (
          <div className="min-h-[24rem] animate-pulse rounded-card bg-surface-container-low" />
        ) : !snapshot?.enabled ? (
          <div className="flex min-h-[24rem] items-center justify-center rounded-card border border-outline-variant/60 bg-surface-container-low p-8 text-center">
            <div>
              <Icon name="videocam_off" className="mx-auto text-outline" size={48} />
              <p className="mt-4 text-body-md text-on-surface">Live HAR is not running</p>
              <p className="mt-1 text-body-sm text-outline">
                Start vision-ops-backend with HAR_ENABLED and HAR_LIVE_ENABLED.
              </p>
            </div>
          </div>
        ) : (
          <div className="flex flex-col gap-3">
            <div className="flex shrink-0 flex-wrap items-center gap-2">
              <span className="font-label text-[10px] font-bold uppercase tracking-wider text-outline">
                Layout
              </span>
              <LiveColumnToggle
                label="Camera"
                active={columns.video}
                onClick={() => toggleColumn("video")}
              />
              <LiveColumnToggle
                label="Session & controls"
                active={columns.side}
                onClick={() => toggleColumn("side")}
                count={displayTracks.length}
              />
            </div>

            <div
              className={cn(
                "grid items-start gap-4",
                twoColumn ? "grid-cols-1 lg:grid-cols-2" : "grid-cols-1",
              )}
            >
              {columns.video ? (
                <div className="min-w-0 lg:sticky lg:top-20 lg:self-start">
                  <LiveCollapsiblePanel
                    id="live-feed"
                    title="Live feed"
                    subtitle={`${modelLabel} · ${videoName || "—"}`}
                    defaultOpen
                    bodyClassName="overflow-hidden p-0"
                    headerExtra={
                      <div className="flex flex-wrap items-center gap-1.5">
                        <span
                          className={cn(
                            "flex items-center gap-1 font-label text-[10px]",
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
                          {showOverlays ? "Overlays" : "Plain"}
                        </button>
                        {playing && (
                          <button
                            type="button"
                            disabled={snapshotLoading}
                            onClick={() => void handleAnalyzeSnapshot()}
                            className={cn(
                              "rounded border px-2 py-0.5 font-label text-[10px] transition-colors",
                              snapshotLoading
                                ? "border-outline-variant text-outline"
                                : "border-secondary bg-secondary/10 text-secondary hover:bg-secondary/20",
                            )}
                          >
                            {snapshotLoading ? "Analyzing…" : "Analyze all"}
                          </button>
                        )}
                      </div>
                    }
                  >
                    <div className="aspect-video w-full bg-black lg:aspect-auto lg:h-[calc(100dvh-11rem)] lg:max-h-[calc(100dvh-11rem)]">
                      <MockVideoQuadGrid
                        videos={snapshot.videos}
                        layout={viewLayout}
                        activeVideoName={videoName}
                        fullViewIndex={fullViewIndex}
                        onFullViewChange={handleFullViewChange}
                        onSelectVideo={handleSelectVideo}
                        playing={playing}
                        onPlayPause={togglePlayPause}
                        showOverlays={showMjpeg}
                        fillHeight
                      />
                    </div>
                  </LiveCollapsiblePanel>

                  {(snapshotLoading || snapshotResult || latestSnapshotUrl) ? (
                    <LiveCollapsiblePanel
                      id="analyzed-frame"
                      title="Analyzed frame"
                      subtitle={
                        snapshotLoading
                          ? "Capturing…"
                          : snapshotResult
                            ? `${snapshotResult.track_count ?? 0} person(s) detected`
                            : pred
                              ? `${pred.label} · ${Math.round(pred.confidence * 100)}%`
                              : undefined
                      }
                      defaultOpen
                      bodyClassName="overflow-hidden p-0"
                    >
                      {snapshotLoading ? (
                        <div className="flex h-40 items-center justify-center bg-surface-container-low">
                          <p className="animate-pulse text-body-sm text-outline">
                            Capturing snapshot…
                          </p>
                        </div>
                      ) : (
                        <>
                          <div className="relative w-full bg-black">
                            {/* eslint-disable-next-line @next/next/no-img-element */}
                            <img
                              key={snapshotResult?.snapshot_url ?? latestSnapshotUrl}
                              src={snapshotResult?.snapshot_url ?? latestSnapshotUrl ?? ""}
                              alt="Analyzed frame"
                              className="w-full object-contain"
                            />
                          </div>
                          {snapshotResult?.tracks && snapshotResult.tracks.length > 0 && (
                            <ul className="divide-y divide-outline-variant/20 p-0">
                              {snapshotResult.tracks.map((tr) => (
                                <li key={tr.track_id} className="flex items-center gap-3 px-3 py-2">
                                  <span className="font-mono text-[11px] font-semibold text-on-surface">
                                    {tr.display_name?.trim() || `#${tr.track_id}`}
                                  </span>
                                  <span className="text-[11px] text-outline">
                                    {tr.inferring
                                      ? "analyzing…"
                                      : tr.action_label ?? "no label yet"}
                                  </span>
                                  {tr.action_confidence != null && !tr.inferring && (
                                    <span
                                      className="ml-auto font-mono text-[11px]"
                                      style={{ color: accent }}
                                    >
                                      {Math.round(tr.action_confidence * 100)}%
                                    </span>
                                  )}
                                </li>
                              ))}
                            </ul>
                          )}
                        </>
                      )}
                    </LiveCollapsiblePanel>
                  ) : null}
                </div>
              ) : null}

              {columns.side ? (
                <div className="flex min-w-0 flex-col gap-3">
                  <LiveCollapsiblePanel
                    id="session-tracks"
                    title="Session & person tracks"
                    subtitle={
                      sessionId
                        ? `${displayTracks.length} track(s) · ${sessionId.slice(0, 8)}…`
                        : playing
                          ? "Opening session…"
                          : "Play to record"
                    }
                    defaultOpen
                    className="min-h-0 shrink-0"
                    bodyClassName="flex min-h-[10rem] flex-col p-0 lg:min-h-[12rem]"
                  >
                    <LiveSessionTracksPanel
                      playing={playing}
                      sessionId={sessionId}
                      videoName={videoName}
                      modelLabel={modelLabel}
                      tracks={displayTracks}
                      accent={accent}
                    />
                  </LiveCollapsiblePanel>

                  <LiveCollapsiblePanel
                    id="inference-log"
                    title="Inference log"
                    subtitle={logs.length ? `${logs.length} entries` : playing ? "Waiting…" : "Paused"}
                    defaultOpen
                    className="min-h-0 shrink-0"
                    bodyClassName="p-0"
                  >
                    <ul className="max-h-48 space-y-1.5 overflow-y-auto overscroll-y-contain p-3 text-body-sm lg:max-h-56">
                      {logs.length === 0 ? (
                        <li className="text-body-sm text-outline">
                          {playing ? "Waiting for predictions…" : "Press play to start inference."}
                        </li>
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
                  </LiveCollapsiblePanel>

                  <LiveCollapsiblePanel
                    id="har-controls"
                    title="HAR controls"
                    subtitle={modelLabel}
                    defaultOpen
                    bodyClassName="p-0"
                  >
                    <HarBenchControls
                      modelId={modelId}
                      videoName={videoName}
                      viewLayout={viewLayout}
                      videos={snapshot.videos}
                      config={config}
                      models={snapshot.models}
                      onModelChange={setModelId}
                      onVideoChange={handleSelectVideo}
                      onViewLayoutChange={handleViewLayoutChange}
                      onConfigChange={setConfig}
                      embedded
                    />
                  </LiveCollapsiblePanel>

                  <LiveCollapsiblePanel
                    id="model-probes"
                    title="Model probes"
                    subtitle="Batch runs on mock clips"
                    defaultOpen={false}
                    bodyClassName="p-0"
                  >
                    <HarModelsPanel embedded />
                  </LiveCollapsiblePanel>

                  <LiveCollapsiblePanel
                    id="scene-summary"
                    title={perPersonMode ? "Scene summary" : "Current prediction"}
                    subtitle={
                      pred ? `${pred.label} · ${Math.round(pred.confidence * 100)}%` : undefined
                    }
                    defaultOpen
                    bodyClassName="space-y-3"
                  >
                    <div className="space-y-2">
                      {!playing && (
                        <p className="text-body-sm text-outline">Paused — no new inference</p>
                      )}
                      {playing && inferring && !pred && (
                        <p className="text-body-sm text-outline">Analyzing window…</p>
                      )}
                      {error && <p className="text-body-sm text-error">{error}</p>}
                      {perPersonMode && (
                        <p className="text-[10px] text-outline">
                          {personCount} person(s) in current frame
                        </p>
                      )}
                      {!perPersonMode && pred ? (
                        <>
                          <p className="font-mono text-lg font-bold text-on-surface">
                            {pred.label}{" "}
                            <span style={{ color: accent }}>
                              {Math.round(pred.confidence * 100)}%
                            </span>
                          </p>
                          <div className="h-2 overflow-hidden rounded-full bg-outline-variant/30">
                            <div
                              className="h-full rounded-full transition-all duration-300"
                              style={{
                                width: `${pred.confidence * 100}%`,
                                backgroundColor: accent,
                              }}
                            />
                          </div>
                          {topK.length > 0 && (
                            <ul className="max-h-36 space-y-2 overflow-y-auto pt-1">
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
                      ) : perPersonMode && playing && !inferring && trackPredictions.length === 0 ? (
                        <p className="text-body-sm text-outline">Waiting for person detections…</p>
                      ) : !perPersonMode && playing && !inferring && !error ? (
                        <p className="text-body-sm text-outline">Waiting for first inference…</p>
                      ) : perPersonMode && pred ? (
                        <p className="font-mono text-lg font-bold text-on-surface">
                          Top: {pred.label}{" "}
                          <span style={{ color: accent }}>
                            {Math.round(pred.confidence * 100)}%
                          </span>
                          {pred.track_id != null ? (
                            <span className="text-body-sm text-outline"> · #{pred.track_id}</span>
                          ) : null}
                        </p>
                      ) : null}
                    </div>
                    {(backend || lastInferMs != null) && (
                      <p className="border-t border-outline-variant/30 pt-3 font-mono text-[10px] text-outline">
                        {backend}
                        {device ? ` · ${device}` : ""}
                        {lastInferMs != null ? ` · ${lastInferMs} ms` : ""}
                      </p>
                    )}
                  </LiveCollapsiblePanel>

                  <LiveCollapsiblePanel
                    id="camera-assistant"
                    title="Camera assistant"
                    defaultOpen={false}
                    bodyClassName="p-0"
                  >
                    <CameraHarChat
                      cameraId={pollCameraId}
                      sessionId={sessionId}
                      accent={accent}
                      embedded
                      defaultExpanded
                    />
                  </LiveCollapsiblePanel>
                </div>
              ) : null}
            </div>
          </div>
        )}
      </div>
      <AddCameraModal
        open={addCameraOpen}
        onClose={() => setAddCameraOpen(false)}
        onSubmit={async (input: CameraCreateInput) => {
          const created = await createCamera(input);
          if (created) {
            setAddCameraOpen(false);
            return true;
          }
          return false;
        }}
      />
    </AppShell>
  );
}
