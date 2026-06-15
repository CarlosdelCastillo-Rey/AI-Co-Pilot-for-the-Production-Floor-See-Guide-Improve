"use client";

import Link from "next/link";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { CameraStream } from "@/components/live/CameraStream";
import { useHarBenchBootstrap } from "@/components/live/HarBenchControls";
import { VideoFeedControls } from "@/components/live/VideoFeedControls";
import { HAR_MODEL_COLORS, useHarLiveState } from "@/components/live/useHarLiveState";
import { Icon } from "@/components/ui/Icon";
import {
  mockSlotCameraId,
  mockSlotStreamUrl,
  syncHarMockWall,
  type HarModelId,
} from "@/lib/api";
import { cn } from "@/lib/cn";

/** Persistent live HAR feed for dashboard split layouts (matches /live mock wall). */
export function LiveFeedPanel() {
  const { snapshot, loading } = useHarBenchBootstrap();
  const [videoName, setVideoName] = useState("");
  const [modelId, setModelId] = useState<HarModelId>("v2-vjepa");
  const [playing, setPlaying] = useState(false);
  const [showOverlays, setShowOverlays] = useState(true);
  const wallReady = useRef(false);
  const bootstrappedRef = useRef(false);

  useEffect(() => {
    if (!snapshot?.enabled || bootstrappedRef.current) return;
    bootstrappedRef.current = true;
    if (snapshot.videos[0]?.name) {
      setVideoName(snapshot.videos[0].name);
    }
    if (snapshot.state?.model_id) setModelId(snapshot.state.model_id as HarModelId);
    if (snapshot.state?.video) setVideoName(snapshot.state.video);
    wallReady.current = true;
  }, [snapshot]);

  const primarySlotIndex = useMemo(() => {
    if (!snapshot?.videos.length) return 0;
    const idx = snapshot.videos.findIndex((v) => v.name === videoName);
    return idx >= 0 ? idx : 0;
  }, [snapshot?.videos, videoName]);

  const pollCameraId = mockSlotCameraId(primarySlotIndex);

  const { prediction, inferring, error, modelId: liveModel, trackPredictions, perPersonMode } =
    useHarLiveState(pollCameraId, null, playing);

  const accent = HAR_MODEL_COLORS[liveModel ?? modelId] ?? "#81C784";
  const streamUrl = mockSlotStreamUrl(primarySlotIndex);
  const showMjpeg = showOverlays && Boolean(snapshot?.enabled);
  const modelLabel =
    snapshot?.models.find((m) => m.model_id === (liveModel ?? modelId))?.label ??
    liveModel ??
    modelId ??
    "HAR";

  useEffect(() => {
    if (!wallReady.current || !videoName) return;
    void syncHarMockWall({
      layout: "full",
      playing,
      model_id: modelId,
      active_video: videoName,
      full_view_index: primarySlotIndex,
    });
  }, [playing, modelId, videoName, primarySlotIndex]);

  const togglePlayPause = useCallback(() => {
    setPlaying((p) => !p);
  }, []);

  const selectVideo = useCallback(
    (name: string) => {
      const idx = snapshot?.videos.findIndex((v) => v.name === name) ?? 0;
      setVideoName(name);
      void syncHarMockWall({
        layout: "full",
        playing,
        model_id: modelId,
        active_video: name,
        full_view_index: idx >= 0 ? idx : 0,
      });
    },
    [snapshot?.videos, playing, modelId],
  );

  const activeVideo =
    snapshot?.videos.find((v) => v.name === videoName) ?? snapshot?.videos[0] ?? null;

  return (
    <div className="flex h-full min-h-0 flex-col">
      <div className="flex shrink-0 items-center justify-between gap-2 border-b border-outline-variant/50 px-4 py-3">
        <div className="min-w-0">
          <p className="font-label text-[10px] font-bold uppercase tracking-wider text-outline">
            Live floor feed
          </p>
          <p className="truncate text-body-sm font-semibold text-on-surface">{modelLabel}</p>
          {videoName ? (
            <p className="truncate text-[11px] text-outline" title={videoName}>
              {videoName}
            </p>
          ) : null}
        </div>
        <div className="flex shrink-0 items-center gap-2">
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
            {showOverlays ? "Overlays" : "Plain"}
          </button>
          <Link
            href="/live"
            className="rounded border border-outline-variant/60 px-2 py-0.5 font-label text-[10px] text-primary hover:bg-primary/10"
          >
            Expand
          </Link>
        </div>
      </div>

      <div className="relative min-h-0 flex-1 bg-black">
        {loading ? (
          <div className="absolute inset-0 animate-pulse bg-surface-container-high" />
        ) : !snapshot?.enabled ? (
          <div className="absolute inset-0 flex flex-col items-center justify-center gap-2 p-6 text-center">
            <Icon name="videocam_off" className="text-outline" size={40} />
            <p className="text-body-sm text-on-surface">Live HAR offline</p>
            <p className="text-[11px] text-outline">Start backend with HAR_ENABLED</p>
          </div>
        ) : showMjpeg && streamUrl ? (
          <CameraStream
            key={pollCameraId}
            streamUrl={streamUrl}
            alt="Live HAR feed"
            paused={!playing}
          />
        ) : activeVideo ? (
          <video
            key={activeVideo.url}
            src={activeVideo.url}
            className="absolute inset-0 h-full w-full object-cover"
            autoPlay={playing}
            muted
            loop
            playsInline
          />
        ) : (
          <div className="absolute inset-0 flex items-center justify-center">
            <Icon name="videocam_off" className="text-white/30" size={40} />
          </div>
        )}

        {snapshot?.enabled && snapshot.videos.length > 1 ? (
          <div className="absolute right-3 top-3 z-10 flex max-w-[45%] flex-col gap-1">
            {snapshot.videos.map((v) => (
              <button
                key={v.name}
                type="button"
                onClick={() => selectVideo(v.name)}
                className={cn(
                  "truncate rounded bg-black/55 px-2 py-1 text-left font-mono text-[9px] text-white/90 backdrop-blur-sm",
                  v.name === videoName && "ring-1 ring-primary",
                )}
              >
                {v.name}
              </button>
            ))}
          </div>
        ) : null}

        <div className="absolute bottom-3 left-3 z-10">
          <VideoFeedControls playing={playing} onPlayPause={togglePlayPause} />
        </div>
      </div>

      <div className="shrink-0 border-t border-outline-variant/50 bg-surface-container-low px-4 py-3">
        {error ? <p className="text-body-sm text-error">{error}</p> : null}
        {playing && inferring && !prediction ? (
          <p className="text-body-sm text-outline">Analyzing…</p>
        ) : null}
        {perPersonMode && trackPredictions.length > 0 ? (
          <ul className="max-h-24 space-y-1 overflow-y-auto">
            {trackPredictions.slice(0, 4).map((tr) => (
              <li key={tr.track_id} className="font-mono text-[11px] text-on-surface">
                #{tr.track_id}{" "}
                {tr.action_label ? (
                  <>
                    {tr.action_label}{" "}
                    <span style={{ color: accent }}>
                      {Math.round((tr.action_confidence ?? 0) * 100)}%
                    </span>
                  </>
                ) : (
                  <span className="text-outline">warming…</span>
                )}
              </li>
            ))}
          </ul>
        ) : prediction ? (
          <p className="font-mono text-body-sm font-semibold text-on-surface">
            {prediction.label}{" "}
            <span style={{ color: accent }}>{Math.round(prediction.confidence * 100)}%</span>
          </p>
        ) : (
          <p className="text-body-sm text-outline">
            {playing ? "Waiting for inference…" : "Press play to start live HAR"}
          </p>
        )}
      </div>
    </div>
  );
}
