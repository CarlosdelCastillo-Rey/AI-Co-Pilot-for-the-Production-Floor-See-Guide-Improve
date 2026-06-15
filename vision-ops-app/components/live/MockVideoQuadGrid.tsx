"use client";

import { useCallback, useEffect, useMemo, useRef } from "react";
import { CameraStream } from "@/components/live/CameraStream";
import { VideoFeedControls } from "@/components/live/VideoFeedControls";
import { Icon } from "@/components/ui/Icon";
import { mockSlotStreamUrl } from "@/lib/api";
import { cn } from "@/lib/cn";

export type LiveViewLayout = "full" | "dual" | "quad";

export type MockVideoItem = { name: string; url: string };

type LayoutSlot = {
  slotIndex: number;
  label: string;
  video: MockVideoItem | null;
};

function panelCountForLayout(layout: LiveViewLayout): number {
  if (layout === "full") return 1;
  if (layout === "dual") return 2;
  return 4;
}

export function visibleSlotIndices(
  layout: LiveViewLayout,
  fullViewIndex: number,
  videoCount: number,
): number[] {
  if (videoCount === 0) return [];
  if (layout === "full") {
    const idx = Math.max(0, Math.min(fullViewIndex, videoCount - 1));
    return [idx];
  }
  if (layout === "dual") return Array.from({ length: Math.min(2, videoCount) }, (_, i) => i);
  return Array.from({ length: Math.min(4, videoCount) }, (_, i) => i);
}

function buildLayoutSlots(
  videos: MockVideoItem[],
  layout: LiveViewLayout,
  fullViewIndex: number,
): LayoutSlot[] {
  const count = panelCountForLayout(layout);

  if (layout === "full") {
    const idx = Math.max(0, Math.min(fullViewIndex, Math.max(videos.length - 1, 0)));
    return [{ slotIndex: idx, label: "CAM 01", video: videos[idx] ?? null }];
  }

  return Array.from({ length: count }, (_, i) => ({
    slotIndex: i,
    label: `CAM ${String(i + 1).padStart(2, "0")}`,
    video: videos[i] ?? null,
  }));
}

function gridClassForLayout(layout: LiveViewLayout): string {
  if (layout === "full") return "grid-cols-1 grid-rows-1";
  if (layout === "dual") return "grid-cols-2 grid-rows-1";
  return "grid-cols-2 grid-rows-2";
}

type Props = {
  videos: MockVideoItem[];
  layout: LiveViewLayout;
  activeVideoName: string;
  fullViewIndex: number;
  onFullViewChange: (index: number) => void;
  onSelectVideo: (name: string) => void;
  playing: boolean;
  onPlayPause: () => void;
  showOverlays: boolean;
  selecting?: boolean;
  /** Fill parent height instead of fixed 16:9 aspect ratio */
  fillHeight?: boolean;
};

function EmptySlot({ label }: { label: string }) {
  return (
    <div className="relative flex min-h-0 min-w-0 flex-col items-center justify-center bg-black">
      <div className="pointer-events-none absolute inset-x-0 top-0 bg-gradient-to-b from-black/80 to-transparent px-2 py-1.5">
        <span className="font-label text-[9px] font-bold uppercase tracking-wider text-white/40 sm:text-[10px]">
          {label}
        </span>
      </div>
      <Icon name="videocam_off" className="text-white/25" size={32} />
      <span className="mt-2 font-label text-[10px] uppercase tracking-wider text-white/30">
        No signal
      </span>
    </div>
  );
}

export function MockVideoQuadGrid({
  videos,
  layout,
  activeVideoName,
  fullViewIndex,
  onFullViewChange,
  onSelectVideo,
  playing,
  onPlayPause,
  showOverlays,
  selecting = false,
  fillHeight = false,
}: Props) {
  const slots = useMemo(
    () => buildLayoutSlots(videos, layout, fullViewIndex),
    [videos, layout, fullViewIndex],
  );
  const inferringSlots = useMemo(
    () => new Set(visibleSlotIndices(layout, fullViewIndex, videos.length)),
    [layout, fullViewIndex, videos.length],
  );
  const videoRefs = useRef<(HTMLVideoElement | null)[]>([]);

  useEffect(() => {
    videoRefs.current.forEach((el) => {
      if (!el) return;
      if (playing) void el.play().catch(() => undefined);
      else el.pause();
    });
  }, [playing, slots]);

  const setVideoRef = useCallback((index: number, el: HTMLVideoElement | null) => {
    videoRefs.current[index] = el;
  }, []);

  const canGoPrev = layout === "full" && fullViewIndex > 0;
  const canGoNext = layout === "full" && fullViewIndex < videos.length - 1;

  const goPrev = useCallback(() => {
    if (!canGoPrev) return;
    onFullViewChange(fullViewIndex - 1);
  }, [canGoPrev, fullViewIndex, onFullViewChange]);

  const goNext = useCallback(() => {
    if (!canGoNext) return;
    onFullViewChange(fullViewIndex + 1);
  }, [canGoNext, fullViewIndex, onFullViewChange]);

  if (videos.length === 0) {
    return (
      <div
        className={cn(
          "relative flex w-full items-center justify-center bg-black",
          fillHeight ? "h-full min-h-[12rem]" : "aspect-video",
        )}
      >
        <Icon name="videocam_off" className="text-outline" size={48} />
      </div>
    );
  }

  return (
    <div
      className={cn(
        "relative w-full overflow-hidden bg-black",
        fillHeight ? "h-full min-h-[12rem]" : "aspect-video",
      )}
    >
      <div
        className={cn(
          "absolute inset-0 grid gap-px bg-black",
          gridClassForLayout(layout),
        )}
      >
        {slots.map((slot) => {
          if (!slot.video) {
            return <EmptySlot key={slot.slotIndex} label={slot.label} />;
          }

          const isActive = slot.video.name === activeVideoName;
          const isInferring = inferringSlots.has(slot.slotIndex);
          const showMjpeg = isInferring && showOverlays && playing;
          const streamUrl = mockSlotStreamUrl(slot.slotIndex);

          return (
            <button
              key={slot.slotIndex}
              type="button"
              disabled={selecting}
              onClick={() => onSelectVideo(slot.video!.name)}
              className={cn(
                "group relative min-h-0 min-w-0 overflow-hidden bg-black text-left transition-shadow",
                isActive
                  ? "z-[1] ring-2 ring-inset ring-primary"
                  : "hover:ring-1 hover:ring-inset hover:ring-primary/40",
                selecting && "pointer-events-none opacity-80",
              )}
              title={
                isActive
                  ? `${slot.video.name} — primary HAR log`
                  : `Set primary HAR log to ${slot.video.name}`
              }
            >
              {showMjpeg ? (
                <CameraStream
                  streamUrl={streamUrl}
                  alt={`${slot.label} HAR`}
                  paused={!playing}
                />
              ) : (
                <video
                  ref={(el) => setVideoRef(slot.slotIndex, el)}
                  src={slot.video.url}
                  className="absolute inset-0 h-full w-full object-cover"
                  muted
                  loop
                  playsInline
                  preload="metadata"
                />
              )}

              <div className="pointer-events-none absolute inset-x-0 top-0 flex items-start justify-between gap-1 bg-gradient-to-b from-black/55 to-transparent px-2 py-1.5">
                <span className="font-label text-[9px] font-bold uppercase tracking-wider text-white/90 sm:text-[10px]">
                  {slot.label}
                </span>
                <div className="flex gap-1">
                  {isInferring ? (
                    <span className="rounded bg-success/90 px-1.5 py-0.5 font-label text-[8px] font-bold uppercase text-on-primary sm:text-[9px]">
                      LIVE
                    </span>
                  ) : null}
                  {isActive ? (
                    <span className="rounded bg-primary/90 px-1.5 py-0.5 font-label text-[8px] font-bold uppercase text-on-primary sm:text-[9px]">
                      LOG
                    </span>
                  ) : null}
                </div>
              </div>

              <div className="pointer-events-none absolute inset-x-0 bottom-0 truncate bg-black/50 px-2 py-1">
                <span className="font-mono text-[9px] text-white/90 sm:text-[10px]">
                  {slot.video.name}
                </span>
              </div>

              {!playing && (
                <div className="pointer-events-none absolute inset-0 flex items-center justify-center bg-black/20">
                  <Icon name="pause" className="text-white/70" size={28} />
                </div>
              )}
            </button>
          );
        })}
      </div>

      {layout === "full" && videos.length > 1 ? (
        <>
          <button
            type="button"
            disabled={!canGoPrev || selecting}
            onClick={goPrev}
            className={cn(
              "absolute left-3 top-1/2 z-10 flex h-10 w-10 -translate-y-1/2 items-center justify-center rounded-full bg-black/55 text-white transition hover:bg-black/75",
              (!canGoPrev || selecting) && "pointer-events-none opacity-30",
            )}
            aria-label="Previous video"
          >
            <Icon name="chevron_left" size={28} />
          </button>
          <button
            type="button"
            disabled={!canGoNext || selecting}
            onClick={goNext}
            className={cn(
              "absolute right-3 top-1/2 z-10 flex h-10 w-10 -translate-y-1/2 items-center justify-center rounded-full bg-black/55 text-white transition hover:bg-black/75",
              (!canGoNext || selecting) && "pointer-events-none opacity-30",
            )}
            aria-label="Next video"
          >
            <Icon name="chevron_right" size={28} />
          </button>
          <div className="pointer-events-none absolute left-1/2 top-3 z-10 -translate-x-1/2 rounded-full bg-black/55 px-3 py-1 font-label text-[10px] text-white/90">
            {fullViewIndex + 1} / {videos.length}
          </div>
        </>
      ) : null}

      <div className="absolute bottom-3 left-3 z-10">
        <VideoFeedControls playing={playing} onPlayPause={onPlayPause} />
      </div>
    </div>
  );
}
