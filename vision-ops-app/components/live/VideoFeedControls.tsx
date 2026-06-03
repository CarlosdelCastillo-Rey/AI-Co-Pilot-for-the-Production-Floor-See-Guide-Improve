"use client";

import { Icon } from "@/components/ui/Icon";
import { cn } from "@/lib/cn";

type VideoFeedControlsProps = {
  playing: boolean;
  onPlayPause: () => void;
  className?: string;
};

export function VideoFeedControls({ playing, onPlayPause, className }: VideoFeedControlsProps) {
  return (
    <div
      className={cn(
        "inline-flex rounded-lg bg-on-surface/75 px-3 py-2 backdrop-blur-sm",
        className,
      )}
    >
      <button
        type="button"
        aria-label={playing ? "Pause" : "Play"}
        onClick={onPlayPause}
        className="rounded p-1 text-white hover:bg-white/15"
      >
        <Icon name={playing ? "pause" : "play_arrow"} size={22} />
      </button>
    </div>
  );
}
