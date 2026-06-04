"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import Image from "next/image";
import { CameraStream } from "@/components/live/CameraStream";
import { VideoFeedControls } from "@/components/live/VideoFeedControls";
import { useLivePlaybackSync } from "@/components/live/useLivePlaybackSync";
import { Icon } from "@/components/ui/Icon";
import type { CameraFeed } from "@/lib/types";

export function CameraFeedCard({ feed }: { feed: CameraFeed }) {
  const isWebcamFeed = feed.backendCameraId === "webcam-0" || feed.id === "webcam-0";
  const showStream = feed.status === "live" && Boolean(feed.streamUrl);
  const showVideo = Boolean(feed.videoUrl) && !showStream;

  const videoRef = useRef<HTMLVideoElement>(null);
  const [playing, setPlaying] = useState(false);
  useLivePlaybackSync(setPlaying);

  useEffect(() => {
    const v = videoRef.current;
    if (!v || !showVideo) return;
    if (playing) void v.play().catch(() => setPlaying(false));
    else v.pause();
  }, [playing, showVideo, feed.videoUrl]);

  const togglePlayPause = useCallback(() => setPlaying((p) => !p), []);

  return (
    <article className="overflow-hidden rounded-card border border-outline-variant/60 bg-surface-container-lowest shadow-sm">
      <div className="relative aspect-video w-full max-h-[min(50vh,420px)] overflow-hidden bg-on-surface-variant/5">
        {showStream && feed.streamUrl ? (
          <CameraStream
            streamUrl={feed.streamUrl}
            alt={feed.name}
            offlineMessage={feed.error}
            paused={!playing}
          />
        ) : showVideo ? (
          <video
            ref={videoRef}
            src={feed.videoUrl}
            className="absolute inset-0 h-full w-full object-cover"
            autoPlay={playing}
            muted
            loop
            playsInline
            onPlay={() => setPlaying(true)}
            onPause={() => setPlaying(false)}
          />
        ) : isWebcamFeed && feed.status !== "live" ? (
          <div className="absolute inset-0 flex flex-col items-center justify-center gap-2 bg-surface-container p-md text-center">
            <Icon name="videocam_off" className="text-4xl text-outline" />
            <p className="text-label-md text-on-surface">Webcam not connected</p>
            <p className="max-w-sm text-body-sm text-outline">
              {feed.error ?? "Run vision-ops-backend (port 8000)."}
            </p>
          </div>
        ) : feed.image ? (
          <Image src={feed.image} alt={feed.name} fill className="object-cover" unoptimized />
        ) : (
          <div className="absolute inset-0 flex items-center justify-center">
            <Icon name="videocam_off" className="text-outline" size={48} />
          </div>
        )}

        {(showStream || showVideo) && (
          <div className="absolute bottom-3 left-3 z-10">
            <VideoFeedControls playing={playing} onPlayPause={togglePlayPause} />
          </div>
        )}
      </div>
      <div className="px-4 py-3">
        <h3 className="text-body-sm font-semibold text-on-surface">{feed.name}</h3>
        <p className="text-body-sm text-outline">{feed.location}</p>
      </div>
    </article>
  );
}

function isHarCamera(feed: CameraFeed): boolean {
  return feed.id.startsWith("cam-har-") || (feed.backendCameraId?.startsWith("cam-har-") ?? false);
}

export { isHarCamera };
