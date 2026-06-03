"use client";

import { useEffect, useRef, useState } from "react";

type CameraStreamProps = {
  streamUrl: string;
  alt: string;
  offlineMessage?: string | null;
  /** When true, disconnect MJPEG (freeze last frame). */
  paused?: boolean;
};

/** Native img for MJPEG multipart streams (not compatible with next/image). */
export function CameraStream({ streamUrl, alt, offlineMessage, paused = false }: CameraStreamProps) {
  const [failed, setFailed] = useState(false);
  const [activeUrl, setActiveUrl] = useState(streamUrl);
  const lastFrameRef = useRef<string | null>(null);

  useEffect(() => {
    if (paused) {
      setActiveUrl("");
      return;
    }
    setFailed(false);
    setActiveUrl(streamUrl);
  }, [paused, streamUrl]);

  useEffect(() => {
    return () => {
      lastFrameRef.current = null;
    };
  }, []);

  if (failed && !paused) {
    return (
      <div className="absolute inset-0 flex flex-col items-center justify-center gap-2 bg-on-surface-variant/10 p-md text-center">
        <span className="material-symbols-outlined text-4xl text-outline">videocam_off</span>
        <p className="text-label-md text-on-surface">Stream unavailable</p>
        <p className="max-w-xs text-body-sm text-outline">
          {offlineMessage ??
            "Start vision-ops-backend on port 8000 and allow camera access for Terminal or Cursor."}
        </p>
      </div>
    );
  }

  return (
    <div className="absolute inset-0 overflow-hidden">
      {paused && lastFrameRef.current ? (
        // eslint-disable-next-line @next/next/no-img-element
        <img
          src={lastFrameRef.current}
          alt={alt}
          className="h-full w-full max-w-none object-cover opacity-90"
        />
      ) : null}
      {!paused && activeUrl ? (
        // eslint-disable-next-line @next/next/no-img-element
        <img
          src={activeUrl}
          alt={alt}
          className="h-full w-full max-w-none object-cover"
          onLoad={(e) => {
            const img = e.currentTarget;
            if (img.src) lastFrameRef.current = img.src;
          }}
          onError={() => setFailed(true)}
        />
      ) : paused && !lastFrameRef.current ? (
        <div className="flex h-full w-full items-center justify-center bg-on-surface-variant/20">
          <span className="font-label text-label-sm text-white/80">Paused</span>
        </div>
      ) : null}
    </div>
  );
}
