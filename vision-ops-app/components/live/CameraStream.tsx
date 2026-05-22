"use client";

import { useState } from "react";

type CameraStreamProps = {
  streamUrl: string;
  alt: string;
  offlineMessage?: string | null;
};

/** Native img for MJPEG multipart streams (not compatible with next/image). */
export function CameraStream({ streamUrl, alt, offlineMessage }: CameraStreamProps) {
  const [failed, setFailed] = useState(false);

  if (failed) {
    return (
      <div className="absolute inset-0 flex flex-col items-center justify-center gap-2 bg-on-surface-variant/10 p-md text-center">
        <span className="material-symbols-outlined text-4xl text-outline">videocam_off</span>
        <p className="text-label-md text-on-surface">Webcam stream unavailable</p>
        <p className="max-w-xs text-body-sm text-outline">
          {offlineMessage ??
            "Start vision-ops-backend on port 8000 and allow camera access for Terminal or Cursor."}
        </p>
      </div>
    );
  }

  return (
    // eslint-disable-next-line @next/next/no-img-element
    <img
      src={streamUrl}
      alt={alt}
      className="absolute inset-0 h-full w-full object-cover"
      onError={() => setFailed(true)}
    />
  );
}
