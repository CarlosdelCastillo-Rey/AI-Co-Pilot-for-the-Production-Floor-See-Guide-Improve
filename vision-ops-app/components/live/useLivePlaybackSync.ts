"use client";

import { useEffect, useRef, type Dispatch, type SetStateAction } from "react";
import { LIVE_PLAYBACK_ALL_EVENT, type LivePlaybackAllDetail } from "@/components/live/livePlaybackEvents";

/** Sync local play/pause with page-level stop/resume-all controls. */
export function useLivePlaybackSync(setPlaying: Dispatch<SetStateAction<boolean>>) {
  const skipBackendSyncRef = useRef(false);

  useEffect(() => {
    const handler = (event: Event) => {
      const detail = (event as CustomEvent<LivePlaybackAllDetail>).detail;
      if (typeof detail?.playing !== "boolean") return;
      skipBackendSyncRef.current = true;
      setPlaying(detail.playing);
    };
    window.addEventListener(LIVE_PLAYBACK_ALL_EVENT, handler);
    return () => window.removeEventListener(LIVE_PLAYBACK_ALL_EVENT, handler);
  }, [setPlaying]);

  return skipBackendSyncRef;
}
