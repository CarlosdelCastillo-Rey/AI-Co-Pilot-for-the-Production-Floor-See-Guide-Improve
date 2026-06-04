export const LIVE_PLAYBACK_ALL_EVENT = "visionops:live-playback-all";

export type LivePlaybackAllDetail = {
  playing: boolean;
};

export function dispatchLivePlaybackAll(playing: boolean): void {
  window.dispatchEvent(
    new CustomEvent<LivePlaybackAllDetail>(LIVE_PLAYBACK_ALL_EVENT, {
      detail: { playing },
    }),
  );
}
