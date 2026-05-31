import Image from "next/image";
import { CameraStream } from "@/components/live/CameraStream";
import { Icon } from "@/components/ui/Icon";
import type { CameraFeed } from "@/lib/types";
import { cn } from "@/lib/cn";

function MockCameraPoster({ feed }: { feed: CameraFeed }) {
  const posterSrc = feed.previewUrl ?? feed.heatmapUrl;
  if (posterSrc) {
    return (
      // eslint-disable-next-line @next/next/no-img-element
      <img
        src={posterSrc}
        alt={feed.name}
        className="absolute inset-0 h-full w-full object-cover"
      />
    );
  }
  if (feed.image.startsWith("/") || feed.image.startsWith("http")) {
    return (
      // eslint-disable-next-line @next/next/no-img-element
      <img
        src={feed.image}
        alt={feed.name}
        className="absolute inset-0 h-full w-full object-cover"
      />
    );
  }
  if (!feed.image) {
    return (
      <div className="absolute inset-0 flex items-center justify-center bg-surface-container">
        <Icon name="videocam_off" className="text-outline" size={48} />
      </div>
    );
  }
  return (
    <Image
      src={feed.image}
      alt={feed.name}
      fill
      className="object-cover"
      sizes="(max-width: 1280px) 100vw, 50vw"
      unoptimized
    />
  );
}

const borderVariants = {
  primary: "border-primary",
  tertiary: "border-tertiary opacity-60",
  error: "border-error",
};

const labelVariants = {
  primary: "bg-primary",
  tertiary: "bg-tertiary",
  error: "bg-error",
};

export function CameraFeedCard({ feed }: { feed: CameraFeed }) {
  const isWebcamFeed = feed.backendCameraId === "webcam-0" || feed.id === "webcam-0";
  const showStream = feed.status === "live" && Boolean(feed.streamUrl);
  const bakedVisionPoster = Boolean(feed.previewUrl ?? feed.heatmapUrl);
  const showHtmlOverlays = !showStream && !bakedVisionPoster;
  const badge = feed.modelBadge;
  const hasAnomalyScore = feed.anomalyScore != null;

  return (
    <article className="group overflow-hidden rounded-card border border-outline-variant/60 bg-surface-container-lowest shadow-sm transition-shadow hover:shadow-md">
      <div className="relative aspect-video bg-on-surface-variant/5">
        {showStream && feed.streamUrl ? (
          <CameraStream
            streamUrl={feed.streamUrl}
            alt={feed.name}
            offlineMessage={feed.error}
          />
        ) : isWebcamFeed && feed.status !== "live" ? (
          <div className="absolute inset-0 flex flex-col items-center justify-center gap-2 bg-surface-container p-md text-center">
            <Icon name="videocam_off" className="text-4xl text-outline" />
            <p className="text-label-md text-on-surface">Webcam not connected</p>
            <p className="max-w-sm text-body-sm text-outline">
              {feed.error ??
                "Run vision-ops-backend (port 8000) and allow camera access."}
            </p>
          </div>
        ) : (
          <MockCameraPoster feed={feed} />
        )}

        {badge && (
          <div className="absolute left-3 top-3 flex flex-col gap-1">
            <span className="inline-flex items-center gap-1 rounded-md bg-on-surface/75 px-2 py-1 font-label text-[10px] font-medium uppercase tracking-wide text-white backdrop-blur-sm">
              {badge.model}
            </span>
            {badge.task && (
              <span className="inline-flex w-fit rounded-md bg-primary/90 px-2 py-0.5 font-label text-[10px] text-white">
                · {badge.task}
              </span>
            )}
          </div>
        )}

        <div className="pointer-events-none absolute inset-0">
          {showHtmlOverlays &&
            feed.overlays.map((overlay) => (
              <div
                key={overlay.label}
                className={cn(
                  "ai-bounding-box absolute",
                  borderVariants[overlay.variant ?? "primary"],
                )}
                style={{
                  top: overlay.top,
                  left: overlay.left,
                  width: overlay.width,
                  height: overlay.height,
                }}
              >
                <div
                  className={cn(
                    "ai-label absolute -top-5 left-0",
                    labelVariants[overlay.variant ?? "primary"],
                  )}
                >
                  {overlay.label}
                </div>
              </div>
            ))}
        </div>

        <div className="command-hud absolute right-3 top-3 flex items-center gap-1.5 rounded-full px-2.5 py-1">
          <span
            className={cn(
              "h-1.5 w-1.5 rounded-full",
              feed.status === "live" ? "animate-pulse bg-[#36C886]" : "bg-outline",
            )}
          />
          <span className="font-label text-[10px] font-bold tracking-wider text-white">LIVE</span>
        </div>

        {feed.coords && (
          <div className="command-hud absolute bottom-3 left-3 max-w-[70%] rounded px-2 py-1 font-label text-[10px] text-white">
            {feed.coords}
          </div>
        )}

        {hasAnomalyScore && (
          <div className="command-hud absolute bottom-3 right-3 rounded px-2 py-1 font-label text-[10px] text-white">
            score {feed.anomalyScore!.toFixed(2)}
          </div>
        )}

        <div className="command-hud absolute bottom-0 left-0 right-0 flex justify-between p-3 opacity-0 transition-opacity group-hover:opacity-100">
          <div className="flex gap-3">
            <Icon name="play_arrow" className="cursor-pointer text-white" size={20} />
            <Icon name="volume_up" className="cursor-pointer text-white" size={20} />
          </div>
          <Icon name="fullscreen" className="cursor-pointer text-white" size={20} />
        </div>
      </div>

      <div className="flex items-center justify-between px-4 py-3">
        <div className="min-w-0">
          <h3 className="truncate text-body-sm font-semibold text-on-surface">{feed.name}</h3>
          <p className="truncate text-body-sm text-outline">{feed.location}</p>
        </div>
        <div className="flex shrink-0 gap-1 text-outline">
          <button type="button" className="rounded p-1.5 hover:bg-surface-container-low hover:text-primary">
            <Icon name="analytics" size={18} />
          </button>
          <button type="button" className="rounded p-1.5 hover:bg-surface-container-low hover:text-primary">
            <Icon name="more_vert" size={18} />
          </button>
        </div>
      </div>
    </article>
  );
}
