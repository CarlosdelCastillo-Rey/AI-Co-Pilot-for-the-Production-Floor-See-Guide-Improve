import Image from "next/image";
import { CameraStream } from "@/components/live/CameraStream";
import { Icon } from "@/components/ui/Icon";
import type { CameraFeed } from "@/lib/mock-data";
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
  if (feed.image.startsWith("/") || feed.image.startsWith("http://localhost")) {
    return (
      // eslint-disable-next-line @next/next/no-img-element
      <img
        src={feed.image}
        alt={feed.name}
        className="absolute inset-0 h-full w-full object-cover"
      />
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
  tertiary: "border-tertiary opacity-50",
  error: "border-error",
};

const labelVariants = {
  primary: "bg-primary",
  tertiary: "bg-tertiary",
  error: "bg-error",
};

export function CameraFeedCard({ feed }: { feed: CameraFeed }) {
  const isWebcamFeed = feed.id === "webcam-0" || Boolean(feed.streamUrl);
  const showStream = feed.status === "live" && Boolean(feed.streamUrl);
  const bakedVisionPoster = Boolean(feed.previewUrl ?? feed.heatmapUrl);
  const showHtmlOverlays = !showStream && !bakedVisionPoster;
  const hudLabel = isWebcamFeed
    ? feed.status === "live"
      ? "LIVE - WEBCAM"
      : "WEBCAM OFFLINE"
    : "LIVE - RTSP CONNECTED";

  return (
    <article className="group overflow-hidden rounded-lg border border-outline-variant bg-surface-container-lowest shadow-sm">
      <div className="relative aspect-video bg-on-surface-variant/5">
        {showStream && feed.streamUrl ? (
          <CameraStream
            streamUrl={feed.streamUrl}
            alt={feed.name}
            offlineMessage={feed.error}
          />
        ) : isWebcamFeed ? (
          <div className="absolute inset-0 flex flex-col items-center justify-center gap-2 bg-on-surface-variant/10 p-md text-center">
            <span className="material-symbols-outlined text-4xl text-outline">videocam_off</span>
            <p className="text-label-md text-on-surface">Webcam not connected</p>
            <p className="max-w-sm text-body-sm text-outline">
              {feed.error ??
                "Run vision-ops-backend (port 8000) and allow camera access in macOS Settings → Privacy → Camera."}
            </p>
          </div>
        ) : (
          <MockCameraPoster feed={feed} />
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
                  overlay.variant === "tertiary" && "bg-tertiary",
                  overlay.variant === "error" && "bg-error",
                )}
              >
                {overlay.label}
              </div>
            </div>
          ))}
        </div>
        <div className="command-hud absolute right-4 top-4 flex items-center gap-2 rounded-full px-3 py-1">
          <span
            className={cn(
              "h-2 w-2 rounded-full",
              feed.status === "live" ? "animate-pulse bg-[#4CAF50]" : "bg-outline",
            )}
          />
          <span className="text-label-sm text-white">{hudLabel}</span>
        </div>
        <div className="command-hud absolute left-4 top-4 rounded px-2 py-1 font-label text-label-sm text-white">
          {feed.coords}
        </div>
        <div className="command-hud absolute bottom-0 left-0 right-0 flex justify-between p-4 opacity-0 transition-opacity group-hover:opacity-100">
          <div className="flex gap-4">
            <Icon name="play_arrow" className="cursor-pointer text-white" />
            <Icon name="volume_up" className="cursor-pointer text-white" />
          </div>
          <Icon name="fullscreen" className="cursor-pointer text-white" />
        </div>
      </div>
      <div className="flex items-center justify-between p-md">
        <div>
          <h3 className="text-label-md text-on-surface">{feed.name}</h3>
          <p className="text-body-sm text-outline">{feed.location}</p>
        </div>
        <div className="flex gap-2 text-on-surface-variant">
          <Icon name="analytics" className="cursor-pointer" size={20} />
          <Icon name="more_vert" className="cursor-pointer" size={20} />
        </div>
      </div>
    </article>
  );
}
