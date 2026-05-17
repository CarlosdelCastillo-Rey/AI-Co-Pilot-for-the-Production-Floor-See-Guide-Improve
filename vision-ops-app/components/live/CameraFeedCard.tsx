import Image from "next/image";
import { Icon } from "@/components/ui/Icon";
import type { CameraFeed } from "@/lib/mock-data";
import { cn } from "@/lib/cn";

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
  return (
    <article className="group overflow-hidden rounded-lg border border-outline-variant bg-surface-container-lowest shadow-sm">
      <div className="relative aspect-video bg-on-surface-variant/5">
        <Image
          src={feed.image}
          alt={feed.name}
          fill
          className="object-cover"
          sizes="(max-width: 1280px) 100vw, 50vw"
        />
        <div className="pointer-events-none absolute inset-0">
          {feed.overlays.map((overlay) => (
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
          <span className="h-2 w-2 animate-pulse rounded-full bg-[#4CAF50]" />
          <span className="text-label-sm text-white">LIVE - RTSP CONNECTED</span>
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
