"use client";

import { Icon } from "@/components/ui/Icon";
import { harV2ArtifactUrl } from "@/lib/har-v2-api";

export function PersonCropThumb({
  url,
  size = "md",
}: {
  url: string | null | undefined;
  size?: "sm" | "md" | "lg";
}) {
  const src = harV2ArtifactUrl(url);
  const sizeClass =
    size === "lg" ? "h-40 w-32" : size === "sm" ? "h-10 w-10" : "h-14 w-14";
  if (!src) {
    return (
      <span
        className={`${sizeClass} flex shrink-0 items-center justify-center rounded-md bg-surface-container-high text-outline`}
      >
        <Icon name="person" size={size === "lg" ? 36 : 20} />
      </span>
    );
  }
  return (
    <span
      className={`${sizeClass} relative block shrink-0 overflow-hidden rounded-md bg-surface-container-high`}
    >
      {/* eslint-disable-next-line @next/next/no-img-element */}
      <img src={src} alt="" className="h-full w-full object-cover" />
    </span>
  );
}
