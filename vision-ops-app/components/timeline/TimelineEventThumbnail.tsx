"use client";

import { useEffect, useState } from "react";
import { Icon } from "@/components/ui/Icon";

type TimelineEventThumbnailProps = {
  src: string;
  title: string;
  clipDuration?: string | null;
};

export function TimelineEventThumbnail({ src, title, clipDuration }: TimelineEventThumbnailProps) {
  const [open, setOpen] = useState(false);

  useEffect(() => {
    if (!open) return;
    const onKey = (event: KeyboardEvent) => {
      if (event.key === "Escape") setOpen(false);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [open]);

  return (
    <>
      <button
        type="button"
        onClick={() => setOpen(true)}
        className="group relative m-4 mt-0 block h-[180px] w-full shrink-0 cursor-zoom-in overflow-hidden rounded-[10px] border border-[#EBEDF1] bg-[#F5F6F8] focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-[#0059BB] md:m-0 md:h-auto md:w-[200px] md:min-h-[160px] md:self-stretch md:rounded-none md:rounded-r-[10px] md:border-0 md:border-l md:border-[#EBEDF1]"
        aria-label={`Expand evidence image for ${title}`}
      >
        {/* eslint-disable-next-line @next/next/no-img-element */}
        <img
          src={src}
          alt={`Evidence: ${title}`}
          className="h-full w-full object-cover transition-transform duration-200 group-hover:scale-[1.03]"
        />
        <span className="pointer-events-none absolute inset-0 flex items-center justify-center bg-black/0 transition-colors group-hover:bg-black/25">
          <Icon
            name="zoom_in"
            size={28}
            className="text-white opacity-0 drop-shadow-md transition-opacity group-hover:opacity-100"
          />
        </span>
        {clipDuration && clipDuration !== "00:00" ? (
          <span className="pointer-events-none absolute bottom-1.5 right-1.5 rounded bg-black/75 px-1.5 py-0.5 font-label text-[10px] text-white">
            {clipDuration}
          </span>
        ) : null}
      </button>

      {open ? (
        <div className="fixed inset-0 z-[110] flex items-center justify-center p-4 sm:p-8">
          <button
            type="button"
            className="absolute inset-0 bg-[#0C0F13]/70 backdrop-blur-sm"
            onClick={() => setOpen(false)}
            aria-label="Close image preview"
          />
          <div className="relative z-10 flex max-h-[92vh] w-full max-w-[min(960px,92vw)] flex-col overflow-hidden rounded-[14px] border border-[#EBEDF1] bg-white shadow-[0_24px_48px_rgba(12,15,19,0.2)]">
            <div className="flex items-center justify-between gap-3 border-b border-[#EBEDF1] px-4 py-3">
              <p className="truncate font-headline text-[15px] font-semibold text-[#0C0F13]">{title}</p>
              <button
                type="button"
                onClick={() => setOpen(false)}
                className="rounded-lg p-2 text-[#687079] hover:bg-[#F5F6F8]"
                aria-label="Close"
              >
                <Icon name="close" size={20} />
              </button>
            </div>
            <div className="flex min-h-0 flex-1 items-center justify-center bg-[#161B22] p-2 sm:p-4">
              {/* eslint-disable-next-line @next/next/no-img-element */}
              <img
                src={src}
                alt={`Evidence: ${title}`}
                className="max-h-[calc(92vh-80px)] max-w-full object-contain"
              />
            </div>
          </div>
        </div>
      ) : null}
    </>
  );
}
