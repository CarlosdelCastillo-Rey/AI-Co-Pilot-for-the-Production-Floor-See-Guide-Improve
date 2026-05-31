"use client";

import { cn } from "@/lib/cn";
import type { ResolutionStatus } from "@/lib/api";

const STATUS_STYLES: Record<ResolutionStatus, string> = {
  OPEN: "bg-[#FBE7E4] text-[#C0362C] border-[#F0C4BF]",
  ACKNOWLEDGED: "bg-[#FBF0DB] text-[#B7791F] border-[#E8D4A8]",
  RESOLVED: "bg-[#E6F4EC] text-[#1F8A5B] border-[#B8DFC8]",
  FALSE_POSITIVE: "bg-[#F1F3F6] text-[#687079] border-[#DDE1E6]",
};

export function ResolutionStatusBadge({ status }: { status: ResolutionStatus }) {
  const label =
    status === "FALSE_POSITIVE"
      ? "Closed"
      : status === "RESOLVED"
        ? "Resolved"
        : status.charAt(0) + status.slice(1).toLowerCase();

  return (
    <span
      className={cn(
        "inline-flex items-center rounded-full border px-2.5 py-0.5 font-label text-[11px] font-bold uppercase tracking-wide",
        STATUS_STYLES[status],
      )}
    >
      {label}
    </span>
  );
}
