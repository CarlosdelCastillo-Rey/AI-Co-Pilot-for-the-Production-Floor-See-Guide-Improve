"use client";

import { cn } from "@/lib/cn";

interface FloorStatusBannerProps {
  allClear: boolean;
  openCriticalCount?: number;
  openCount?: number;
  className?: string;
}

export function FloorStatusBanner({
  allClear,
  openCriticalCount = 0,
  openCount = 0,
  className,
}: FloorStatusBannerProps) {
  const critical = openCriticalCount > 0;

  return (
    <div
      className={cn(
        "flex min-h-touch-critical items-center justify-between border-b-4 px-6 py-3 transition-colors",
        allClear
          ? "border-success bg-success-container floor-status-clear"
          : critical
            ? "border-error bg-error/10 floor-status-critical"
            : "border-warning bg-warning-container floor-status-warning",
        className,
      )}
      role="status"
      aria-live="polite"
    >
      <div className="flex items-center gap-3">
        <span
          className={cn(
            "inline-block h-4 w-4 rounded-full",
            allClear ? "bg-success" : critical ? "bg-error animate-pulse" : "bg-warning",
          )}
        />
        <p className="font-headline text-headline-md industrial-kpi-label">
          {allClear ? "ALL CLEAR" : critical ? "CRITICAL ALERT" : "ATTENTION REQUIRED"}
        </p>
      </div>
      <p className="font-label text-label-md uppercase tracking-wide">
        {allClear
          ? "No open critical incidents"
          : `${openCriticalCount} critical · ${openCount} open total`}
      </p>
    </div>
  );
}
