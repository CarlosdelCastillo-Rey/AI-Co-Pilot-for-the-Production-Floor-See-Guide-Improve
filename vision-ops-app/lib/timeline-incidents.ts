import type { Severity } from "@/lib/types";

/** Timeline incidents require triage — info/normal are activity logs only. */
export function isTimelineIncident(severity: Severity | string): boolean {
  return severity === "critical" || severity === "warning";
}
