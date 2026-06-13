import type { AnalyticsHeatmapApi, AnalyticsParetoApi } from "@/lib/api";

const OEE_TARGET_PCT = 85;

const ZONE_LAYOUT = [
  { id: "IN", title: "Infeed", left: "3%", width: "13%" },
  { id: "S1", title: "Feed", left: "18%", width: "14%" },
  { id: "S2", title: "Place", left: "34%", width: "14%" },
  { id: "S3", title: "Press", left: "50%", width: "15%" },
  { id: "S4", title: "Weld", left: "67%", width: "13%" },
  { id: "S5", title: "QA Gate", left: "82%", width: "15%" },
] as const;

export type FloorZone = {
  id: string;
  title: string;
  left: string;
  width: string;
  heatClass: `h${0 | 1 | 2 | 3 | 4}`;
  label: string;
  isBottleneck: boolean;
};

export function oeeTargetMarkerLeft(segmentTargetPct: number): string {
  return `${Math.min(100, Math.max(0, segmentTargetPct))}%`;
}

export { OEE_TARGET_PCT };

export function parseTrendDirection(trend: string): "up" | "down" | "flat" {
  const t = trend.trim();
  if (t.startsWith("+")) return "up";
  if (t.startsWith("-")) return "down";
  return "flat";
}

export function coqBudgetUsd(totalCostUsd: number): number {
  return Math.max(8000, Math.round(totalCostUsd * 1.35));
}

export function coqBudgetTone(pctOfBudget: number): "ok" | "warn" | "" {
  if (pctOfBudget >= 90) return "warn";
  if (pctOfBudget < 50) return "ok";
  return "";
}

export function paretoTopThreePct(items: AnalyticsParetoApi["items"]): number | null {
  if (!items.length) return null;
  const top = items.slice(0, 3);
  const sum = top.reduce((acc, i) => acc + i.pct, 0);
  return Math.round(sum);
}

export function heatmapToFloorZones(heatmap: AnalyticsHeatmapApi | null): FloorZone[] {
  const cells = heatmap?.grid.cells ?? [];
  const w = heatmap?.grid.width ?? 10;
  const h = heatmap?.grid.height ?? 10;
  const zoneCount = ZONE_LAYOUT.length;

  const colIntensity: number[] = Array(zoneCount).fill(0);
  if (cells.length > 0) {
    for (let zi = 0; zi < zoneCount; zi++) {
      const xStart = Math.floor((zi / zoneCount) * w);
      const xEnd = Math.floor(((zi + 1) / zoneCount) * w);
      let sum = 0;
      let n = 0;
      for (let y = 0; y < h; y++) {
        for (let x = xStart; x < xEnd; x++) {
          sum += cells[y]?.[x] ?? 0;
          n++;
        }
      }
      colIntensity[zi] = n > 0 ? sum / n : 0;
    }
  } else {
    colIntensity.fill(0);
  }

  const max = Math.max(...colIntensity, 0.01);
  let peakIdx = 0;
  colIntensity.forEach((v, i) => {
    if (v >= colIntensity[peakIdx]) peakIdx = i;
  });

  const labels = ["low", "light", "moderate", "elevated", "bottleneck"];
  return ZONE_LAYOUT.map((z, i) => {
    const norm = colIntensity[i] / max;
    const level = Math.min(4, Math.floor(norm * 4)) as 0 | 1 | 2 | 3 | 4;
    const isBottleneck = i === peakIdx && level >= 3;
    const minutes = isBottleneck ? `${(norm * 22).toFixed(1)}m dwell` : labels[level];
    return {
      ...z,
      heatClass: `h${level}` as FloorZone["heatClass"],
      label: minutes,
      isBottleneck,
    };
  });
}

export function formatShiftLabel(shift: string): string {
  return `${shift.charAt(0).toUpperCase()}${shift.slice(1)} shift`;
}
