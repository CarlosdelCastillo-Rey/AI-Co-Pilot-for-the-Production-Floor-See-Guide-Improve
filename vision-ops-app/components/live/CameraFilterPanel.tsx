"use client";

import { Button } from "@/components/ui/Button";
import { cn } from "@/lib/cn";

export interface CameraFilters {
  status: "" | "live" | "offline";
  model: "" | "dinov3" | "vjepa2" | "yolov8";
}

interface CameraFilterPanelProps {
  open: boolean;
  filters: CameraFilters;
  onChange: (filters: CameraFilters) => void;
  onClose: () => void;
}

export function CameraFilterPanel({ open, filters, onChange, onClose }: CameraFilterPanelProps) {
  if (!open) return null;

  return (
    <div className="absolute right-0 top-full z-20 mt-2 w-64 rounded-card border border-outline-variant bg-surface-container-lowest p-4 shadow-overlay">
      <p className="mb-3 font-label text-label-sm font-bold uppercase tracking-wide text-outline">
        Filter feeds
      </p>

      <label className="mb-3 block">
        <span className="mb-1 block text-body-sm text-on-surface-variant">Status</span>
        <select
          value={filters.status}
          onChange={(e) =>
            onChange({ ...filters, status: e.target.value as CameraFilters["status"] })
          }
          className="w-full rounded-lg border border-outline-variant/70 bg-surface-container-low px-3 py-2 text-body-sm"
        >
          <option value="">All</option>
          <option value="live">Live</option>
          <option value="offline">Offline</option>
        </select>
      </label>

      <label className="mb-4 block">
        <span className="mb-1 block text-body-sm text-on-surface-variant">Inference model</span>
        <select
          value={filters.model}
          onChange={(e) =>
            onChange({ ...filters, model: e.target.value as CameraFilters["model"] })
          }
          className="w-full rounded-lg border border-outline-variant/70 bg-surface-container-low px-3 py-2 text-body-sm"
        >
          <option value="">All models</option>
          <option value="dinov3">DINOv3</option>
          <option value="vjepa2">V-JEPA 2</option>
          <option value="yolov8">YOLOv8</option>
        </select>
      </label>

      <div className="flex gap-2">
        <Button
          variant="ghost"
          className="flex-1"
          onClick={() => onChange({ status: "", model: "" })}
        >
          Clear
        </Button>
        <Button className={cn("flex-1")} onClick={onClose}>
          Apply
        </Button>
      </div>
    </div>
  );
}
