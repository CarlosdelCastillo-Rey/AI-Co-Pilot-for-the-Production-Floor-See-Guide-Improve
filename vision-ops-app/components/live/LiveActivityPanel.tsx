"use client";

import { useEffect, useState } from "react";
import { Icon } from "@/components/ui/Icon";
import { fetchRealtimeEvents, fetchTelemetry } from "@/lib/api";
import type { RealtimeEvent } from "@/lib/types";
import { cn } from "@/lib/cn";

const dotColors = {
  critical: "bg-critical",
  primary: "bg-primary",
  neutral: "bg-outline",
};

export function LiveActivityPanel() {
  const [events, setEvents] = useState<RealtimeEvent[]>([]);
  const [inferenceLoad, setInferenceLoad] = useState(24);
  const [gpuDetail, setGpuDetail] = useState("Edge GPU · NVIDIA · 41°C");

  const load = async () => {
    const [evts, telemetry] = await Promise.all([
      fetchRealtimeEvents(6),
      fetchTelemetry(),
    ]);
    setEvents(evts);
    const models = telemetry?.metrics.find((m) => m.service === "vision_models");
    if (models) {
      const pct = models.bars.at(-1) ?? models.bars[0] ?? 24;
      setInferenceLoad(Math.round(pct));
      setGpuDetail(`${models.label.split("(")[0].trim()} · ${models.value}`);
    }
  };

  useEffect(() => {
    void load();
    const id = setInterval(() => void load(), 30_000);
    return () => clearInterval(id);
  }, []);

  return (
    <aside className="flex w-full shrink-0 flex-col border-t border-outline-variant bg-surface-container-lowest xl:w-[320px] xl:border-l xl:border-t-0">
      <div className="flex items-center justify-between border-b border-outline-variant px-4 py-3">
        <h3 className="font-headline text-body-md font-semibold text-on-surface">Live Activity</h3>
        <span className="flex items-center gap-1.5 font-label text-label-sm text-success">
          <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-success" />
          Real-time
        </span>
      </div>

      <div className="flex-1 space-y-0 overflow-y-auto px-4 py-3">
        {events.length === 0 ? (
          <p className="text-body-sm text-outline">No open alerts — floor is clear.</p>
        ) : (
          events.map((event, idx) => (
            <div key={event.id} className="flex gap-3 py-3">
              <div className="flex flex-col items-center pt-1">
                <div className={cn("h-2 w-2 shrink-0 rounded-full", dotColors[event.severity])} />
                {idx < events.length - 1 && (
                  <div className="mt-1 w-px flex-1 bg-outline-variant/60" />
                )}
              </div>
              <div className="min-w-0 pb-1">
                <p className="font-label text-label-sm text-outline">{event.time}</p>
                <p className="text-body-sm font-semibold text-on-surface">{event.title}</p>
                <p className="text-body-sm text-on-surface-variant">{event.description}</p>
              </div>
            </div>
          ))
        )}
      </div>

      <div className="border-t border-outline-variant bg-surface-container-low px-4 py-4">
        <div className="mb-2 flex items-center justify-between">
          <span className="font-label text-label-sm text-outline">AI inference load</span>
          <span className="font-label text-label-sm font-bold text-primary">{inferenceLoad}%</span>
        </div>
        <div className="mb-3 h-1.5 w-full overflow-hidden rounded-full bg-outline-variant/30">
          <div
            className="h-full rounded-full bg-primary transition-all duration-500"
            style={{ width: `${Math.min(100, inferenceLoad)}%` }}
          />
        </div>
        <div className="flex items-center gap-2 text-outline">
          <Icon name="memory" size={16} />
          <span className="font-label text-label-sm">{gpuDetail}</span>
        </div>
      </div>
    </aside>
  );
}
