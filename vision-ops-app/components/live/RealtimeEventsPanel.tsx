import { Icon } from "@/components/ui/Icon";
import type { RealtimeEvent } from "@/lib/types";
import { cn } from "@/lib/cn";

const dotColors = {
  critical: "bg-error",
  primary: "bg-primary",
  neutral: "bg-on-surface-variant",
};

export function RealtimeEventsPanel({ events }: { events: RealtimeEvent[] }) {
  return (
    <aside className="hidden w-80 shrink-0 flex-col border-l border-outline-variant bg-surface-container-lowest 2xl:flex">
      <div className="border-b border-outline-variant bg-surface-container p-md">
        <h3 className="text-label-md font-bold text-on-surface">
          Real-time Events
        </h3>
      </div>
      <div className="flex-1 space-y-4 overflow-y-auto p-md">
        {events.length === 0 ? (
          <p className="text-body-sm text-outline">No events recorded yet.</p>
        ) : (
          events.map((event) => (
            <div key={event.id} className="flex gap-3">
              <div className="flex flex-col items-center">
                <div
                  className={cn(
                    "h-2 w-2 rounded-full",
                    dotColors[event.severity],
                  )}
                />
                <div className="h-full w-[2px] bg-outline-variant" />
              </div>
              <div className="pb-4">
                <p className="text-label-sm text-outline">{event.time}</p>
                <p className="text-body-sm font-bold text-on-surface">
                  {event.title}
                </p>
                <p className="text-body-sm text-outline">{event.description}</p>
              </div>
            </div>
          ))
        )}
      </div>
      <div className="border-t border-outline-variant bg-surface-container-high p-md">
        <div className="mb-2 flex items-center justify-between">
          <span className="text-label-sm text-outline">AI Inference Load</span>
          <span className="text-label-sm font-bold text-primary">24%</span>
        </div>
        <div className="h-1 w-full overflow-hidden rounded-full bg-surface-variant">
          <div className="h-full w-1/4 bg-primary" />
        </div>
      </div>
    </aside>
  );
}
