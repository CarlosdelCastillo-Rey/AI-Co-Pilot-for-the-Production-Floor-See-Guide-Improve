"use client";

import { useRouter } from "next/navigation";
import { useCallback, useEffect, useRef, useState } from "react";
import { DATA_RESET_EVENT } from "@/components/layout/Sidebar";
import { Icon } from "@/components/ui/Icon";
import { fetchRealtimeEvents, fetchTimelineStats } from "@/lib/api";
import type { RealtimeEvent } from "@/lib/types";
import { cn } from "@/lib/cn";

const dotColors = {
  critical: "bg-critical",
  primary: "bg-primary",
  neutral: "bg-outline",
};

export function NotificationsBell() {
  const router = useRouter();
  const [open, setOpen] = useState(false);
  const [events, setEvents] = useState<RealtimeEvent[]>([]);
  const [pendingCount, setPendingCount] = useState(0);
  const [loading, setLoading] = useState(false);
  const rootRef = useRef<HTMLDivElement>(null);

  const goToTimeline = useCallback(() => {
    setOpen(false);
    router.push("/timeline");
  }, [router]);

  const load = useCallback(async () => {
    setLoading(true);
    const [evts, stats] = await Promise.all([fetchRealtimeEvents(12), fetchTimelineStats()]);
    setEvents(evts);
    setPendingCount(stats?.openCount ?? evts.length);
    setLoading(false);
  }, []);

  useEffect(() => {
    void load();
    const id = setInterval(() => void load(), 30_000);
    const onReset = () => void load();
    window.addEventListener(DATA_RESET_EVENT, onReset);
    return () => {
      clearInterval(id);
      window.removeEventListener(DATA_RESET_EVENT, onReset);
    };
  }, [load]);

  useEffect(() => {
    if (!open) return;
    void load();
  }, [open, load]);

  useEffect(() => {
    if (!open) return;
    const onPointerDown = (e: PointerEvent) => {
      const target = e.target as Node | null;
      if (rootRef.current && target && !rootRef.current.contains(target)) {
        setOpen(false);
      }
    };
    const onEscape = (e: KeyboardEvent) => {
      if (e.key === "Escape") setOpen(false);
    };
    document.addEventListener("pointerdown", onPointerDown);
    document.addEventListener("keydown", onEscape);
    return () => {
      document.removeEventListener("pointerdown", onPointerDown);
      document.removeEventListener("keydown", onEscape);
    };
  }, [open]);

  const criticalCount = events.filter((e) => e.severity === "critical").length;
  const badgeCount = pendingCount > 0 ? pendingCount : 0;
  const badgeCritical = criticalCount > 0 || (pendingCount > 0 && events.some((e) => e.severity === "critical"));

  return (
    <div ref={rootRef} className="relative">
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        aria-expanded={open}
        aria-haspopup="true"
        aria-label="Notifications"
        className={cn(
          "relative rounded-lg p-2 transition-colors hover:bg-surface-container-low hover:text-on-surface",
          open && "bg-surface-container-low text-on-surface",
        )}
      >
        <Icon name="notifications" size={20} />
        {badgeCount > 0 && (
          <span
            className={cn(
              "absolute right-1 top-1 flex h-4 min-w-4 items-center justify-center rounded-full border border-surface-container-lowest px-1 text-[10px] font-bold leading-none text-on-primary",
              badgeCritical ? "bg-critical" : "bg-primary",
            )}
          >
            {badgeCount > 9 ? "9+" : badgeCount}
          </span>
        )}
      </button>

      {open && (
        <div
          role="menu"
          onPointerDown={(e) => e.stopPropagation()}
          className="absolute right-0 top-full z-[100] mt-2 w-[min(22rem,calc(100vw-2rem))] overflow-hidden rounded-xl border border-outline-variant bg-surface-container-lowest shadow-lg"
        >
          <div className="flex items-center justify-between border-b border-outline-variant px-4 py-3">
            <h2 className="font-headline text-body-md font-semibold text-on-surface">Notifications</h2>
            <span className="font-label text-label-sm text-outline">
              {events.length > 0 ? `${events.length} need triage` : "All clear"}
            </span>
          </div>

          <div className="max-h-80 overflow-y-auto">
            {loading && events.length === 0 ? (
              <p className="px-4 py-6 text-body-sm text-outline">Loading…</p>
            ) : events.length === 0 ? (
              <div className="px-4 py-6 text-center">
                <span className="mx-auto mb-3 flex h-10 w-10 items-center justify-center rounded-full bg-success-container text-success">
                  <Icon name="check_circle" size={22} />
                </span>
                <p className="text-body-sm font-semibold text-on-surface">All clear</p>
                <p className="mt-1 text-body-sm text-outline">
                  No open alerts. Acknowledged and resolved incidents are on the timeline.
                </p>
              </div>
            ) : (
              <ul className="divide-y divide-outline-variant/50">
                {events.map((event) => (
                  <li key={event.id}>
                    <button
                      type="button"
                      role="menuitem"
                      onClick={goToTimeline}
                      className="flex w-full gap-3 px-4 py-3 text-left transition-colors hover:bg-surface-container-low"
                    >
                      <span
                        className={cn(
                          "mt-1.5 h-2 w-2 shrink-0 rounded-full",
                          dotColors[event.severity],
                        )}
                      />
                      <span className="min-w-0">
                        <span className="block font-label text-label-sm text-outline">
                          {event.time}
                        </span>
                        <span className="block text-body-sm font-semibold text-on-surface">
                          {event.title}
                        </span>
                        <span className="block truncate text-body-sm text-on-surface-variant">
                          {event.description}
                        </span>
                      </span>
                    </button>
                  </li>
                ))}
              </ul>
            )}
          </div>

          <div className="border-t border-outline-variant bg-surface-container-low px-4 py-3">
            <button
              type="button"
              onClick={goToTimeline}
              className="flex min-h-touch w-full items-center justify-center gap-1 text-body-sm font-semibold text-primary hover:underline"
            >
              View timeline
              <Icon name="arrow_forward" size={16} />
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
