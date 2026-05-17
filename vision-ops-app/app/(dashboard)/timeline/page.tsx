import Image from "next/image";
import { AppShell } from "@/components/layout/AppShell";
import { Badge } from "@/components/ui/Badge";
import { Button } from "@/components/ui/Button";
import { Icon } from "@/components/ui/Icon";
import { SHIFT_SUMMARY, TIMELINE_EVENTS } from "@/lib/mock-data";
import type { Severity } from "@/lib/mock-data";
import { cn } from "@/lib/cn";

const severityBadge: Record<
  Severity,
  "critical" | "warning" | "info" | "neutral"
> = {
  critical: "critical",
  warning: "warning",
  info: "info",
  normal: "neutral",
};

const dotColor: Record<Severity, string> = {
  critical: "bg-error",
  warning: "bg-tertiary",
  info: "bg-primary",
  normal: "bg-on-surface-variant",
};

export default function TimelinePage() {
  return (
    <AppShell searchPlaceholder="Search events, operators, or incidents...">
      <div className="flex min-h-[calc(100vh-4rem)]">
        <div className="flex-1 bg-surface-container-lowest">
          <header className="mx-auto max-w-[1000px] border-b border-outline-variant px-8 pb-6 pt-10">
            <div className="flex items-end justify-between">
              <div>
                <h2 className="font-headline text-headline-lg text-on-background">
                  Post-Shift Log
                </h2>
                <p className="mt-1 text-body-md text-secondary">
                  Operational audit for {SHIFT_SUMMARY.date}
                </p>
              </div>
              <div className="flex gap-2">
                <Button variant="secondary" icon="filter_list">
                  Filter
                </Button>
                <Button icon="download">Export PDF</Button>
              </div>
            </div>
          </header>
          <div className="relative mx-auto max-w-[1000px] px-8 py-10">
            <div className="absolute bottom-10 left-8 top-10 w-0.5 bg-outline-variant" />
            <div className="relative space-y-12">
              {TIMELINE_EVENTS.map((event) => (
                <article key={event.id} className="relative pl-12">
                  <div
                    className={cn(
                      "absolute left-[-5px] top-1 h-3 w-3 rounded-full ring-4 ring-surface-container-lowest",
                      dotColor[event.severity],
                    )}
                  />
                  <div className="flex flex-col gap-6 rounded-xl border border-outline-variant bg-white p-6 shadow-sm transition-colors hover:border-error/30 md:flex-row">
                    <div className="flex-1">
                      <div className="mb-3 flex items-center gap-3">
                        <span className="text-label-md text-secondary">
                          {event.time}
                        </span>
                        <Badge variant={severityBadge[event.severity]}>
                          {event.severity === "info"
                            ? "INFO"
                            : event.severity.toUpperCase()}
                        </Badge>
                      </div>
                      <h3 className="mb-2 font-headline text-[18px] text-on-background">
                        {event.title}
                      </h3>
                      <p className="text-body-md leading-relaxed text-on-surface-variant">
                        {event.description}
                      </p>
                      <div className="mt-4 flex gap-4">
                        {event.meta.map((m) => (
                          <span
                            key={m.text}
                            className="flex items-center gap-1.5 text-label-sm text-secondary"
                          >
                            <Icon name={m.icon} size={16} />
                            {m.text}
                          </span>
                        ))}
                      </div>
                    </div>
                    <button
                      type="button"
                      className="group relative w-full shrink-0 cursor-pointer overflow-hidden rounded-lg md:w-56"
                    >
                      <Image
                        src={event.thumbnail}
                        alt={event.title}
                        width={224}
                        height={128}
                        className="h-32 w-full object-cover transition-transform duration-500 group-hover:scale-110"
                      />
                      <div className="absolute inset-0 flex items-center justify-center bg-black/40 opacity-0 backdrop-blur-[2px] transition-opacity group-hover:opacity-100">
                        <div className="flex h-10 w-10 items-center justify-center rounded-full border border-white/40 bg-white/20">
                          <Icon name="play_arrow" filled className="text-white" />
                        </div>
                      </div>
                      <span className="absolute bottom-1 right-1 rounded bg-black/70 px-1.5 py-0.5 font-label text-[10px] text-white">
                        {event.clipDuration}
                      </span>
                    </button>
                  </div>
                </article>
              ))}
            </div>
            <div className="mt-16 flex justify-center">
              <button
                type="button"
                className="rounded-lg border border-outline px-8 py-3 text-label-md text-on-surface-variant transition-colors hover:bg-surface-container"
              >
                Load Older Logs (Oct 23)
              </button>
            </div>
          </div>
        </div>
        <aside className="hidden w-[320px] shrink-0 overflow-y-auto border-l border-outline-variant bg-white p-6 xl:block">
          <h3 className="mb-6 font-headline text-[18px]">Shift Summary</h3>
          <div className="space-y-6">
            <div className="rounded-lg border border-outline-variant bg-surface-container p-4">
              <p className="mb-2 text-label-sm uppercase text-secondary">
                Incident Count
              </p>
              <div className="flex items-center justify-between">
                <span className="font-headline text-headline-md text-error">
                  {String(SHIFT_SUMMARY.incidentCount).padStart(2, "0")}
                </span>
                <span className="text-label-sm text-error">
                  {SHIFT_SUMMARY.incidentDelta}
                </span>
              </div>
              <div className="mt-3 h-1 w-full overflow-hidden rounded-full bg-outline-variant">
                <div className="h-full w-[65%] bg-error" />
              </div>
            </div>
            <div className="rounded-lg border border-outline-variant bg-surface-container p-4">
              <p className="mb-2 text-label-sm uppercase text-secondary">
                Uptime Rate
              </p>
              <div className="flex items-center justify-between">
                <span className="font-headline text-headline-md text-primary">
                  {SHIFT_SUMMARY.uptime}
                </span>
                <Icon name="trending_up" className="text-primary" />
              </div>
            </div>
            <div className="border-t border-outline-variant pt-6">
              <h4 className="mb-4 text-label-md text-on-background">
                Top Affected Assets
              </h4>
              <div className="space-y-3">
                {SHIFT_SUMMARY.assets.map((asset) => (
                  <div
                    key={asset.name}
                    className="flex items-center justify-between"
                  >
                    <span className="text-body-sm text-on-surface-variant">
                      {asset.name}
                    </span>
                    <span className="text-label-sm font-bold">
                      {asset.events} Events
                    </span>
                  </div>
                ))}
              </div>
            </div>
            <Button variant="inverse" icon="description" className="w-full">
              Full Report
            </Button>
          </div>
        </aside>
      </div>
    </AppShell>
  );
}
