import { AppShell } from "@/components/layout/AppShell";
import { Icon } from "@/components/ui/Icon";

export default function AnalyticsPage() {
  return (
    <AppShell fullBleed>
      <div className="flex h-[calc(100vh-4rem)] overflow-hidden">
        <section className="relative flex w-[70%] flex-col border-r border-outline-variant">
          <div className="flex h-14 shrink-0 items-center justify-between border-b border-outline-variant bg-surface px-6">
            <div className="flex items-center gap-4">
              <div className="flex items-center gap-2 rounded border border-outline-variant bg-surface-container-low px-3 py-1.5">
                <Icon name="calendar_today" size={16} />
                <span className="text-label-sm">Oct 24, 2023</span>
              </div>
              <div className="flex gap-1 rounded border border-outline-variant bg-surface-container-high p-1">
                <button type="button" className="rounded bg-surface px-3 py-1 text-label-sm font-bold text-primary shadow-sm">
                  Morning
                </button>
                <button type="button" className="rounded px-3 py-1 text-label-sm hover:bg-surface/50">
                  Evening
                </button>
                <button type="button" className="rounded px-3 py-1 text-label-sm hover:bg-surface/50">
                  Night
                </button>
              </div>
            </div>
            <div className="flex items-center gap-3">
              <label className="text-label-sm text-on-surface-variant">
                Active Camera View:
              </label>
              <select className="rounded border-outline-variant bg-surface py-1 pl-2 pr-8 text-label-sm focus:ring-primary">
                <option>CAM-04 (Assembly East)</option>
                <option>CAM-08 (Loading Dock)</option>
                <option>CAM-12 (Robotics Cell)</option>
              </select>
            </div>
          </div>
          <div className="blueprint-grid flex flex-grow items-center justify-center overflow-hidden bg-white p-gutter">
            <div className="relative h-full max-h-[700px] w-full max-w-5xl overflow-hidden rounded-xl border-2 border-outline-variant/30 bg-surface-container-lowest">
              <div className="heatmap-overlay absolute inset-0 opacity-80" />
              <div className="absolute left-1/4 top-1/4 flex h-64 w-32 items-center justify-center rounded-lg border-2 border-outline">
                <span className="rotate-90 text-label-sm text-outline">LINE A</span>
              </div>
              <div className="absolute left-[45%] top-1/4 flex h-64 w-32 items-center justify-center rounded-lg border-2 border-outline">
                <span className="rotate-90 text-label-sm text-outline">LINE B</span>
              </div>
              <div className="absolute left-[65%] top-1/4 flex h-64 w-32 items-center justify-center rounded-lg border-2 border-outline">
                <span className="rotate-90 text-label-sm text-outline">LINE C</span>
              </div>
              <div className="absolute left-[45%] top-[45%] flex h-24 w-24 animate-pulse items-center justify-center rounded-full border-2 border-error bg-error/20">
                <span className="rounded-full bg-error px-1.5 py-0.5 text-[10px] font-bold text-white">
                  CRITICAL
                </span>
              </div>
              <div className="absolute left-20 top-20 flex h-4 w-4 items-center justify-center rounded-sm border-2 border-primary bg-primary-container">
                <span className="absolute -top-6 left-0 whitespace-nowrap bg-primary px-1 font-label text-[9px] text-white">
                  ID: 884 (0.98)
                </span>
              </div>
              <div className="absolute bottom-6 left-6 z-10 rounded-lg border border-outline-variant bg-white/90 p-4 shadow-sm backdrop-blur-md">
                <h4 className="mb-2 text-label-sm font-bold uppercase tracking-wider">
                  Heatmap Intensity
                </h4>
                <div className="flex items-center gap-2">
                  <span className="font-label text-[10px]">LOW</span>
                  <div className="h-2 w-32 rounded-full bg-gradient-to-r from-blue-400 via-green-400 to-red-500" />
                  <span className="font-label text-[10px]">HIGH</span>
                </div>
              </div>
              <div className="absolute right-6 top-6 rounded-md border border-black/10 bg-black/5 px-3 py-2 font-label text-label-sm backdrop-blur-sm">
                <p className="font-bold">GRID: 34.02.1 / -118.24</p>
                <p className="opacity-70">SENSORS ACTIVE: 42</p>
                <p className="mt-1 font-bold text-error">ANOMALIES: 03</p>
              </div>
            </div>
          </div>
        </section>
        <section className="flex w-[30%] flex-col overflow-y-auto bg-surface-container-low p-6">
          <h2 className="mb-6 flex items-center gap-2 font-headline text-headline-md">
            <Icon name="insert_chart" className="text-primary" />
            Operational Insights
          </h2>
          <div className="space-y-gutter">
            <InsightCard title="Flow Efficiency" value="94.2%" trend="+2.1%" />
            <DowntimeCard />
            <BottleneckCard />
            <RecommendationCard />
          </div>
        </section>
      </div>
    </AppShell>
  );
}

function InsightCard({
  title,
  value,
  trend,
}: {
  title: string;
  value: string;
  trend: string;
}) {
  const heights = [60, 75, 65, 85, 95, 80, 70];
  return (
    <article className="rounded-xl border border-outline-variant bg-surface p-md">
      <div className="mb-4 flex items-start justify-between">
        <div>
          <p className="text-label-sm uppercase text-on-surface-variant">{title}</p>
          <h3 className="font-headline text-headline-md">{value}</h3>
        </div>
        <span className="flex items-center gap-1 text-sm font-bold text-green-600">
          <Icon name="trending_up" size={16} />
          {trend}
        </span>
      </div>
      <div className="flex h-32 items-end gap-1 px-2">
        {heights.map((h, i) => (
          <div
            key={i}
            className={`w-full rounded-t-sm ${i === 4 ? "border-t-2 border-primary bg-primary/30" : "bg-primary/10"}`}
            style={{ height: `${h}%` }}
          />
        ))}
      </div>
      <div className="mt-2 flex justify-between px-1 font-label text-[10px] text-on-surface-variant">
        <span>08:00</span>
        <span>12:00</span>
        <span>16:00</span>
      </div>
    </article>
  );
}

function DowntimeCard() {
  const stations = [
    { name: "Assembly Line A", value: "12.5m", width: "45%", critical: false },
    { name: "Packaging Cell B", value: "48.2m", width: "85%", critical: true },
    { name: "Robotics Station 04", value: "04.1m", width: "15%", critical: false },
  ];
  return (
    <article className="rounded-xl border border-outline-variant bg-surface p-md">
      <p className="mb-4 text-label-sm uppercase text-on-surface-variant">
        Est. Downtime per Station (min)
      </p>
      <div className="space-y-4">
        {stations.map((s) => (
          <div key={s.name} className="space-y-1">
            <div className="flex justify-between text-label-sm">
              <span>{s.name}</span>
              <span className={s.critical ? "font-bold text-error" : "font-bold"}>
                {s.value}
              </span>
            </div>
            <div className="h-2 w-full overflow-hidden rounded-full bg-surface-container-high">
              <div
                className={`h-full rounded-full ${s.critical ? "bg-error" : "bg-primary"}`}
                style={{ width: s.width }}
              />
            </div>
          </div>
        ))}
      </div>
    </article>
  );
}

function BottleneckCard() {
  return (
    <article className="overflow-hidden rounded-xl border border-outline-variant bg-surface">
      <div className="border-b border-outline-variant bg-secondary-container/50 px-4 py-2">
        <p className="text-label-sm font-bold uppercase">Critical Bottleneck Events</p>
      </div>
      <div className="divide-y divide-outline-variant">
        <div className="flex gap-3 bg-error/5 p-4">
          <Icon name="report" filled className="text-error" />
          <div>
            <p className="text-body-sm font-bold">Heavy Congestion: Zone A-12</p>
            <p className="text-label-sm text-on-surface-variant">
              Detected 4 mins ago • Duration 12s
            </p>
          </div>
        </div>
        <div className="flex gap-3 p-4">
          <Icon name="warning" className="text-tertiary" />
          <div>
            <p className="text-body-sm font-bold">Pathway Obstruction</p>
            <p className="text-label-sm text-on-surface-variant">
              Detected 18 mins ago • Cleared
            </p>
          </div>
        </div>
      </div>
      <button
        type="button"
        className="w-full border-t border-outline-variant py-3 text-label-sm text-primary transition-colors hover:bg-surface-container-high"
      >
        View Historical Log
      </button>
    </article>
  );
}

function RecommendationCard() {
  return (
    <article className="flex items-center gap-4 rounded-xl bg-inverse-surface p-md">
      <div className="flex h-12 w-12 items-center justify-center rounded-lg bg-primary-fixed-dim/20">
        <Icon name="auto_awesome" className="text-primary-fixed-dim" />
      </div>
      <div>
        <p className="text-body-sm font-bold text-surface-bright">AI Recommendation</p>
        <p className="text-body-sm text-on-surface-variant">
          Reroute Forklift-03 to North Gate to alleviate Sector 7 congestion.
        </p>
      </div>
    </article>
  );
}
