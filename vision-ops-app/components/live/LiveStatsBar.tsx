import { Icon } from "@/components/ui/Icon";
import { KpiLabel } from "@/components/shared/KpiHelp";
import type { LiveStats } from "@/lib/types";
import { cn } from "@/lib/cn";

interface LiveStatsBarProps {
  stats: LiveStats | null;
  loading?: boolean;
}

function StatCard({
  label,
  kpiId,
  value,
  sub,
  subPositive,
  icon,
}: {
  label: string;
  kpiId: string;
  value: string;
  sub: string;
  subPositive?: boolean;
  icon: string;
}) {
  return (
    <article className="rounded-card border border-outline-variant/50 bg-surface-container-lowest p-4 shadow-sm">
      <div className="mb-3 flex items-center justify-between">
        <KpiLabel label={label} kpiId={kpiId} className="font-label text-label-sm uppercase tracking-wide text-outline" />
        <Icon name={icon} size={18} className="text-outline/70" />
      </div>
      <p className="font-headline text-[28px] font-bold leading-none text-on-surface">{value}</p>
      <p
        className={cn(
          "mt-2 font-label text-label-sm",
          subPositive === true && "text-success",
          subPositive === false && "text-warning",
          subPositive === undefined && "text-outline",
        )}
      >
        {sub}
      </p>
    </article>
  );
}

export function LiveStatsBar({ stats, loading }: LiveStatsBarProps) {
  if (loading || !stats) {
    return (
      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 xl:grid-cols-4">
        {[1, 2, 3, 4].map((i) => (
          <div
            key={i}
            className="h-[108px] animate-pulse rounded-card border border-outline-variant/40 bg-surface-container-low"
          />
        ))}
      </div>
    );
  }

  const { camerasOnline, inferencesPerMin, eventsToday, avgEdgeLatencyMs } = stats;

  return (
    <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 xl:grid-cols-4">
      <StatCard
        label="Cameras Online"
        kpiId="cameras_online"
        value={`${camerasOnline.current}/${camerasOnline.total}`}
        sub={camerasOnline.healthy ? "all healthy" : "check offline feeds"}
        subPositive={camerasOnline.healthy}
        icon="videocam"
      />
      <StatCard
        label="Inferences / min"
        kpiId="inferences_per_min"
        value={inferencesPerMin.value.toLocaleString()}
        sub={inferencesPerMin.trend}
        subPositive={!inferencesPerMin.trend.startsWith("-")}
        icon="memory"
      />
      <StatCard
        label="Events Today"
        kpiId="events_today"
        value={String(eventsToday.value)}
        sub={eventsToday.delta}
        subPositive={eventsToday.value <= Math.max(1, parseInt(eventsToday.delta, 10) || 0)}
        icon="event_note"
      />
      <StatCard
        label="Avg Edge Latency"
        kpiId="avg_edge_latency"
        value={`${avgEdgeLatencyMs.value}`}
        sub={`${avgEdgeLatencyMs.delta} · ms`}
        subPositive={avgEdgeLatencyMs.delta.includes("-") || avgEdgeLatencyMs.delta === "stable"}
        icon="speed"
      />
    </div>
  );
}
