"use client";

import { Icon } from "@/components/ui/Icon";
import type { ShiftAiSummaryApi } from "@/lib/api";
import { cn } from "@/lib/cn";

interface ShiftAiSummaryPanelProps {
  aiSummary: ShiftAiSummaryApi | null;
  loading?: boolean;
}

const statusStyles = {
  all_clear: {
    wrap: "border-[#CDE6D8] bg-[#E6F4EC]",
    dot: "bg-[#1F8A5B]",
    title: "text-[#1F8A5B]",
    sub: "text-[#2F7A58]",
  },
  action_needed: {
    wrap: "border-[#ECDCB5] bg-[#FBF0DB]",
    dot: "bg-[#B7791F]",
    title: "text-[#B7791F]",
    sub: "text-[#8A6524]",
  },
  critical: {
    wrap: "border-[#E8C4BF] bg-[#FBE7E4]",
    dot: "bg-[#C0362C]",
    title: "text-[#C0362C]",
    sub: "text-[#9A3D34]",
  },
};

function ListBlock({ title, items }: { title: string; items: string[] }) {
  if (!items.length) return null;
  return (
    <div className="rounded-[10px] border border-[#EBEDF1] bg-white p-3">
      <p className="mb-2 font-label text-[10px] font-bold uppercase tracking-wide text-[#9AA1AB]">
        {title}
      </p>
      <ul className="space-y-1.5 text-[12px] leading-snug text-[#5A626C]">
        {items.map((item, index) => (
          <li key={`${title}-${index}`} className="flex gap-2">
            <span className="mt-1.5 h-1 w-1 shrink-0 rounded-full bg-[#0059BB]" />
            <span>{item}</span>
          </li>
        ))}
      </ul>
    </div>
  );
}

export function ShiftAiSummaryPanel({ aiSummary, loading }: ShiftAiSummaryPanelProps) {
  if (loading) {
    return (
      <div className="mb-4 animate-pulse rounded-[14px] border border-[#EBEDF1] bg-[#F5F6F8] p-4">
        <div className="mb-3 h-4 w-32 rounded bg-[#EBEDF1]" />
        <div className="h-16 rounded bg-[#EBEDF1]" />
      </div>
    );
  }

  if (!aiSummary) return null;

  const styles = statusStyles[aiSummary.currentStatus];

  return (
    <div className="rounded-[14px] border border-[#EBEDF1] bg-[#F5F6F8] p-4">
      <div className="mb-3 flex items-center gap-2">
        <Icon name="auto_awesome" size={18} className="text-[#0059BB]" />
        <h3 className="font-headline text-[14px] font-semibold text-[#0C0F13]">VisionOps AI Summary</h3>
      </div>

      <div className={cn("mb-3 flex items-start gap-2 rounded-[10px] border px-3 py-2.5", styles.wrap)}>
        <span className={cn("mt-1.5 h-2 w-2 shrink-0 rounded-full", styles.dot)} />
        <div className="min-w-0 flex-1">
          <p className={cn("font-headline text-[13px] font-bold", styles.title)}>
            {aiSummary.statusHeadline}
          </p>
          <p className={cn("text-[12px]", styles.sub)}>{aiSummary.statusDetail}</p>
        </div>
      </div>

      <p className="mb-3 text-[12px] leading-relaxed text-[#5A626C]">{aiSummary.narrative}</p>

      <div className="mb-3 grid grid-cols-2 gap-2 font-label text-[11px]">
        <div className="rounded-[8px] border border-[#EBEDF1] bg-white px-2.5 py-2">
          <p className="text-[#9AA1AB]">OEE</p>
          <p className="font-semibold text-[#0C0F13]">{aiSummary.metrics.oee.toFixed(1)}%</p>
        </div>
        <div className="rounded-[8px] border border-[#EBEDF1] bg-white px-2.5 py-2">
          <p className="text-[#9AA1AB]">Flow</p>
          <p className="font-semibold text-[#0C0F13]">
            {aiSummary.metrics.flowEfficiency.toFixed(1)}%
          </p>
        </div>
        <div className="rounded-[8px] border border-[#EBEDF1] bg-white px-2.5 py-2">
          <p className="text-[#9AA1AB]">COQ</p>
          <p className="font-semibold text-[#0C0F13]">
            ${aiSummary.metrics.coqTotalUsd.toLocaleString(undefined, { maximumFractionDigits: 0 })}
          </p>
        </div>
        <div className="rounded-[8px] border border-[#EBEDF1] bg-white px-2.5 py-2">
          <p className="text-[#9AA1AB]">Uptime</p>
          <p className="font-semibold text-[#0059BB]">{aiSummary.metrics.uptime}</p>
        </div>
      </div>

      <div className="space-y-2">
        <ListBlock title="Highlights" items={aiSummary.highlights} />
        <ListBlock title="Suggestions" items={aiSummary.suggestions} />
        <ListBlock title="Recommendations" items={aiSummary.recommendations} />
      </div>
    </div>
  );
}
