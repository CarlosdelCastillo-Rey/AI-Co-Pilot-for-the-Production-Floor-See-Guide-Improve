"use client";

import { Icon } from "@/components/ui/Icon";
import type { TimelineEventApi } from "@/lib/api";

interface EventWorkflowActionsProps {
  event: TimelineEventApi;
  busy: boolean;
  onAcknowledge: (eventId: string) => void;
  onResolve: (event: TimelineEventApi) => void;
  onDismiss?: (eventId: string) => void;
}

function formatAckDuration(event: TimelineEventApi): string | null {
  if (!event.acknowledgedAt || !event.occurredAt) return null;
  const sec = Math.max(
    0,
    Math.round(
      (new Date(event.acknowledgedAt).getTime() - new Date(event.occurredAt).getTime()) / 1000,
    ),
  );
  if (sec < 60) return `ack ${sec}s`;
  const m = Math.floor(sec / 60);
  const s = sec % 60;
  return `ack ${m}m ${String(s).padStart(2, "0")}s`;
}

export function EventWorkflowActions({
  event,
  busy,
  onAcknowledge,
  onResolve,
  onDismiss,
}: EventWorkflowActionsProps) {
  const closed =
    event.resolutionStatus === "RESOLVED" ||
    event.resolutionStatus === "FALSE_POSITIVE";
  const ackLabel = formatAckDuration(event);

  if (closed) {
    return (
      <div className="mt-4 space-y-3">
        <div className="rounded-[10px] border border-[#EBEDF1] bg-[#F5F6F8] px-4 py-3 text-body-sm text-[#5A626C]">
          {event.resolvedBy && (
            <p>
              Closed by <strong className="text-[#0C0F13]">{event.resolvedBy}</strong>
              {event.industrialReasonCode ? ` · ${event.industrialReasonCode}` : ""}
            </p>
          )}
          {(event.downtimeCausedSeconds ?? 0) > 0 && (
            <p>Downtime logged: {Math.round((event.downtimeCausedSeconds ?? 0) / 60)} min</p>
          )}
          {(event.scrapCausedUnits ?? 0) > 0 && <p>Scrap units: {event.scrapCausedUnits}</p>}
          {event.closureNotes && <p className="mt-1 italic">{event.closureNotes}</p>}
        </div>
        {onDismiss && (
          <button
            type="button"
            disabled={busy}
            onClick={() => onDismiss(event.id)}
            className="inline-flex min-h-[40px] items-center gap-2 rounded-[10px] border border-[#EBEDF1] bg-white px-3 font-label text-label-sm font-medium text-[#687079] transition-colors hover:border-[#C0362C]/40 hover:bg-[#FBE7E4] hover:text-[#C0362C] disabled:opacity-50"
          >
            <Icon name="delete" size={16} />
            Remove from panel
          </button>
        )}
      </div>
    );
  }

  return (
    <div className="mt-5 flex flex-wrap items-center gap-2 border-t border-[#EBEDF1] pt-4">
      {event.resolutionStatus === "ACKNOWLEDGED" && event.acknowledgedBy && (
        <span className="mr-auto font-label text-label-sm text-[#687079]">
          Acknowledged by <strong className="text-[#0C0F13]">{event.acknowledgedBy}</strong>
          {ackLabel ? ` · ${ackLabel}` : ""}
        </span>
      )}
      {!event.acknowledgedBy && ackLabel && (
        <span className="mr-auto font-label text-label-sm text-[#687079]">{ackLabel}</span>
      )}
      {event.resolutionStatus === "OPEN" && (
        <button
          type="button"
          disabled={busy}
          onClick={() => onAcknowledge(event.id)}
          className="inline-flex min-h-[44px] items-center gap-2 rounded-[10px] bg-[#0059BB] px-4 font-label text-label-md font-semibold text-white transition-colors hover:bg-[#0070EA] disabled:opacity-50"
        >
          <Icon name="check_circle" size={18} />
          Acknowledge
        </button>
      )}
      {event.resolutionStatus === "ACKNOWLEDGED" && (
        <button
          type="button"
          disabled={busy}
          onClick={() => onResolve(event)}
          className="inline-flex min-h-[44px] items-center gap-2 rounded-[10px] border border-[#C1C6D7] bg-white px-4 font-label text-label-md font-semibold text-[#0059BB] transition-colors hover:bg-[#EEF3FF] disabled:opacity-50"
        >
          <Icon name="sell" size={18} />
          Tag
        </button>
      )}
      <button
        type="button"
        disabled={busy}
        onClick={() => onResolve(event)}
        className="inline-flex min-h-[44px] items-center gap-2 rounded-[10px] border border-[#C1C6D7] bg-white px-4 font-label text-label-md font-semibold text-[#2B3340] transition-colors hover:bg-[#F5F6F8] disabled:opacity-50"
      >
        <Icon name="task_alt" size={18} />
        {event.resolutionStatus === "OPEN" ? "Resolve" : "Resolve / Close"}
      </button>
    </div>
  );
}
