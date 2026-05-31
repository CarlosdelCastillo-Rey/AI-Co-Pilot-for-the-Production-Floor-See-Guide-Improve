"use client";

import { useEffect, useState } from "react";
import { Button } from "@/components/ui/Button";
import { ResolutionStatusBadge } from "@/components/timeline/TimelineBadges";
import type { ReasonCodeApi, TimelineEventApi } from "@/lib/api";
import { cn } from "@/lib/cn";

interface EventResolveModalProps {
  event: TimelineEventApi;
  reasonCodes: ReasonCodeApi[];
  open: boolean;
  onClose: () => void;
  onSubmit: (payload: {
    status: "RESOLVED" | "FALSE_POSITIVE";
    reasonCode?: string;
    downtimeMinutes: number;
    scrapUnits: number;
    notes: string;
  }) => Promise<void>;
}

export function EventResolveModal({
  event,
  reasonCodes,
  open,
  onClose,
  onSubmit,
}: EventResolveModalProps) {
  const [status, setStatus] = useState<"RESOLVED" | "FALSE_POSITIVE">("RESOLVED");
  const [reasonCode, setReasonCode] = useState("");
  const [downtimeMinutes, setDowntimeMinutes] = useState(0);
  const [scrapUnits, setScrapUnits] = useState(0);
  const [notes, setNotes] = useState("");
  const [saving, setSaving] = useState(false);

  useEffect(() => {
    if (!open) return;
    setStatus("RESOLVED");
    setReasonCode("");
    setDowntimeMinutes(0);
    setScrapUnits(0);
    setNotes("");
  }, [open, event.id]);

  if (!open) return null;

  const handleSubmit = async () => {
    setSaving(true);
    try {
      await onSubmit({
        status,
        reasonCode: status === "FALSE_POSITIVE" ? "FALSE_POS" : reasonCode || undefined,
        downtimeMinutes,
        scrapUnits,
        notes,
      });
      onClose();
    } finally {
      setSaving(false);
    }
  };

  return (
    <div className="fixed inset-0 z-50 flex items-end justify-center bg-black/50 p-4 sm:items-center">
      <div className="w-full max-w-lg rounded-card border border-outline-variant bg-surface-container-lowest p-6 shadow-overlay">
        <h3 className="mb-1 font-headline text-headline-md">Close Incident</h3>
        <p className="mb-6 text-body-sm text-on-surface-variant">{event.title}</p>

        <div className="mb-4 flex gap-2">
          {(["RESOLVED", "FALSE_POSITIVE"] as const).map((s) => (
            <button
              key={s}
              type="button"
              onClick={() => setStatus(s)}
              className={cn(
                "min-h-touch flex-1 rounded-lg border px-3 text-label-md font-bold",
                status === s ? "border-primary bg-primary-fixed/30 text-primary" : "border-outline-variant",
              )}
            >
              {s === "RESOLVED" ? "Resolve" : "False Positive"}
            </button>
          ))}
        </div>

        {status === "RESOLVED" && (
          <>
            <label className="mb-2 block text-label-sm uppercase text-outline">Root Cause</label>
            <select
              value={reasonCode}
              onChange={(e) => setReasonCode(e.target.value)}
              className="mb-4 min-h-touch w-full rounded-lg border border-outline-variant bg-surface px-3 text-body-md"
            >
              <option value="">Select reason code…</option>
              {reasonCodes
                .filter((r) => r.code !== "FALSE_POS")
                .map((r) => (
                  <option key={r.code} value={r.code}>
                    {r.label} ({r.category})
                  </option>
                ))}
            </select>

            <div className="mb-4 grid grid-cols-2 gap-3">
              <div>
                <label className="mb-2 block text-label-sm uppercase text-outline">Downtime (min)</label>
                <input
                  type="number"
                  min={0}
                  value={downtimeMinutes}
                  onChange={(e) => setDowntimeMinutes(Number(e.target.value))}
                  className="min-h-touch w-full rounded-lg border border-outline-variant bg-surface px-3 text-body-md"
                />
              </div>
              <div>
                <label className="mb-2 block text-label-sm uppercase text-outline">Scrap Units</label>
                <input
                  type="number"
                  min={0}
                  value={scrapUnits}
                  onChange={(e) => setScrapUnits(Number(e.target.value))}
                  className="min-h-touch w-full rounded-lg border border-outline-variant bg-surface px-3 text-body-md"
                />
              </div>
            </div>
          </>
        )}

        <label className="mb-2 block text-label-sm uppercase text-outline">Closure Notes</label>
        <textarea
          value={notes}
          onChange={(e) => setNotes(e.target.value)}
          rows={3}
          className="mb-6 w-full rounded-lg border border-outline-variant bg-surface px-3 py-2 text-body-md"
          placeholder="Optional audit notes…"
        />

        <div className="flex gap-3">
          <Button variant="secondary" className="min-h-touch-critical flex-1" onClick={onClose}>
            Cancel
          </Button>
          <Button
            className="min-h-touch-critical flex-1"
            disabled={saving || (status === "RESOLVED" && !reasonCode)}
            onClick={() => void handleSubmit()}
          >
            {saving ? "Saving…" : "Close Incident"}
          </Button>
        </div>
      </div>
    </div>
  );
}
