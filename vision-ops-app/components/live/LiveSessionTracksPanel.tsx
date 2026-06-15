"use client";

import Link from "next/link";
import { HarV2Badge, HarV2Link } from "@/components/har-v2/har-v2-ui";
import type { HarTrackPrediction } from "@/lib/api";
import { cn } from "@/lib/cn";

type Props = {
  playing: boolean;
  sessionId: string | null | undefined;
  videoName: string;
  modelLabel: string;
  tracks: HarTrackPrediction[];
  accent: string;
  className?: string;
  /** Fill available column height — track list scrolls inside */
  fillHeight?: boolean;
};

export function LiveSessionTracksPanel({
  playing,
  sessionId,
  videoName,
  modelLabel,
  tracks,
  accent,
  className,
  fillHeight = false,
}: Props) {
  const sorted = [...tracks].sort((a, b) => a.track_id - b.track_id);

  return (
    <div
      className={cn(
        "p-4",
        !fillHeight && "rounded-lg border border-outline-variant/50 bg-surface-container-low",
        fillHeight && "flex h-full min-h-0 flex-col",
        className,
      )}
    >
      <div className="flex flex-wrap items-start justify-between gap-2">
        <div>
          <p className="font-label text-[10px] font-bold uppercase tracking-wider text-outline">
            Session & person tracks
          </p>
          <p className="mt-1 text-[11px] text-outline">
            {modelLabel} · {videoName || "—"}
          </p>
        </div>
        <HarV2Link href="/har-hitl">Person HITL</HarV2Link>
      </div>

      <div
        className={cn(
          "mt-3 flex flex-wrap items-center gap-2 rounded-md border px-3 py-2 text-body-sm",
          playing
            ? "border-success/40 bg-success/5 text-on-surface"
            : "border-outline-variant/40 bg-surface-container-lowest text-outline",
        )}
      >
        <span
          className={cn(
            "h-2 w-2 rounded-full",
            playing ? "animate-pulse bg-success" : "bg-outline",
          )}
        />
        {playing ? (
          <>
            <span className="font-semibold text-success">Recording</span>
            {sessionId ? (
              <>
                <span className="text-outline">·</span>
                <span className="font-mono text-[11px]">{sessionId}</span>
              </>
            ) : (
              <span className="text-outline">Starting session…</span>
            )}
          </>
        ) : (
          <span>
            Paused — press play to record per-person tracks, crops, and Re-ID to the audit session.
          </span>
        )}
      </div>

      {sessionId && !playing ? (
        <p className="mt-2 text-body-sm text-outline">
          Last session{" "}
          <span className="font-mono text-[11px] text-on-surface">{sessionId}</span> —{" "}
          <Link href="/har-hitl" className="text-primary hover:underline">
            open in review
          </Link>
        </p>
      ) : null}

      <div className={cn("mt-4", fillHeight && "flex min-h-0 flex-1 flex-col")}>
        <p className="font-label text-[10px] font-bold uppercase tracking-wider text-outline">
          Tracked persons ({sorted.length})
        </p>
        {sorted.length === 0 ? (
          <p className="mt-2 text-body-sm text-outline">
            {playing
              ? "Waiting for ByteTrack detections and HAR inference…"
              : "No tracks recorded yet."}
          </p>
        ) : (
          <ul
            className={cn(
              "mt-2 space-y-2 overflow-y-auto overscroll-y-contain",
              fillHeight ? "min-h-0 flex-1" : "max-h-64",
            )}
          >
            {sorted.map((tr) => (
              <li
                key={tr.track_id}
                className="rounded-md border border-outline-variant/40 bg-surface-container-lowest px-3 py-2 text-body-sm"
              >
                <div className="flex flex-wrap items-center gap-2">
                  <span className="font-mono text-label-sm font-semibold text-on-surface">
                    {tr.display_name?.trim() || `#${tr.track_id}`}
                  </span>
                  {tr.display_name?.trim() ? (
                    <span className="font-mono text-[10px] text-outline">#{tr.track_id}</span>
                  ) : null}
                  {tr.global_person_id ? (
                    <HarV2Badge>{tr.global_person_id}</HarV2Badge>
                  ) : null}
                  {tr.inferring ? (
                    <span className="text-[11px] text-outline">analyzing…</span>
                  ) : null}
                </div>
                <div className="mt-0.5 text-on-surface">
                  {tr.action_label ?? "detecting…"}
                  {tr.action_confidence != null && !tr.inferring ? (
                    <span style={{ color: accent }}>
                      {" "}
                      · {Math.round(tr.action_confidence * 100)}%
                    </span>
                  ) : null}
                </div>
                {tr.n_frames != null ? (
                  <div className="text-[11px] text-outline">{tr.n_frames} frames in buffer</div>
                ) : null}
              </li>
            ))}
          </ul>
        )}
      </div>
    </div>
  );

}
