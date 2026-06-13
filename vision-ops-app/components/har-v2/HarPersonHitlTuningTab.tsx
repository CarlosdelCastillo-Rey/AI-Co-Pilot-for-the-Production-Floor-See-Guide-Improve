"use client";

import { useEffect, useState } from "react";
import { HarV2Empty, HarV2Link, HarV2Message, HarV2Panel } from "@/components/har-v2/har-v2-ui";
import {
  fetchHarV2Persons,
  fetchHarV2ReviewQueue,
  fetchHarV2Sessions,
} from "@/lib/har-v2-api";

export function HarPersonHitlTuningTab() {
  const [stats, setStats] = useState<{ sessions: number; persons: number; queue: number } | null>(null);
  const [err, setErr] = useState("");

  useEffect(() => {
    void Promise.all([fetchHarV2Sessions(200), fetchHarV2Persons(500), fetchHarV2ReviewQueue(200)])
      .then(([sessions, persons, queue]) => {
        setStats({ sessions: sessions.length, persons: persons.length, queue: queue.length });
      })
      .catch((e) => setErr(String(e)));
  }, []);

  return (
    <div className="grid gap-6 lg:grid-cols-2">
      <HarV2Panel title="Dataset & registry">
        {stats ? (
          <ul className="space-y-2 text-body-sm text-on-surface-variant">
            <li>
              <strong className="text-on-surface">{stats.sessions}</strong> audit sessions logged
            </li>
            <li>
              <strong className="text-on-surface">{stats.persons}</strong> registry persons (Re-ID embeddings)
            </li>
            <li>
              <strong className="text-on-surface">{stats.queue}</strong> crops waiting in review queue
            </li>
          </ul>
        ) : err ? (
          <HarV2Message tone="error">{err}</HarV2Message>
        ) : (
          <div className="h-16 animate-pulse rounded-md bg-surface-container-low" />
        )}
        <p className="mt-4 text-body-sm text-outline">
          Session tags, registry names, and HITL verdicts feed the next training cycle. Export labels from{" "}
          <code className="text-[11px]">har-research/07_Human_Label_Review.ipynb</code> or the backend SQLite DB.
        </p>
      </HarV2Panel>

      <HarV2Panel title="Re-ID & embeddings">
        <ul className="list-disc space-y-2 pl-5 text-body-sm text-on-surface-variant">
          <li>
            Person identity uses <strong className="text-on-surface">HAR backbone embeddings</strong> (V-JEPA2 /
            DINOv2), not YOLO — no detector retrain needed for names.
          </li>
          <li>
            Cross-session matching: set{" "}
            <code className="text-[11px]">VISIONOPS_HAR_REID_AUTO_LINK=true</code> and tune{" "}
            <code className="text-[11px]">VISIONOPS_HAR_REID_MATCH_THRESHOLD</code> (start ~0.85) in{" "}
            <code className="text-[11px]">vision-ops-backend/.env</code>.
          </li>
          <li>
            Embeddings saved under <code className="text-[11px]">vision-ops-backend/data/har_sessions/</code> on
            label change or uncertain predictions.
          </li>
        </ul>
      </HarV2Panel>

      <HarV2Panel title="Model retraining pipeline" className="lg:col-span-2">
        <p className="text-body-sm text-on-surface-variant">
          Full HAR head and embedding tuning lives in the research track. Use labeled crops and session exports to
          iterate on accuracy.
        </p>
        <ol className="mt-3 list-decimal space-y-2 pl-5 text-body-sm text-on-surface-variant">
          <li>
            <HarV2Link href="/live">Record live sessions and run eval on mock videos</HarV2Link>
          </li>
          <li>Tag persons and confirm uncertain crops in this Person HITL workspace</li>
          <li>
            Retrain in <code className="text-[11px]">har-research/03_Train_HAR_Head.ipynb</code> with new labels
          </li>
          <li>
            Refresh checkpoints in <code className="text-[11px]">har-research/checkpoints/</code> and restart backend
          </li>
        </ol>
        <div className="mt-4 rounded-md border border-outline-variant/50 bg-surface-container-low px-4 py-3 text-body-sm text-outline">
          <p className="font-medium text-on-surface">Production checkpoints</p>
          <p className="mt-1 font-mono text-[11px]">har_vjepa_12c_topdown_allavail · har_dinov2_12c_topdown_allavail</p>
          <p className="mt-2">
            See <HarV2Link href="/analytics">Dashboard</HarV2Link> for plant ops and model performance metrics.
          </p>
        </div>
      </HarV2Panel>

      <HarV2Panel title="Roadmap" className="lg:col-span-2">
        <HarV2Empty>
          Planned: one-click export of HITL labels to har-research, embedding threshold slider, and fine-tune jobs
          from the UI. For now use notebooks under <code className="text-[11px]">har-research/</code>.
        </HarV2Empty>
      </HarV2Panel>
    </div>
  );
}
