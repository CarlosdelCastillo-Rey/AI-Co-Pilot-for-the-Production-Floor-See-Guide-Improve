"use client";

import { useCallback, useEffect, useState } from "react";
import Link from "next/link";
import { Button } from "@/components/ui/Button";
import {
  fetchVisionStatus,
  runVisionProbe,
  type VisionStatus,
} from "@/lib/api";

export function VisionLabPanel() {
  const [status, setStatus] = useState<VisionStatus | null>(null);
  const [loading, setLoading] = useState<string | null>(null);
  const [message, setMessage] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    setStatus(await fetchVisionStatus());
  }, []);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  async function probe(cameraId: string, setBaseline = false) {
    setLoading(cameraId);
    setMessage(null);
    const result = await runVisionProbe(cameraId, { setBaseline });
    setLoading(null);
    setMessage(result.message);
    if (result.ok) {
      await refresh();
    }
  }

  const cam01 = status?.cameras?.["cam-01"];
  const cam02 = status?.cameras?.["cam-02"];

  return (
    <div className="mx-auto max-w-3xl space-y-lg">
      <p className="text-body-md text-outline">
        Batch probes for mock industrial cameras.{" "}
        <strong className="text-on-surface">cam-01</strong> runs a DINO-style spatial heatmap (D2).{" "}
        <strong className="text-on-surface">cam-02</strong> runs V-JEPA or motion embedding + anomaly
        score (V1). Results appear on{" "}
        <Link href="/live" className="text-primary underline">
          Live Streams
        </Link>
        .
      </p>

      {message ? (
        <p className="rounded-lg border border-outline-variant bg-surface-container-low px-md py-sm text-body-sm text-on-surface">
          {message}
        </p>
      ) : null}

      <section className="rounded-lg border border-outline-variant bg-surface-container-lowest p-lg">
        <h3 className="text-label-md text-on-surface">Camera 01 — Assembly (DINO heatmap)</h3>
        <p className="mt-1 text-body-sm text-outline">{cam01?.label ?? "D2 patch similarity / DINOv2 HF"}</p>
        <div className="mt-md flex flex-wrap gap-2">
          <Button
            variant="primary"
            disabled={loading !== null}
            onClick={() => void probe("cam-01")}
          >
            {loading === "cam-01" ? "Running…" : "Run heatmap probe"}
          </Button>
        </div>
        {cam01?.heatmapUrl ? (
          <div className="mt-md overflow-hidden rounded-lg border border-outline-variant">
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img src={cam01.heatmapUrl} alt="DINO heatmap overlay" className="w-full" />
          </div>
        ) : null}
      </section>

      <section className="rounded-lg border border-outline-variant bg-surface-container-lowest p-lg">
        <h3 className="text-label-md text-on-surface">Camera 02 — Warehouse (V-JEPA)</h3>
        <p className="mt-1 text-body-sm text-outline">
          {cam02?.label ?? "V1 embedding + anomaly"}
          {cam02?.anomaly_score != null ? ` — score ${cam02.anomaly_score}` : ""}
          {cam02?.last_severity ? ` (${cam02.last_severity})` : ""}
        </p>
        <div className="mt-md flex flex-wrap gap-2">
          <Button
            variant="primary"
            disabled={loading !== null}
            onClick={() => void probe("cam-02", true)}
          >
            {loading === "cam-02" ? "Running…" : "Set baseline + probe"}
          </Button>
          <Button
            variant="secondary"
            disabled={loading !== null}
            onClick={() => void probe("cam-02")}
          >
            Probe again (compare)
          </Button>
        </div>
      </section>

      <p className="text-body-sm text-outline">
        Optional: <code className="text-label-sm">pip install torch transformers</code> in the
        backend venv for Hugging Face DINOv2 / V-JEPA weights. Without them, demos use patch
        similarity and motion fallback.
      </p>
    </div>
  );
}
