"use client";

import { HarModelsPanel } from "@/components/vision/HarModelsPanel";

export function VisionLabPanel() {
  return (
    <div className="mx-auto max-w-4xl space-y-lg">
      <p className="text-body-md text-outline">
        HAR activity models run on clips from{" "}
        <code className="text-label-sm">vision-ops-app/public/mock-videos</code> — one random video
        per camera. Probe summaries appear in the table below after you run a model.
      </p>
      <HarModelsPanel />
    </div>
  );
}
