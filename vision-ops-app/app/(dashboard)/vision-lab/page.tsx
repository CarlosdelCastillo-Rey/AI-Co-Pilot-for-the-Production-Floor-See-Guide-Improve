import { AppShell } from "@/components/layout/AppShell";
import { VisionLabPanel } from "@/components/vision/VisionLabPanel";

export default function VisionLabPage() {
  return (
    <AppShell searchPlaceholder="Vision models…">
      <div className="p-lg">
        <header className="mb-lg">
          <h2 className="font-headline text-headline-md text-on-surface">Vision Lab</h2>
          <p className="text-body-sm text-outline">
            DINOv3 / V-JEPA probes on industrial mock cameras (batch, not real-time)
          </p>
        </header>
        <VisionLabPanel />
      </div>
    </AppShell>
  );
}
