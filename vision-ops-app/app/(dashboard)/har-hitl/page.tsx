import { Suspense } from "react";
import { HarPersonHitlPageClient } from "@/components/har-v2/HarPersonHitlPageClient";

export default function HarHitlPage() {
  return (
    <Suspense fallback={null}>
      <HarPersonHitlPageClient />
    </Suspense>
  );
}
