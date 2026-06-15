"use client";

import Link from "next/link";
import { useRouter, useSearchParams } from "next/navigation";
import { useCallback, useMemo, useState } from "react";
import {
  HarPersonHitlTab,
  HarV2PageShell,
  HarV2TabBar,
} from "@/components/har-v2/har-v2-ui";
import { HarPersonHitlRegistryTab } from "@/components/har-v2/HarPersonHitlRegistryTab";
import { HarPersonHitlSessionsTab } from "@/components/har-v2/HarPersonHitlSessionsTab";
import { HarPersonHitlTuningTab } from "@/components/har-v2/HarPersonHitlTuningTab";

const TABS: { id: HarPersonHitlTab; label: string; hint: string }[] = [
  {
    id: "sessions",
    label: "Review",
    hint: "Browse sessions, tag tracks, and review uncertain crops",
  },
  {
    id: "registry",
    label: "Registry",
    hint: "Stable person identities, rename, merge duplicates, unlink false merges",
  },
  {
    id: "tuning",
    label: "Tuning",
    hint: "Re-ID settings, dataset stats, retraining pipeline",
  },
];

function parseTab(raw: string | null): HarPersonHitlTab {
  if (raw === "registry" || raw === "tuning") return raw;
  // "queue" and "sessions" both map to the merged review tab
  return "sessions";
}

export function HarPersonHitlPageClient() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const tab = parseTab(searchParams.get("tab"));
  const rawTab = searchParams.get("tab");
  const personParam = searchParams.get("person") ?? undefined;
  const [registryPersonId, setRegistryPersonId] = useState<string | undefined>(personParam);

  const setTab = useCallback(
    (next: HarPersonHitlTab, person?: string) => {
      const params = new URLSearchParams();
      params.set("tab", next);
      if (person) params.set("person", person);
      router.replace(`/har-hitl?${params.toString()}`);
      if (person) setRegistryPersonId(person);
    },
    [router],
  );

  const openRegistry = useCallback(
    (globalPersonId: string) => {
      setTab("registry", globalPersonId);
    },
    [setTab],
  );

  const activePersonId = useMemo(
    () => (tab === "registry" ? personParam ?? registryPersonId : undefined),
    [tab, personParam, registryPersonId],
  );

  return (
    <HarV2PageShell
      title="Person HITL"
      description="Detect persons, review sessions track-by-track, maintain the Re-ID registry, confirm uncertain predictions, and plan embedding/model tuning — one workspace."
      action={
        <Link
          href="/live"
          className="rounded-md border border-primary/40 px-3 py-2 text-body-sm text-primary hover:bg-primary/10"
        >
          Open Live Streams
        </Link>
      }
    >
      <HarV2TabBar tabs={TABS} active={tab} onChange={(id) => setTab(id)} />

      <div className="mt-6">
        {tab === "sessions" ? (
          <HarPersonHitlSessionsTab
            onOpenRegistry={openRegistry}
            initialFocus={rawTab === "queue" ? "queue" : "session"}
          />
        ) : null}
        {tab === "registry" ? <HarPersonHitlRegistryTab initialPersonId={activePersonId} /> : null}
        {tab === "tuning" ? <HarPersonHitlTuningTab /> : null}
      </div>
    </HarV2PageShell>
  );
}
