"use client";

import { useCallback, useEffect, useState } from "react";
import { AppShell } from "@/components/layout/AppShell";
import { AddCameraModal } from "@/components/live/AddCameraModal";
import { CameraFeedCard, isHarCamera } from "@/components/live/CameraFeedCard";
import { HarCameraRow } from "@/components/live/HarCameraRow";
import {
  CameraFilterPanel,
  type CameraFilters,
} from "@/components/live/CameraFilterPanel";
import { LiveStatsBar } from "@/components/live/LiveStatsBar";
import { Button } from "@/components/ui/Button";
import {
  createCamera,
  fetchLiveStats,
  getLiveCameraFeeds,
} from "@/lib/api";
import type { CameraFeed, LiveStats } from "@/lib/types";
import { cn } from "@/lib/cn";

export function LivePageClient() {
  const [search, setSearch] = useState("");
  const [feeds, setFeeds] = useState<CameraFeed[]>([]);
  const [stats, setStats] = useState<LiveStats | null>(null);
  const [loading, setLoading] = useState(true);
  const [filterOpen, setFilterOpen] = useState(false);
  const [addOpen, setAddOpen] = useState(false);
  const [filters, setFilters] = useState<CameraFilters>({ status: "", model: "" });

  const load = useCallback(async () => {
    const [cameraData, statsData] = await Promise.all([
      getLiveCameraFeeds({
        status: filters.status || undefined,
        model: filters.model || undefined,
        q: search.trim() || undefined,
      }),
      fetchLiveStats(),
    ]);
    setFeeds(cameraData);
    setStats(statsData);
    setLoading(false);
  }, [filters.model, filters.status, search]);

  useEffect(() => {
    void load();
    const id = setInterval(() => void load(), 8_000);
    return () => clearInterval(id);
  }, [load]);

  const handleAddCamera = async (input: Parameters<typeof createCamera>[0]) => {
    const created = await createCamera(input);
    if (created) {
      await load();
      return true;
    }
    return false;
  };

  const activeFilterCount = Number(Boolean(filters.status)) + Number(Boolean(filters.model));
  const harFeeds = feeds.filter(isHarCamera);
  const otherFeeds = feeds.filter((f) => !isHarCamera(f));

  return (
    <>
      <AppShell
        searchPlaceholder="Search streams…"
        searchValue={search}
        onSearchChange={setSearch}
        fullBleed
      >
        <div className="min-h-[calc(100vh-4rem)] overflow-y-auto p-6 lg:p-8">
          <div className="mx-auto max-w-[1400px] space-y-6">
            <header className="flex flex-col gap-4 lg:flex-row lg:items-start lg:justify-between">
              <div>
                <h2 className="font-headline text-headline-lg text-on-surface">
                  Edge Control Console
                </h2>
                <p className="mt-1 text-body-md text-outline">
                  One HAR model per row — video on the left, predictions and logs on the right
                </p>
              </div>
              <div className="flex flex-wrap items-center gap-2">
                <div className="relative">
                  <Button
                    variant="ghost"
                    icon="filter_list"
                    className={cn(
                      "rounded-lg border border-outline-variant/70 bg-surface-container-lowest",
                      activeFilterCount > 0 && "border-primary text-primary",
                    )}
                    onClick={() => setFilterOpen((o) => !o)}
                  >
                    Filter{activeFilterCount > 0 ? ` (${activeFilterCount})` : ""}
                  </Button>
                  <CameraFilterPanel
                    open={filterOpen}
                    filters={filters}
                    onChange={setFilters}
                    onClose={() => {
                      setFilterOpen(false);
                      void load();
                    }}
                  />
                </div>
                <Button icon="add_a_photo" className="rounded-lg" onClick={() => setAddOpen(true)}>
                  Add Camera
                </Button>
              </div>
            </header>

            <LiveStatsBar stats={stats} loading={loading} />

            <div className="space-y-6">
              {loading ? (
                [1, 2, 3].map((i) => (
                  <div
                    key={i}
                    className="h-[320px] animate-pulse rounded-card bg-surface-container-low"
                  />
                ))
              ) : (
                <>
                  {harFeeds.map((feed) => (
                    <HarCameraRow key={feed.id} feed={feed} />
                  ))}
                  {otherFeeds.length > 0 && (
                    <div className="space-y-4">
                      <h3 className="font-label text-label-md font-bold uppercase tracking-wide text-outline">
                        Other feeds
                      </h3>
                      {otherFeeds.map((feed) => (
                        <CameraFeedCard key={feed.id} feed={feed} />
                      ))}
                    </div>
                  )}
                </>
              )}
            </div>
          </div>
        </div>
      </AppShell>

      <AddCameraModal
        open={addOpen}
        onClose={() => setAddOpen(false)}
        onSubmit={handleAddCamera}
      />
    </>
  );
}
