"use client";

import { useCallback, useEffect, useState } from "react";
import { AppShell } from "@/components/layout/AppShell";
import { AddCameraModal } from "@/components/live/AddCameraModal";
import { CameraFeedCard } from "@/components/live/CameraFeedCard";
import {
  CameraFilterPanel,
  type CameraFilters,
} from "@/components/live/CameraFilterPanel";
import { LiveActivityPanel } from "@/components/live/LiveActivityPanel";
import { LiveStatsBar } from "@/components/live/LiveStatsBar";
import { Button } from "@/components/ui/Button";
import { Icon } from "@/components/ui/Icon";
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
  const [gridCols, setGridCols] = useState<1 | 2>(2);
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
    const id = setInterval(() => void load(), 30_000);
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

  return (
    <>
      <AppShell
        searchPlaceholder="Search streams…"
        searchValue={search}
        onSearchChange={setSearch}
        fullBleed
      >
        <div className="flex min-h-[calc(100vh-4rem)] flex-col xl:flex-row">
          <div className="flex-1 overflow-y-auto p-6 lg:p-8">
            <div className="mx-auto max-w-[1600px] space-y-6">
              <header className="flex flex-col gap-4 lg:flex-row lg:items-start lg:justify-between">
                <div>
                  <h2 className="font-headline text-headline-lg text-on-surface">
                    Edge Control Console
                  </h2>
                  <p className="mt-1 text-body-md text-outline">
                    Real-time IP camera feeds with industrial AI inference
                  </p>
                </div>
                <div className="flex flex-wrap items-center gap-2">
                  <Button
                    variant="secondary"
                    icon="grid_view"
                    className={cn("rounded-lg", gridCols === 2 && "ring-1 ring-primary/40")}
                    onClick={() => setGridCols((c) => (c === 2 ? 1 : 2))}
                  >
                    Grid View
                  </Button>
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

              <div
                className={cn(
                  "grid gap-5",
                  gridCols === 2 ? "grid-cols-1 xl:grid-cols-2" : "grid-cols-1",
                )}
              >
                {loading ? (
                  [1, 2, 3].map((i) => (
                    <div
                      key={i}
                      className="aspect-video animate-pulse rounded-card bg-surface-container-low"
                    />
                  ))
                ) : (
                  feeds.map((feed) => <CameraFeedCard key={feed.id} feed={feed} />)
                )}

                {!loading && (
                  <button
                    type="button"
                    onClick={() => setAddOpen(true)}
                    className="flex min-h-[220px] flex-col items-center justify-center rounded-card border-2 border-dashed border-outline-variant/80 bg-surface-container-lowest/50 p-8 transition-colors hover:border-primary hover:bg-primary-fixed/20"
                  >
                    <div className="mb-3 flex h-12 w-12 items-center justify-center rounded-full bg-surface-container-high">
                      <Icon name="add_a_photo" className="text-outline" />
                    </div>
                    <span className="font-label text-label-md font-bold uppercase tracking-wide text-outline">
                      Add Camera Feed
                    </span>
                    <span className="mt-1 font-label text-label-sm text-outline/70">
                      RTSP · ONVIF · webcam
                    </span>
                  </button>
                )}
              </div>
            </div>
          </div>

          <LiveActivityPanel />
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
