import { AppShell } from "@/components/layout/AppShell";
import { CameraFeedCard } from "@/components/live/CameraFeedCard";
import { RealtimeEventsPanel } from "@/components/live/RealtimeEventsPanel";
import { Button } from "@/components/ui/Button";
import { Icon } from "@/components/ui/Icon";
import { CAMERA_FEEDS } from "@/lib/mock-data";

export default function LivePage() {
  return (
    <AppShell searchPlaceholder="Search streams..." fullBleed>
      <div className="flex min-h-[calc(100vh-4rem)]">
        <div className="flex-1 overflow-y-auto bg-surface-container-low p-lg">
          <div className="mx-auto max-w-[1600px]">
            <div className="mb-lg flex items-center justify-between">
              <div>
                <h2 className="font-headline text-headline-md text-on-surface">
                  Edge Control Console
                </h2>
                <p className="text-body-sm text-outline">
                  Real-time IP camera feeds with industrial AI inference
                </p>
              </div>
              <div className="flex gap-2">
                <Button variant="secondary" icon="grid_view">
                  Grid View
                </Button>
                <Button variant="ghost" icon="filter_list" className="border border-outline-variant bg-surface-container-lowest">
                  Filter
                </Button>
              </div>
            </div>
            <div className="grid grid-cols-1 gap-gutter xl:grid-cols-2">
              {CAMERA_FEEDS.map((feed) => (
                <CameraFeedCard key={feed.id} feed={feed} />
              ))}
              <button
                type="button"
                className="flex min-h-[200px] cursor-pointer flex-col items-center justify-center rounded-lg border-2 border-dashed border-outline-variant p-xl transition-colors hover:bg-surface-container-high"
              >
                <div className="mb-4 flex h-12 w-12 items-center justify-center rounded-full bg-surface-container-highest transition-transform group-hover:scale-110">
                  <Icon name="add_a_photo" className="text-outline" />
                </div>
                <span className="text-label-md text-outline">Add Camera Feed</span>
              </button>
            </div>
          </div>
        </div>
        <RealtimeEventsPanel />
      </div>
      <button
        type="button"
        className="fixed bottom-8 right-8 z-50 flex items-center gap-3 rounded-full bg-primary py-4 pl-4 pr-6 text-on-primary shadow-lg transition-all hover:bg-primary-container active:scale-95"
      >
        <Icon name="add" />
        <span className="text-label-md font-bold">Add New Camera</span>
      </button>
    </AppShell>
  );
}
