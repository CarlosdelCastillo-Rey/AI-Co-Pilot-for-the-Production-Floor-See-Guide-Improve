"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useCallback, useEffect, useState } from "react";
import { useAuth } from "@/components/auth/AuthProvider";
import { Icon } from "@/components/ui/Icon";
import { resetDynamicData } from "@/lib/api";
import { DEFAULT_ROUTE, NAV_ITEMS, type NavId } from "@/lib/navigation";
import { currentShiftLabel, initialsFromName } from "@/lib/shift";
import { cn } from "@/lib/cn";

export const DATA_RESET_EVENT = "visionops:data-reset";

function defaultNavId(): NavId {
  const href = DEFAULT_ROUTE.replace(/^\//, "");
  const match = NAV_ITEMS.find((item) => item.href === `/${href}`);
  return match?.id ?? "analytics";
}

function activeNavId(pathname: string): NavId {
  const sorted = [...NAV_ITEMS].sort((a, b) => b.href.length - a.href.length);
  const match = sorted.find(
    (item) => pathname === item.href || pathname.startsWith(`${item.href}/`),
  );
  return match?.id ?? defaultNavId();
}

export function Sidebar() {
  const pathname = usePathname();
  const active = activeNavId(pathname);
  const { user } = useAuth();
  const shift = currentShiftLabel();
  const [confirmOpen, setConfirmOpen] = useState(false);
  const [resetting, setResetting] = useState(false);
  const [resetMessage, setResetMessage] = useState<string | null>(null);

  const handleReset = useCallback(async () => {
    setResetting(true);
    setResetMessage(null);
    const result = await resetDynamicData();
    setResetting(false);
    if (!result) {
      setResetMessage("Reset failed — sign in and try again.");
      return;
    }
    setConfirmOpen(false);
    setResetMessage(`Cleared ${result.totalRowsCleared} rows. Fresh start ready.`);
    window.dispatchEvent(new CustomEvent(DATA_RESET_EVENT));
  }, []);

  useEffect(() => {
    if (!resetMessage) return;
    const id = window.setTimeout(() => setResetMessage(null), 5000);
    return () => window.clearTimeout(id);
  }, [resetMessage]);

  return (
    <aside className="fixed left-0 top-0 z-50 flex h-screen w-[240px] flex-col border-r border-outline-variant bg-surface-container-lowest px-4 py-6">
      <div className="mb-10 flex items-center gap-3 px-2">
        <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-primary">
          <Icon
            name="precision_manufacturing"
            filled
            className="text-on-primary"
            size={20}
          />
        </div>
        <div>
          <h1 className="font-headline text-[18px] font-bold leading-none text-on-surface">
            VisionOps
          </h1>
          <p className="mt-1 text-label-sm uppercase tracking-widest text-outline">
            Industrial AI
          </p>
        </div>
      </div>

      <nav className="flex flex-1 flex-col gap-0.5">
        {NAV_ITEMS.map((item) => {
          const isActive = active === item.id;
          return (
            <Link
              key={item.id}
              href={item.href}
              className={cn(
                "flex items-center gap-3 rounded-lg px-3 py-2.5 transition-colors duration-150",
                isActive
                  ? "border-l-[3px] border-primary bg-primary-fixed/40 font-semibold text-primary pl-[9px]"
                  : "font-medium text-on-surface-variant hover:bg-surface-container-low",
              )}
            >
              <Icon name={item.icon} filled={isActive} size={20} />
              <span className="text-body-sm">{item.label}</span>
            </Link>
          );
        })}
      </nav>

      {user && (
        <div className="mb-4 px-1">
          <button
            type="button"
            onClick={() => setConfirmOpen(true)}
            disabled={resetting}
            className="flex w-full items-center gap-3 rounded-lg border border-outline-variant/70 px-3 py-2.5 text-left transition-colors hover:bg-surface-container-low disabled:opacity-50"
          >
            <Icon name="restart_alt" size={20} className="text-outline" />
            <span className="text-body-sm font-medium text-on-surface-variant">Fresh start</span>
          </button>
          {resetMessage ? (
            <p className="mt-2 px-1 text-label-sm text-success">{resetMessage}</p>
          ) : null}
        </div>
      )}

      {confirmOpen && (
        <>
          <button
            type="button"
            aria-label="Close reset dialog"
            className="fixed inset-0 z-[200] bg-black/30"
            onClick={() => !resetting && setConfirmOpen(false)}
          />
          <div className="fixed bottom-6 left-6 z-[210] w-[min(22rem,calc(100vw-3rem))] rounded-xl border border-outline-variant bg-surface-container-lowest p-5 shadow-lg">
            <h2 className="font-headline text-body-md font-semibold text-on-surface">Clear shift data?</h2>
            <p className="mt-2 text-body-sm text-on-surface-variant">
              Removes events, alert deliveries, analytics rollups, and HAR activity logs. Cameras,
              users, and alert rules are kept.
            </p>
            <div className="mt-4 flex justify-end gap-2">
              <button
                type="button"
                disabled={resetting}
                onClick={() => setConfirmOpen(false)}
                className="rounded-lg px-3 py-2 text-body-sm font-medium text-on-surface-variant hover:bg-surface-container-low disabled:opacity-50"
              >
                Cancel
              </button>
              <button
                type="button"
                disabled={resetting}
                onClick={() => void handleReset()}
                className="rounded-lg bg-primary px-3 py-2 text-body-sm font-semibold text-on-primary hover:bg-primary/90 disabled:opacity-50"
              >
                {resetting ? "Clearing…" : "Clear data"}
              </button>
            </div>
          </div>
        </>
      )}

      {user && (
        <div className="mt-auto border-t border-outline-variant/60 px-2 pt-6">
          <div className="flex items-center gap-3">
            <div className="flex h-9 w-9 shrink-0 items-center justify-center rounded-full bg-primary-fixed text-primary">
              <span className="font-label text-label-sm font-bold">
                {initialsFromName(user.name)}
              </span>
            </div>
            <div className="min-w-0">
              <p className="truncate text-body-sm font-semibold text-on-surface">{user.name}</p>
              <p className="truncate font-label text-label-sm text-outline">
                {user.role ?? "Supervisor"} · {shift}
              </p>
            </div>
          </div>
        </div>
      )}
    </aside>
  );
}
