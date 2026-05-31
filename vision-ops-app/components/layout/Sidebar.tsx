"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useAuth } from "@/components/auth/AuthProvider";
import { Icon } from "@/components/ui/Icon";
import { NAV_ITEMS, type NavId } from "@/lib/navigation";
import { currentShiftLabel, initialsFromName } from "@/lib/shift";
import { cn } from "@/lib/cn";

function activeNavId(pathname: string): NavId {
  if (pathname.startsWith("/vision-lab")) return "vision";
  if (pathname.startsWith("/identity")) return "identity";
  if (pathname.startsWith("/analytics")) return "analytics";
  if (pathname.startsWith("/timeline")) return "timeline";
  if (pathname.startsWith("/alerts")) return "alerts";
  return "live";
}

export function Sidebar() {
  const pathname = usePathname();
  const active = activeNavId(pathname);
  const { user } = useAuth();
  const shift = currentShiftLabel();

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
