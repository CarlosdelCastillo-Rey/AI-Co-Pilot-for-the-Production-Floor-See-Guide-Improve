"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { Icon } from "@/components/ui/Icon";
import { NAV_ITEMS, type NavId } from "@/lib/navigation";
import { cn } from "@/lib/cn";

function activeNavId(pathname: string): NavId {
  if (pathname.startsWith("/identity")) return "identity";
  if (pathname.startsWith("/analytics")) return "analytics";
  if (pathname.startsWith("/timeline")) return "timeline";
  if (pathname.startsWith("/alerts")) return "alerts";
  return "live";
}

export function Sidebar() {
  const pathname = usePathname();
  const active = activeNavId(pathname);

  return (
    <aside className="fixed left-0 top-0 z-50 flex h-screen w-[240px] flex-col border-r border-outline-variant bg-inverse-surface px-4 py-6">
      <div className="mb-10 flex items-center gap-3 px-2">
        <div className="flex h-8 w-8 items-center justify-center rounded bg-primary">
          <Icon
            name="precision_manufacturing"
            filled
            className="text-surface-bright"
            size={20}
          />
        </div>
        <div>
          <h1 className="font-headline text-headline-md font-bold leading-none text-surface-bright">
            VisionOps
          </h1>
          <p className="mt-1 text-label-sm uppercase tracking-widest text-primary-fixed-dim opacity-80">
            Industrial AI
          </p>
        </div>
      </div>

      <nav className="flex flex-1 flex-col gap-1">
        {NAV_ITEMS.map((item) => {
          const isActive = active === item.id;
          return (
            <Link
              key={item.id}
              href={item.href}
              className={cn(
                "flex items-center gap-3 px-3 py-3 transition-colors duration-150",
                isActive
                  ? "border-l-4 border-primary-fixed-dim bg-on-surface-variant/10 font-bold text-primary-fixed-dim"
                  : "font-medium text-on-surface-variant hover:bg-on-surface-variant/20",
              )}
            >
              <Icon name={item.icon} filled={isActive} size={22} />
              <span className="text-label-md">{item.label}</span>
            </Link>
          );
        })}
      </nav>

      <div className="mt-auto border-t border-outline/20 px-2 pt-6">
        <div className="flex items-center gap-3">
          <div className="flex h-8 w-8 items-center justify-center rounded-full bg-primary-container">
            <Icon name="person" className="text-on-primary" size={18} />
          </div>
          <div className="min-w-0">
            <p className="truncate text-label-md text-surface-bright">Ops Lead</p>
            <p className="truncate text-label-sm text-on-surface-variant">
              Shift A-4
            </p>
          </div>
        </div>
      </div>
    </aside>
  );
}
