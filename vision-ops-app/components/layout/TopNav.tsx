"use client";

import { Icon } from "@/components/ui/Icon";

interface TopNavProps {
  searchPlaceholder?: string;
}

export function TopNav({
  searchPlaceholder = "Search facilities...",
}: TopNavProps) {
  return (
    <header className="fixed left-[240px] right-0 top-0 z-40 flex h-16 items-center justify-between border-b border-outline-variant bg-inverse-surface px-8">
      <div className="flex max-w-2xl flex-1 items-center gap-8">
        <div className="relative w-80">
          <Icon
            name="search"
            className="absolute left-3 top-1/2 -translate-y-1/2 text-on-surface-variant"
            size={20}
          />
          <input
            type="text"
            placeholder={searchPlaceholder}
            className="w-full rounded-lg border-none bg-surface-container-lowest/10 py-2 pl-10 pr-4 text-body-sm text-surface-bright placeholder:text-on-surface-variant/60 focus:ring-1 focus:ring-primary-fixed-dim"
          />
        </div>
        <nav className="hidden md:flex">
          <span className="flex h-16 items-center border-b-2 border-primary-fixed-dim text-body-md font-bold text-primary-fixed-dim">
            System Status
          </span>
        </nav>
      </div>
      <div className="flex items-center gap-4 text-on-surface-variant">
        <button
          type="button"
          className="relative transition-colors hover:text-surface-bright"
          aria-label="Notifications"
        >
          <Icon name="notifications" size={22} />
          <span className="absolute right-0 top-0 h-2 w-2 rounded-full border border-inverse-surface bg-error" />
        </button>
        <button
          type="button"
          className="transition-colors hover:text-surface-bright"
          aria-label="Settings"
        >
          <Icon name="settings" size={22} />
        </button>
      </div>
    </header>
  );
}
