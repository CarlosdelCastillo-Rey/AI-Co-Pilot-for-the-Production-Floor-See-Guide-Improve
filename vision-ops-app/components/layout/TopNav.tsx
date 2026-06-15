"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useAuth } from "@/components/auth/AuthProvider";
import { NotificationsBell } from "@/components/layout/NotificationsBell";
import { Icon } from "@/components/ui/Icon";
import { cn } from "@/lib/cn";

interface TopNavProps {
  searchPlaceholder?: string;
  searchValue?: string;
  onSearchChange?: (value: string) => void;
}

export function TopNav({
  searchPlaceholder = "Search facilities...",
  searchValue,
  onSearchChange,
}: TopNavProps) {
  const pathname = usePathname();
  const { user, logout } = useAuth();

  return (
    <header className="fixed left-[240px] right-0 top-0 z-50 flex h-16 items-center justify-between overflow-visible border-b border-outline-variant bg-surface-container-lowest px-8">
      <div className="relative w-72 max-w-full">
        <Icon
          name="search"
          className="absolute left-3 top-1/2 -translate-y-1/2 text-outline"
          size={18}
        />
        <input
          type="text"
          value={searchValue}
          onChange={(e) => onSearchChange?.(e.target.value)}
          placeholder={searchPlaceholder}
          className="w-full rounded-lg border border-outline-variant/60 bg-surface-container-low py-2 pl-10 pr-4 text-body-sm text-on-surface placeholder:text-outline/70 focus:border-primary focus:outline-none focus:ring-1 focus:ring-primary/30"
        />
      </div>
      <div className="flex items-center gap-3 text-outline">
        <Link
          href="/settings"
          className={cn(
            "inline-flex min-h-touch min-w-touch items-center justify-center rounded-lg transition-colors hover:bg-surface-container-low hover:text-on-surface",
            pathname === "/settings" && "bg-surface-container-low text-primary",
          )}
          title="Plant settings"
        >
          <Icon name="settings" size={20} />
        </Link>
        <NotificationsBell />
        {user ? (
          <div className="flex items-center gap-2 border-l border-outline-variant/60 pl-3">
            <span className="hidden max-w-[160px] truncate text-body-sm sm:inline">
              <span className="font-medium text-on-surface">{user.name}</span>
              <span className="text-outline"> · {user.role}</span>
            </span>
            <button
              type="button"
              onClick={logout}
              className="inline-flex min-h-touch min-w-touch items-center justify-center rounded-lg transition-colors hover:bg-surface-container-low hover:text-on-surface"
              title="Sign out"
            >
              <Icon name="logout" size={20} />
            </button>
          </div>
        ) : null}
      </div>
    </header>
  );
}
