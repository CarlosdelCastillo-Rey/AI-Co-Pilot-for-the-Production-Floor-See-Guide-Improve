"use client";

import { useEffect, useState } from "react";
import { Icon } from "@/components/ui/Icon";
import { cn } from "@/lib/cn";

type Props = {
  id: string;
  title: string;
  subtitle?: string;
  defaultOpen?: boolean;
  /** Persist open state in localStorage (key: visionops:live-panel:{id}) */
  persist?: boolean;
  children: React.ReactNode;
  className?: string;
  bodyClassName?: string;
  headerExtra?: React.ReactNode;
};

function storageKey(id: string) {
  return `visionops:live-panel:${id}`;
}

export function LiveCollapsiblePanel({
  id,
  title,
  subtitle,
  defaultOpen = true,
  persist = true,
  children,
  className,
  bodyClassName,
  headerExtra,
}: Props) {
  const [open, setOpen] = useState(defaultOpen);

  useEffect(() => {
    if (!persist || typeof window === "undefined") return;
    const raw = window.localStorage.getItem(storageKey(id));
    if (raw === "0") setOpen(false);
    else if (raw === "1") setOpen(true);
  }, [id, persist]);

  const toggle = () => {
    setOpen((prev) => {
      const next = !prev;
      if (persist && typeof window !== "undefined") {
        window.localStorage.setItem(storageKey(id), next ? "1" : "0");
      }
      return next;
    });
  };

  return (
    <section
      className={cn(
        "flex min-h-0 flex-col overflow-hidden rounded-lg border border-outline-variant/60 bg-surface-container-lowest shadow-sm",
        className,
      )}
    >
      <div className="flex shrink-0 items-start gap-2 border-b border-outline-variant/50 px-3 py-2.5">
        <button
          type="button"
          onClick={toggle}
          className="flex min-w-0 flex-1 items-start gap-2 rounded-md text-left hover:bg-surface-container-high/80"
          aria-expanded={open}
        >
          <Icon
            name={open ? "expand_less" : "expand_more"}
            size={20}
            className="mt-0.5 shrink-0 text-outline"
          />
          <span className="min-w-0 flex-1">
            <span className="block font-label text-[10px] font-bold uppercase tracking-wider text-outline">
              {title}
            </span>
            {subtitle ? (
              <span className="mt-0.5 block truncate text-body-sm text-on-surface">{subtitle}</span>
            ) : null}
          </span>
        </button>
        {headerExtra ? <div className="shrink-0 pt-0.5">{headerExtra}</div> : null}
      </div>
      {open ? (
        <div
          className={cn(
            "min-h-0 flex-1 overflow-y-auto overscroll-y-contain p-4",
            bodyClassName,
          )}
        >
          {children}
        </div>
      ) : null}
    </section>
  );
}

type ColumnToggleProps = {
  label: string;
  active: boolean;
  onClick: () => void;
  count?: number;
};

export function LiveColumnToggle({ label, active, onClick, count }: ColumnToggleProps) {
  return (
    <button
      type="button"
      onClick={onClick}
      className={cn(
        "inline-flex items-center gap-1.5 rounded-md border px-2.5 py-1.5 font-label text-[11px] transition-colors",
        active
          ? "border-primary bg-primary/10 text-primary"
          : "border-outline-variant/60 text-outline hover:border-primary/40 hover:text-on-surface",
      )}
      aria-pressed={active}
    >
      <Icon name={active ? "visibility" : "visibility_off"} size={16} />
      {label}
      {count != null && count > 0 ? (
        <span className="rounded-full bg-primary/15 px-1.5 font-mono text-[10px]">{count}</span>
      ) : null}
    </button>
  );
}
