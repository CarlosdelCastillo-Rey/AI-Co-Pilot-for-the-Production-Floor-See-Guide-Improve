"use client";

import { Icon } from "@/components/ui/Icon";
import { cn } from "@/lib/cn";

interface DataEmptyStateProps {
  icon?: string;
  title: string;
  description: string;
  className?: string;
}

export function DataEmptyState({
  icon = "inbox",
  title,
  description,
  className,
}: DataEmptyStateProps) {
  return (
    <div
      className={cn(
        "flex items-start gap-3 rounded-[14px] border border-outline-variant/60 bg-surface-container-lowest px-5 py-6",
        className,
      )}
    >
      <span className="flex h-10 w-10 shrink-0 items-center justify-center rounded-full bg-primary-fixed text-primary">
        <Icon name={icon} size={22} />
      </span>
      <div>
        <p className="text-body-sm font-semibold text-on-surface">{title}</p>
        <p className="mt-1 text-body-sm text-outline">{description}</p>
      </div>
    </div>
  );
}
