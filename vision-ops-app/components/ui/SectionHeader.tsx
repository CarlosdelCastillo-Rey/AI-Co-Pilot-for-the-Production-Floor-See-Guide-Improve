import { cn } from "@/lib/cn";

interface SectionHeaderProps {
  title: string;
  action?: React.ReactNode;
  className?: string;
}

export function SectionHeader({ title, action, className }: SectionHeaderProps) {
  return (
    <div
      className={cn(
        "flex items-center justify-between border-b border-outline-variant bg-surface-container-highest px-4 py-2",
        className,
      )}
    >
      <span className="text-label-sm uppercase tracking-wider text-on-surface-variant">
        {title}
      </span>
      {action}
    </div>
  );
}
