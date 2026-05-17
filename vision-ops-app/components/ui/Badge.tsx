import { cn } from "@/lib/cn";

type BadgeVariant = "critical" | "warning" | "info" | "neutral" | "zone" | "disabled";

const variants: Record<BadgeVariant, string> = {
  critical: "bg-error-container text-on-error-container border border-error/20",
  warning: "bg-tertiary-fixed text-on-tertiary-fixed border border-tertiary/20",
  info: "bg-primary-fixed text-on-primary-fixed border border-primary/20",
  neutral: "bg-surface-container text-on-surface-variant",
  zone: "bg-surface-container text-on-surface-variant",
  disabled: "bg-secondary-container text-on-secondary-container",
};

interface BadgeProps {
  children: React.ReactNode;
  variant?: BadgeVariant;
  className?: string;
}

export function Badge({ children, variant = "neutral", className }: BadgeProps) {
  return (
    <span
      className={cn(
        "inline-flex px-2 py-0.5 text-label-sm uppercase tracking-wider font-bold",
        variants[variant],
        className,
      )}
    >
      {children}
    </span>
  );
}
