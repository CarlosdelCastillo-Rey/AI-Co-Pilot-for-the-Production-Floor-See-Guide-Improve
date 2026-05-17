import { cn } from "@/lib/cn";

interface IconProps {
  name: string;
  filled?: boolean;
  className?: string;
  size?: number;
}

export function Icon({ name, filled, className, size = 24 }: IconProps) {
  return (
    <span
      className={cn(
        "material-symbols-outlined leading-none",
        filled && "material-symbols-filled",
        className,
      )}
      style={{ fontSize: size }}
      aria-hidden
    >
      {name}
    </span>
  );
}
