import { cn } from "@/lib/cn";

interface CardProps {
  children: React.ReactNode;
  className?: string;
  hover?: boolean;
}

export function Card({ children, className, hover }: CardProps) {
  return (
    <div
      className={cn(
        "bg-surface-container-lowest border border-outline-variant",
        hover && "hover:border-primary transition-colors",
        className,
      )}
    >
      {children}
    </div>
  );
}
