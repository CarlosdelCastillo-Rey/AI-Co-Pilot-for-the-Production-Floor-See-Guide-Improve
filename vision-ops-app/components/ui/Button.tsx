import { cn } from "@/lib/cn";
import { Icon } from "./Icon";

type ButtonVariant = "primary" | "secondary" | "outline" | "ghost" | "inverse";

const variants: Record<ButtonVariant, string> = {
  primary: "bg-primary text-on-primary hover:bg-primary-container",
  secondary: "bg-surface-container text-on-surface hover:bg-surface-container-high",
  outline: "border border-primary text-primary hover:bg-primary-fixed/20",
  ghost: "text-on-surface-variant hover:bg-surface-container",
  inverse: "bg-inverse-surface text-surface-bright hover:opacity-90",
};

interface ButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: ButtonVariant;
  icon?: string;
  uppercase?: boolean;
}

export function Button({
  children,
  variant = "primary",
  icon,
  uppercase,
  className,
  ...props
}: ButtonProps) {
  return (
    <button
      className={cn(
        "inline-flex items-center justify-center gap-2 px-4 py-2 text-label-md transition-all duration-150 active:scale-[0.98] disabled:opacity-50",
        variants[variant],
        uppercase && "uppercase font-bold",
        className,
      )}
      {...props}
    >
      {icon && <Icon name={icon} size={18} />}
      {children}
    </button>
  );
}
