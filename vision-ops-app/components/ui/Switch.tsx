"use client";

import { cn } from "@/lib/cn";

interface SwitchProps {
  checked: boolean;
  onChange?: (checked: boolean) => void;
  label?: string;
}

export function Switch({ checked, onChange, label }: SwitchProps) {
  return (
    <label className="relative inline-flex cursor-pointer items-center">
      {label && <span className="sr-only">{label}</span>}
      <input
        type="checkbox"
        className="peer sr-only"
        checked={checked}
        onChange={(e) => onChange?.(e.target.checked)}
      />
      <div
        className={cn(
          "relative h-6 w-11 rounded-full bg-secondary transition-colors duration-150",
          "peer-checked:bg-primary",
          "after:absolute after:left-[2px] after:top-[2px] after:h-5 after:w-5",
          "after:rounded-full after:bg-white after:transition-all after:content-['']",
          "peer-checked:after:translate-x-5",
        )}
      />
    </label>
  );
}
