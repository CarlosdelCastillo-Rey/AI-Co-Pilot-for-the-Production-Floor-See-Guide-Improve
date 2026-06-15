"use client";

import Link from "next/link";
import { AppShell } from "@/components/layout/AppShell";
import { Button } from "@/components/ui/Button";
import { Card } from "@/components/ui/Card";
import { SectionHeader } from "@/components/ui/SectionHeader";
import { cn } from "@/lib/cn";

/** Shared form controls — match Model Analysis / Timeline pages */
export const fieldLabel =
  "mb-1 block font-label text-[10px] uppercase tracking-wide text-outline";
export const inputClass =
  "w-full rounded-md border border-outline-variant/60 bg-surface-container-lowest px-3 py-2 text-body-sm text-on-surface";
export const selectClass = inputClass;
export const textareaClass = cn(inputClass, "min-h-[4rem] resize-y");

export function HarV2PageShell({
  title,
  description,
  action,
  children,
}: {
  title: string;
  description: string;
  action?: React.ReactNode;
  children: React.ReactNode;
}) {
  return (
    <AppShell fullBleed>
      <div className="h-full overflow-y-auto p-6 lg:p-8">
        <div className="mx-auto max-w-[1400px] space-y-6">
          <header className="flex flex-col gap-4 lg:flex-row lg:items-start lg:justify-between">
            <div>
              <h2 className="font-headline text-headline-lg text-on-surface">{title}</h2>
              <p className="mt-1 max-w-2xl text-body-md text-outline">{description}</p>
            </div>
            {action ? <div className="flex flex-wrap items-center gap-2">{action}</div> : null}
          </header>
          {children}
        </div>
      </div>
    </AppShell>
  );
}

export function HarV2Panel({
  title,
  children,
  className,
  bodyClassName,
}: {
  title: string;
  children: React.ReactNode;
  className?: string;
  bodyClassName?: string;
}) {
  return (
    <Card className={cn("overflow-hidden rounded-lg", className)}>
      <SectionHeader title={title} />
      <div className={cn("p-4", bodyClassName)}>{children}</div>
    </Card>
  );
}

export function HarV2ListButton({
  active,
  onClick,
  children,
}: {
  active?: boolean;
  onClick: () => void;
  children: React.ReactNode;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      className={cn(
        "w-full rounded-md px-3 py-2 text-left text-body-sm transition-colors",
        active
          ? "bg-primary/10 font-medium text-primary ring-1 ring-primary/25"
          : "text-on-surface-variant hover:bg-surface-container-high",
      )}
    >
      {children}
    </button>
  );
}

export function HarV2Empty({ children }: { children: React.ReactNode }) {
  return (
    <p className="rounded-md border border-dashed border-outline-variant/60 bg-surface-container-low px-4 py-8 text-center text-body-sm text-outline">
      {children}
    </p>
  );
}

export function HarV2Message({ tone, children }: { tone?: "info" | "error"; children: React.ReactNode }) {
  return (
    <p
      className={cn(
        "rounded-md px-4 py-2 text-body-sm",
        tone === "error"
          ? "border border-error/30 bg-error-container text-on-error-container"
          : "border border-primary/25 bg-primary/5 text-on-surface",
      )}
    >
      {children}
    </p>
  );
}

export function HarV2Link(props: React.ComponentProps<typeof Link>) {
  return (
    <Link
      {...props}
      className={cn("text-body-sm text-primary hover:underline", props.className)}
    />
  );
}

export function HarV2Table({ children }: { children: React.ReactNode }) {
  return (
    <div className="overflow-x-auto">
      <table className="w-full min-w-[280px] text-left text-body-sm">{children}</table>
    </div>
  );
}

export function HarV2TableHead({ children }: { children: React.ReactNode }) {
  return (
    <thead>
      <tr className="border-b border-outline-variant/60 font-label text-[10px] uppercase tracking-wide text-outline">
        {children}
      </tr>
    </thead>
  );
}

export function HarV2Th({ children, className }: { children: React.ReactNode; className?: string }) {
  return <th className={cn("py-2 pr-3 text-left font-normal", className)}>{children}</th>;
}

export function HarV2TableRow({ children }: { children: React.ReactNode }) {
  return <tr className="border-b border-outline-variant/30 text-on-surface-variant">{children}</tr>;
}

export function HarV2TableCell({
  children,
  className,
}: {
  children: React.ReactNode;
  className?: string;
}) {
  return <td className={cn("py-2 pr-3", className)}>{children}</td>;
}

export function HarV2Badge({ children }: { children: React.ReactNode }) {
  return (
    <span className="inline-block rounded-md bg-success-container px-2 py-0.5 font-mono text-[11px] text-success">
      {children}
    </span>
  );
}

export type HarPersonHitlTab = "sessions" | "registry" | "tuning";

export function HarV2TabBar({
  tabs,
  active,
  onChange,
}: {
  tabs: { id: HarPersonHitlTab; label: string; hint?: string }[];
  active: HarPersonHitlTab;
  onChange: (id: HarPersonHitlTab) => void;
}) {
  return (
    <nav
      className="flex flex-wrap gap-1 rounded-lg border border-outline-variant/50 bg-surface-container-low p-1"
      aria-label="Person HITL sections"
    >
      {tabs.map((tab) => (
        <button
          key={tab.id}
          type="button"
          onClick={() => onChange(tab.id)}
          title={tab.hint}
          className={cn(
            "rounded-md px-3 py-2 text-body-sm transition-colors",
            active === tab.id
              ? "bg-primary/12 font-medium text-primary shadow-sm ring-1 ring-primary/20"
              : "text-on-surface-variant hover:bg-surface-container-high",
          )}
        >
          {tab.label}
        </button>
      ))}
    </nav>
  );
}

export { Button };
