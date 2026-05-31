"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { createPortal } from "react-dom";
import { Icon } from "@/components/ui/Icon";
import { fetchKpiDefinitions, type KpiDefinitionApi } from "@/lib/api";
import { cn } from "@/lib/cn";

let definitionsPromise: Promise<KpiDefinitionApi[]> | null = null;

function loadDefinitions() {
  if (!definitionsPromise) {
    definitionsPromise = fetchKpiDefinitions();
  }
  return definitionsPromise;
}

interface KpiHelpProps {
  kpiId: string;
  className?: string;
}

type PopoverPos = { top: number; left: number; placement: "below" | "above" };

function computePosition(anchor: DOMRect): PopoverPos {
  const pad = 8;
  const popoverW = 288;
  const popoverH = 160;
  const vw = window.innerWidth;
  const vh = window.innerHeight;

  let left = anchor.right - popoverW;
  left = Math.max(pad, Math.min(left, vw - popoverW - pad));

  const spaceBelow = vh - anchor.bottom;
  const placement: "below" | "above" = spaceBelow >= popoverH + pad || spaceBelow >= anchor.top ? "below" : "above";
  const top =
    placement === "below"
      ? Math.min(anchor.bottom + pad, vh - popoverH - pad)
      : Math.max(pad, anchor.top - popoverH - pad);

  return { top, left, placement };
}

export function KpiHelp({ kpiId, className }: KpiHelpProps) {
  const [open, setOpen] = useState(false);
  const [def, setDef] = useState<KpiDefinitionApi | null>(null);
  const [pos, setPos] = useState<PopoverPos | null>(null);
  const [mounted, setMounted] = useState(false);
  const btnRef = useRef<HTMLButtonElement>(null);

  useEffect(() => {
    setMounted(true);
    void loadDefinitions().then((items) => setDef(items.find((d) => d.id === kpiId) ?? null));
  }, [kpiId]);

  const updatePosition = useCallback(() => {
    if (!btnRef.current) return;
    setPos(computePosition(btnRef.current.getBoundingClientRect()));
  }, []);

  useEffect(() => {
    if (!open) return;
    updatePosition();
    const onScroll = () => updatePosition();
    const onResize = () => updatePosition();
    window.addEventListener("scroll", onScroll, true);
    window.addEventListener("resize", onResize);
    return () => {
      window.removeEventListener("scroll", onScroll, true);
      window.removeEventListener("resize", onResize);
    };
  }, [open, updatePosition]);

  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") setOpen(false);
    };
    document.addEventListener("keydown", onKey);
    return () => document.removeEventListener("keydown", onKey);
  }, [open]);

  const toggle = (e: React.MouseEvent) => {
    e.stopPropagation();
    e.preventDefault();
    if (open) {
      setOpen(false);
      return;
    }
    updatePosition();
    setOpen(true);
  };

  if (!def) return null;

  const popover =
    open && mounted && pos
      ? createPortal(
          <>
            <button
              type="button"
              className="fixed inset-0 z-[500]"
              aria-label="Close KPI help"
              onClick={() => setOpen(false)}
            />
            <div
              role="tooltip"
              className="fixed z-[501] w-72 rounded-lg border border-outline-variant bg-surface-container-lowest p-3 text-left shadow-overlay"
              style={{ top: pos.top, left: pos.left }}
            >
              <p className="mb-1 font-label text-label-sm font-bold text-on-surface">{def.label}</p>
              <p className="mb-2 text-body-sm text-on-surface-variant">{def.description}</p>
              <p className="rounded bg-surface-container-low px-2 py-1 font-label text-[11px] text-outline">
                {def.formula}
              </p>
              {def.settingsKeys.length > 0 && (
                <p className="mt-2 font-label text-[10px] text-outline">
                  Editable in Settings: {def.settingsKeys.join(", ")}
                </p>
              )}
            </div>
          </>,
          document.body,
        )
      : null;

  return (
    <>
      <span className={cn("relative inline-flex shrink-0", className)}>
        <button
          ref={btnRef}
          type="button"
          aria-label={`How ${def.label} is calculated`}
          aria-expanded={open}
          onClick={toggle}
          className={cn(
            "rounded-full p-0.5 text-outline transition-colors hover:bg-surface-container-low hover:text-primary",
            open && "bg-primary-fixed/30 text-primary",
          )}
        >
          <Icon name="info" size={16} />
        </button>
      </span>
      {popover}
    </>
  );
}

export function KpiLabel({
  label,
  kpiId,
  className,
}: {
  label: string;
  kpiId: string;
  className?: string;
}) {
  return (
    <span className={cn("inline-flex items-center gap-1.5", className)}>
      <span>{label}</span>
      <KpiHelp kpiId={kpiId} />
    </span>
  );
}
