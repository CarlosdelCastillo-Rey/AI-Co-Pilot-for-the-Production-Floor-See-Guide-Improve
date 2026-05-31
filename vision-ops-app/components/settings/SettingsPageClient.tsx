"use client";

import { useCallback, useEffect, useState } from "react";
import { AppShell } from "@/components/layout/AppShell";
import { Button } from "@/components/ui/Button";
import { Icon } from "@/components/ui/Icon";
import {
  clearKpiDefinitionsCache,
  fetchKpiDefinitions,
  fetchPlantSettings,
  updatePlantSettings,
  type KpiDefinitionApi,
  type PlantSettingsApi,
} from "@/lib/api";

type FieldDef = {
  key: keyof PlantSettingsApi;
  label: string;
  type: "text" | "number";
  step?: string;
  hint?: string;
};

const SECTIONS: { title: string; icon: string; fields: FieldDef[] }[] = [
  {
    title: "Plant & Site",
    icon: "factory",
    fields: [{ key: "siteName", label: "Site name", type: "text" }],
  },
  {
    title: "Cost of Quality",
    icon: "payments",
    fields: [
      { key: "lineCostPerMinute", label: "Line cost ($/min)", type: "number", step: "0.01" },
      { key: "materialCostPerUnit", label: "Material cost ($/unit)", type: "number", step: "0.01" },
    ],
  },
  {
    title: "Production & Shifts",
    icon: "schedule",
    fields: [
      { key: "targetCycleSec", label: "Target cycle time (sec)", type: "number", step: "1" },
      { key: "shiftHours", label: "Shift length (hours)", type: "number", step: "0.5" },
      { key: "defaultClipDurationSec", label: "Default clip duration (sec)", type: "number", step: "1" },
      {
        key: "downtimeCriticalThresholdPct",
        label: "Downtime bar critical threshold (0–1)",
        type: "number",
        step: "0.05",
      },
    ],
  },
  {
    title: "Uptime Formula",
    icon: "monitoring",
    fields: [
      { key: "uptimeCriticalPenalty", label: "Critical event penalty (%)", type: "number", step: "0.1" },
      { key: "uptimeWarningPenalty", label: "Warning event penalty (%)", type: "number", step: "0.1" },
      { key: "uptimeFloorPct", label: "Uptime floor (%)", type: "number", step: "0.1" },
      { key: "uptimeCeilingPct", label: "Uptime ceiling (%)", type: "number", step: "0.1" },
    ],
  },
  {
    title: "OEE Clamps",
    icon: "speed",
    fields: [
      { key: "performanceFloorPct", label: "Performance floor (%)", type: "number", step: "1" },
      { key: "performanceCeilingPct", label: "Performance ceiling (%)", type: "number", step: "1" },
      { key: "qualityFloorPct", label: "Quality floor (%)", type: "number", step: "1" },
      { key: "qualityCeilingPct", label: "Quality ceiling (%)", type: "number", step: "1" },
    ],
  },
  {
    title: "Live Edge Estimates",
    icon: "memory",
    fields: [
      { key: "inferenceBasePerCamera", label: "Inferences base / camera", type: "number", step: "1" },
      { key: "inferenceProbeBonus", label: "Bonus per vision probe", type: "number", step: "1" },
      { key: "inferenceEventMultiplier", label: "Multiplier × events today", type: "number", step: "1" },
      { key: "inferenceMinPerCamera", label: "Minimum / online camera", type: "number", step: "1" },
    ],
  },
];

export function SettingsPageClient() {
  const [form, setForm] = useState<PlantSettingsApi | null>(null);
  const [kpiDefs, setKpiDefs] = useState<KpiDefinitionApi[]>([]);
  const [saving, setSaving] = useState(false);
  const [message, setMessage] = useState("");

  const load = useCallback(async () => {
    const [settings, defs] = await Promise.all([fetchPlantSettings(), fetchKpiDefinitions()]);
    if (settings) setForm(settings);
    setKpiDefs(defs);
  }, []);

  useEffect(() => {
    void load();
  }, [load]);

  const setField = (key: keyof PlantSettingsApi, value: string | number) => {
    setForm((prev) => (prev ? { ...prev, [key]: value } : prev));
  };

  const handleSave = async () => {
    if (!form) return;
    setSaving(true);
    setMessage("");
    const updated = await updatePlantSettings(form);
    clearKpiDefinitionsCache();
    setSaving(false);
    if (updated) {
      setForm(updated);
      setMessage("Settings saved. KPIs will use new values on next refresh.");
    } else {
      setMessage("Failed to save settings.");
    }
  };

  if (!form) {
    return (
      <AppShell>
        <p className="p-8 text-body-sm text-outline">Loading settings…</p>
      </AppShell>
    );
  }

  return (
    <AppShell searchPlaceholder="Search settings…">
      <div className="mx-auto max-w-[960px] px-8 py-10">
        <header className="mb-8 flex flex-wrap items-end justify-between gap-4">
          <div>
            <h1 className="font-headline text-headline-lg text-on-background">Plant Settings</h1>
            <p className="mt-1 text-body-md text-outline">
              Configure formulas and cost variables used across Live, Timeline, and Analytics.
            </p>
            {form.updatedBy && (
              <p className="mt-2 text-body-sm text-outline">
                Last saved by <span className="font-medium text-on-surface">{form.updatedBy}</span>
              </p>
            )}
          </div>
          <Button icon="save" className="min-h-touch" disabled={saving} onClick={() => void handleSave()}>
            {saving ? "Saving…" : "Save changes"}
          </Button>
        </header>

        {message && (
          <p className="mb-6 rounded-lg border border-outline-variant bg-surface-container-low px-4 py-3 text-body-sm">
            {message}
          </p>
        )}

        <div className="space-y-8">
          {SECTIONS.map((section) => (
            <section
              key={section.title}
              className="rounded-card border border-outline-variant bg-surface-container-lowest p-6"
            >
              <h2 className="mb-4 flex items-center gap-2 font-headline text-body-lg font-semibold">
                <Icon name={section.icon} className="text-primary" size={20} />
                {section.title}
              </h2>
              <div className="grid gap-4 sm:grid-cols-2">
                {section.fields.map((field) => (
                  <label key={field.key} className="block">
                    <span className="mb-1 block font-label text-label-sm text-outline">{field.label}</span>
                    <input
                      type={field.type}
                      step={field.step}
                      value={String(form[field.key] ?? "")}
                      onChange={(e) =>
                        setField(
                          field.key,
                          field.type === "number" ? Number(e.target.value) : e.target.value,
                        )
                      }
                      className="min-h-touch w-full rounded-lg border border-outline-variant bg-surface px-3 text-body-md"
                    />
                  </label>
                ))}
              </div>
            </section>
          ))}

          <section className="rounded-card border border-outline-variant bg-surface-container-lowest p-6">
            <h2 className="mb-4 flex items-center gap-2 font-headline text-body-lg font-semibold">
              <Icon name="help" className="text-primary" size={20} />
              KPI Calculation Reference
            </h2>
            <p className="mb-4 text-body-sm text-outline">
              These formulas power the ⓘ icons on dashboards. Values update when you save settings above.
            </p>
            <div className="divide-y divide-outline-variant/60">
              {kpiDefs.map((d) => (
                <details key={d.id} className="group py-3">
                  <summary className="cursor-pointer list-none font-label text-label-md font-semibold text-on-surface">
                    {d.label}
                  </summary>
                  <p className="mt-2 text-body-sm text-on-surface-variant">{d.description}</p>
                  <p className="mt-1 rounded bg-surface-container-low px-2 py-1 font-label text-[11px] text-outline">
                    {d.formula}
                  </p>
                </details>
              ))}
            </div>
          </section>
        </div>
      </div>
    </AppShell>
  );
}
