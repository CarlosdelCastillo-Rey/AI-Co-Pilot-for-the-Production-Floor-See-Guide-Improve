"use client";

import { useState } from "react";
import { AppShell } from "@/components/layout/AppShell";
import { Badge } from "@/components/ui/Badge";
import { Button } from "@/components/ui/Button";
import { Icon } from "@/components/ui/Icon";
import { SectionHeader } from "@/components/ui/SectionHeader";
import { Switch } from "@/components/ui/Switch";
import { ALERT_RULES, type AlertRule } from "@/lib/mock-data";

export default function AlertsPage() {
  const [rules, setRules] = useState(ALERT_RULES);

  const toggleRule = (id: string) => {
    setRules((prev) =>
      prev.map((r) => (r.id === id ? { ...r, enabled: !r.enabled } : r)),
    );
  };

  const activeCount = rules.filter((r) => r.enabled).length;

  return (
    <AppShell searchPlaceholder="Search rules...">
      <div className="mx-auto max-w-[1600px] space-y-lg p-gutter">
        <header className="flex items-end justify-between pb-4">
          <div>
            <h2 className="font-headline text-headline-lg text-on-background">
              Alert Rules & Integrations
            </h2>
            <p className="text-body-md text-on-surface-variant">
              Configure autonomous vision triggers and downstream connectivity.
            </p>
          </div>
          <Button icon="add" uppercase>
            Create New Rule
          </Button>
        </header>

        <div className="grid grid-cols-12 gap-gutter">
          <section className="col-span-12 space-y-md lg:col-span-8">
            <SectionHeader
              title="Video Analytic Rules"
              action={
                <span className="text-label-sm font-bold text-primary">
                  {activeCount} Active
                </span>
              }
            />
            <div className="grid grid-cols-1 gap-md md:grid-cols-2">
              {rules.map((rule) => (
                <RuleCard
                  key={rule.id}
                  rule={rule}
                  onToggle={() => toggleRule(rule.id)}
                />
              ))}
            </div>
          </section>

          <section className="col-span-12 space-y-md lg:col-span-4">
            <SectionHeader title="External Notifications" />
            <TelegramPanel />
          </section>

          <section className="col-span-12 space-y-md">
            <SectionHeader title="System Telemetry" />
            <TelemetryPanel />
          </section>
        </div>
      </div>
    </AppShell>
  );
}

function RuleCard({
  rule,
  onToggle,
}: {
  rule: AlertRule;
  onToggle: () => void;
}) {
  const severityVariant =
    rule.severity === "CRITICAL"
      ? "critical"
      : rule.severity === "WARNING"
        ? "warning"
        : "disabled";

  return (
    <article className="flex flex-col justify-between border border-outline-variant bg-white p-md transition-colors hover:border-primary">
      <div className="mb-4 flex items-start justify-between">
        <div className="bg-surface-container-low p-2">
          <Icon name={rule.icon} className="text-primary" />
        </div>
        <Switch checked={rule.enabled} onChange={onToggle} label={rule.title} />
      </div>
      <div>
        <h4 className="text-body-lg font-semibold">{rule.title}</h4>
        <p className="mt-1 text-body-sm text-on-surface-variant">
          {rule.description}
        </p>
        <div className="mt-4 flex gap-2">
          <Badge variant="zone">{rule.zone}</Badge>
          <Badge variant={severityVariant}>{rule.severity}</Badge>
        </div>
      </div>
    </article>
  );
}

function TelegramPanel() {
  return (
    <article className="space-y-6 border border-outline-variant bg-white p-lg">
      <div className="mb-4 flex items-center gap-4">
        <div className="flex h-12 w-12 items-center justify-center rounded-lg bg-[#0088cc]/10">
          <svg width="24" height="24" viewBox="0 0 24 24" fill="none" aria-hidden>
            <path
              d="M21.1 2.85C20.88 2.63 20.58 2.5 20.25 2.5H3.75C3.06 2.5 2.5 3.06 2.5 3.75V20.25C2.5 20.94 3.06 21.5 3.75 21.5H20.25C20.94 21.5 21.5 20.94 21.5 20.25V3.75C21.5 3.42 21.37 3.12 21.1 2.85Z"
              fill="#0088cc"
            />
          </svg>
        </div>
        <div>
          <h4 className="text-body-lg font-semibold">Telegram Messenger</h4>
          <p className="text-body-sm text-on-surface-variant">
            Real-time alert relay
          </p>
        </div>
      </div>
      <div className="space-y-4">
        <FloatingInput label="Bot_Token" type="password" value="6392019485:AAF-7Xz9W..." active />
        <FloatingInput label="Chat_ID" value="-100194857203" />
        <Button variant="outline" uppercase className="w-full py-3">
          Send Test Alert
        </Button>
      </div>
      <div className="border-t border-outline-variant pt-4">
        <div className="flex items-center justify-between">
          <span className="text-body-sm font-medium">Status</span>
          <span className="flex items-center gap-1.5 text-primary">
            <span className="h-2 w-2 rounded-full bg-primary" />
            <span className="text-label-sm font-bold">CONNECTED</span>
          </span>
        </div>
      </div>
    </article>
  );
}

function FloatingInput({
  label,
  value,
  type = "text",
  active,
}: {
  label: string;
  value: string;
  type?: string;
  active?: boolean;
}) {
  return (
    <div className="relative">
      <label
        className={`absolute -top-2 left-3 bg-white px-1 text-label-sm ${active ? "text-primary" : "text-on-surface-variant"}`}
      >
        {label}
      </label>
      <input
        type={type}
        defaultValue={value}
        readOnly
        className="w-full rounded-lg border border-outline p-3 text-label-md focus:border-primary focus:ring-primary"
      />
    </div>
  );
}

function TelemetryPanel() {
  const metrics = [
    { label: "Backend Stability (FastAPI)", value: "99.98%", bars: [80, 85, 82, 90, 95, 92, 98, 88, 94, 100], highlight: true },
    { label: "Redis Cache Latency", value: "2.4ms", bars: [30, 35, 28, 40, 45, 32, 38, 25, 34, 30], highlight: false },
    { label: "Worker Load (Gunicorn)", value: "14.2%", bars: [10, 15, 12, 20, 15, 12, 18, 15, 14, 20], highlight: false },
  ];

  return (
    <article className="grid grid-cols-1 gap-lg border border-outline-variant bg-white p-lg md:grid-cols-3">
      {metrics.map((m) => (
        <div key={m.label} className="space-y-4">
          <div className="flex items-center justify-between">
            <span className="text-label-md text-on-surface-variant">{m.label}</span>
            <span className="text-label-sm text-primary">{m.value}</span>
          </div>
          <div className="flex h-16 items-end gap-1">
            {m.bars.map((h, i) => (
              <div
                key={i}
                className={`w-full ${m.highlight || i === 4 ? "bg-primary-fixed" : "bg-on-secondary-container/20"}`}
                style={{ height: `${h}%` }}
              />
            ))}
          </div>
        </div>
      ))}
    </article>
  );
}
