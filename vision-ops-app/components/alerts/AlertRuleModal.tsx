"use client";

import { useEffect, useState } from "react";
import { Button } from "@/components/ui/Button";
import { Icon } from "@/components/ui/Icon";
import type { AlertActionApi, AlertCaseType, AlertRuleApi, EmailTemplateApi } from "@/lib/api";
import { cn } from "@/lib/cn";

const DEFAULT_HAR_CONFIDENCE = 0.3;

export type AlertRuleFormValues = {
  title: string;
  description: string;
  zone: string;
  caseType: AlertCaseType;
  severity: "CRITICAL" | "WARNING" | "INFO";
  enabled: boolean;
  notifyEmail: boolean;
  notifyTelegram: boolean;
  notifyInApp: boolean;
  icon: string;
  emailTemplateId?: string;
  confidenceThreshold?: number;
  harActionLabel?: string;
};

interface AlertRuleModalProps {
  open: boolean;
  mode: "create" | "edit";
  actions: AlertActionApi[];
  emailTemplates: EmailTemplateApi[];
  initial?: AlertRuleApi | null;
  onClose: () => void;
  onSubmit: (values: AlertRuleFormValues) => Promise<boolean>;
};

function severityFromRule(rule: AlertRuleApi): "CRITICAL" | "WARNING" | "INFO" {
  if (rule.severity === "CRITICAL") return "CRITICAL";
  if (rule.severity === "WARNING") return "WARNING";
  return "INFO";
}

function defaultsFromAction(action: AlertActionApi, templates: EmailTemplateApi[]): AlertRuleFormValues {
  const tpl = templates.find((t) => t.id === "har.action_detected");
  return {
    title: `HAR — ${action.label}`,
    description: action.description,
    zone: action.defaultZone,
    caseType: "har_action_detected",
    severity: (action.defaultSeverity === "CRITICAL" || action.defaultSeverity === "WARNING"
      ? action.defaultSeverity
      : "INFO") as "CRITICAL" | "WARNING" | "INFO",
    enabled: action.defaultEnabled,
    notifyEmail: true,
    notifyTelegram: false,
    notifyInApp: false,
    icon: action.icon,
    emailTemplateId: tpl?.id,
    confidenceThreshold: DEFAULT_HAR_CONFIDENCE,
    harActionLabel: action.harActionLabel ?? action.label,
  };
}

export function AlertRuleModal({
  open,
  mode,
  actions,
  emailTemplates,
  initial,
  onClose,
  onSubmit,
}: AlertRuleModalProps) {
  const [values, setValues] = useState<AlertRuleFormValues>(() =>
    actions[0]
      ? defaultsFromAction(actions[0], emailTemplates)
      : {
          title: "",
          description: "",
          zone: "HAR LIVE",
          caseType: "har_action_detected",
          severity: "INFO",
          enabled: true,
          notifyEmail: true,
          notifyTelegram: false,
          notifyInApp: false,
          icon: "smart_toy",
        },
  );
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!open) return;
    if (mode === "edit" && initial) {
      setValues({
        title: initial.title,
        description: initial.description,
        zone: initial.zone,
        caseType: "har_action_detected",
        severity: severityFromRule(initial),
        enabled: initial.enabled,
        notifyEmail: initial.notifyEmail ?? true,
        notifyTelegram: initial.notifyTelegram ?? false,
        notifyInApp: initial.notifyInApp ?? false,
        icon: initial.icon,
        emailTemplateId: initial.emailTemplateId ?? undefined,
        confidenceThreshold: initial.confidenceThreshold ?? DEFAULT_HAR_CONFIDENCE,
        harActionLabel: initial.harActionLabel,
      });
    } else if (actions[0]) {
      setValues(defaultsFromAction(actions[0], emailTemplates));
    }
    setError(null);
  }, [open, mode, initial, actions, emailTemplates]);

  const templatesForCase = emailTemplates.filter((t) => t.enabled && t.id === "har.action_detected");
  const harConfidencePct = Math.round((values.confidenceThreshold ?? DEFAULT_HAR_CONFIDENCE) * 100);

  if (!open) return null;

  const handleActionChange = (action: AlertActionApi) => {
    const tpl = emailTemplates.find((t) => t.id === "har.action_detected");
    setValues((v) => ({
      ...v,
      caseType: "har_action_detected",
      icon: action.icon,
      harActionLabel: action.harActionLabel ?? action.label,
      emailTemplateId: tpl?.id,
      notifyTelegram: false,
      notifyInApp: false,
      confidenceThreshold: DEFAULT_HAR_CONFIDENCE,
      ...(mode === "create"
        ? {
            title: `HAR — ${action.label}`,
            description: action.description,
            zone: action.defaultZone,
            severity: (action.defaultSeverity === "CRITICAL" || action.defaultSeverity === "WARNING"
              ? action.defaultSeverity
              : "INFO") as "CRITICAL" | "WARNING" | "INFO",
          }
        : {}),
    }));
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!values.title.trim() || !values.zone.trim()) {
      setError("Title and zone are required.");
      return;
    }
    if (values.enabled && !values.notifyEmail && !values.notifyTelegram && !values.notifyInApp) {
      setError("Enable at least one channel: In-app, Email, or Telegram.");
      return;
    }
    if (mode === "create" && !values.harActionLabel) {
      setError("Select a HAR action.");
      return;
    }
    setSaving(true);
    setError(null);
    const ok = await onSubmit(values);
    setSaving(false);
    if (ok) onClose();
    else setError("Could not save rule. Check the backend is running on :8000.");
  };

  return (
    <div className="fixed inset-0 z-[100] flex items-center justify-center p-4">
      <button
        type="button"
        className="absolute inset-0 bg-on-surface/40 backdrop-blur-sm"
        onClick={onClose}
        aria-label="Close"
      />
      <div className="relative z-10 flex max-h-[90dvh] w-full max-w-lg flex-col rounded-card border border-outline-variant bg-surface-container-lowest shadow-overlay">
        <div className="flex shrink-0 items-center justify-between border-b border-outline-variant px-6 py-4">
          <div>
            <h2 className="font-headline text-headline-md text-on-surface">
              {mode === "create" ? "Create HAR Alert" : "Edit HAR Alert"}
            </h2>
            <p className="text-body-sm text-outline">Per-action threshold, severity, and notification channels</p>
          </div>
          <button type="button" onClick={onClose} className="rounded-lg p-2 text-outline hover:bg-surface-container-low">
            <Icon name="close" size={20} />
          </button>
        </div>

        <form onSubmit={(e) => void handleSubmit(e)} className="flex min-h-0 flex-1 flex-col">
          <div className="min-h-0 flex-1 space-y-4 overflow-y-auto px-6 py-5">
            {mode === "edit" && values.harActionLabel ? (
              <div className="rounded-lg border border-outline-variant/70 bg-surface-container-low px-3 py-2">
                <p className="font-label text-label-sm text-outline">HAR action</p>
                <p className="text-body-md font-medium text-on-surface">{values.harActionLabel}</p>
              </div>
            ) : (
              <div>
                <span className="mb-2 block font-label text-label-sm text-outline">HAR action</span>
                <div className="grid max-h-48 grid-cols-2 gap-2 overflow-y-auto pr-1">
                  {actions.map((action) => {
                    const selected = values.harActionLabel === (action.harActionLabel ?? action.label);
                    return (
                      <button
                        key={action.harActionLabel ?? action.label}
                        type="button"
                        onClick={() => handleActionChange(action)}
                        className={cn(
                          "flex items-center gap-2 rounded-lg border px-3 py-2 text-left text-body-sm transition-colors",
                          selected
                            ? "border-primary bg-primary-fixed/40 text-primary"
                            : "border-outline-variant/70 hover:bg-surface-container-low",
                        )}
                      >
                        <Icon name={action.icon} size={16} />
                        <span className="line-clamp-2">{action.label}</span>
                      </button>
                    );
                  })}
                </div>
              </div>
            )}

            <label className="block">
              <span className="mb-1.5 block font-label text-label-sm text-outline">Rule title</span>
              <input
                value={values.title}
                onChange={(e) => setValues((v) => ({ ...v, title: e.target.value }))}
                className="w-full rounded-lg border border-outline-variant/70 bg-surface-container-low px-3 py-2 text-body-sm focus:border-primary focus:outline-none focus:ring-1 focus:ring-primary/30"
              />
            </label>

            <label className="block">
              <span className="mb-1.5 block font-label text-label-sm text-outline">Send notification when confidence ≥</span>
              <div className="flex items-center gap-3">
                <input
                  type="range"
                  min={5}
                  max={95}
                  step={5}
                  value={harConfidencePct}
                  onChange={(e) =>
                    setValues((v) => ({
                      ...v,
                      confidenceThreshold: Number(e.target.value) / 100,
                    }))
                  }
                  className="h-2 flex-1 cursor-pointer accent-primary"
                />
                <span className="w-12 shrink-0 text-body-sm font-semibold text-on-surface">{harConfidencePct}%</span>
              </div>
            </label>

            <div className="grid gap-4 sm:grid-cols-2">
              <label className="block">
                <span className="mb-1.5 block font-label text-label-sm text-outline">Zone</span>
                <input
                  value={values.zone}
                  onChange={(e) => setValues((v) => ({ ...v, zone: e.target.value }))}
                  className="w-full rounded-lg border border-outline-variant/70 bg-surface-container-low px-3 py-2 text-body-sm focus:border-primary focus:outline-none"
                />
              </label>
              <label className="block">
                <span className="mb-1.5 block font-label text-label-sm text-outline">Alert type</span>
                <select
                  value={values.severity}
                  onChange={(e) =>
                    setValues((v) => ({
                      ...v,
                      severity: e.target.value as "CRITICAL" | "WARNING" | "INFO",
                    }))
                  }
                  className="w-full rounded-lg border border-outline-variant/70 bg-surface-container-low px-3 py-2 text-body-sm"
                >
                  <option value="INFO">Info</option>
                  <option value="WARNING">Warning</option>
                  <option value="CRITICAL">Critical</option>
                </select>
              </label>
            </div>

            <label className="block">
              <span className="mb-1.5 block font-label text-label-sm text-outline">Email template</span>
              <select
                value={values.emailTemplateId ?? ""}
                onChange={(e) => setValues((v) => ({ ...v, emailTemplateId: e.target.value || undefined }))}
                className="w-full rounded-lg border border-outline-variant/70 bg-surface-container-low px-3 py-2 text-body-sm"
              >
                <option value="">Default for action</option>
                {templatesForCase.map((t) => (
                  <option key={t.id} value={t.id}>
                    {t.name} {t.isBuiltin ? "(built-in)" : "(custom)"}
                  </option>
                ))}
              </select>
            </label>

            <div className="flex flex-wrap gap-4">
              <label className="flex items-center gap-2 text-body-sm">
                <input
                  type="checkbox"
                  checked={values.enabled}
                  onChange={(e) => setValues((v) => ({ ...v, enabled: e.target.checked }))}
                  className="rounded border-outline-variant"
                />
                Enabled
              </label>
              <label className="flex items-center gap-2 text-body-sm" title="Bell, timeline, and notification panel">
                <input
                  type="checkbox"
                  checked={values.notifyInApp}
                  onChange={(e) => setValues((v) => ({ ...v, notifyInApp: e.target.checked }))}
                  className="rounded border-outline-variant"
                />
                In-app
              </label>
              <label className="flex items-center gap-2 text-body-sm">
                <input
                  type="checkbox"
                  checked={values.notifyEmail}
                  onChange={(e) => setValues((v) => ({ ...v, notifyEmail: e.target.checked }))}
                  className="rounded border-outline-variant"
                />
                Email
              </label>
              <label className="flex items-center gap-2 text-body-sm">
                <input
                  type="checkbox"
                  checked={values.notifyTelegram}
                  onChange={(e) => setValues((v) => ({ ...v, notifyTelegram: e.target.checked }))}
                  className="rounded border-outline-variant"
                />
                Telegram
              </label>
            </div>

            {error && <p className="text-body-sm text-error">{error}</p>}
          </div>

          <div className="flex shrink-0 justify-end gap-3 border-t border-outline-variant/60 px-6 py-4">
            <Button type="button" variant="ghost" onClick={onClose}>
              Cancel
            </Button>
            <Button type="submit" icon="save" disabled={saving}>
              {saving ? "Saving…" : mode === "create" ? "Create Rule" : "Save Changes"}
            </Button>
          </div>
        </form>
      </div>
    </div>
  );
}
