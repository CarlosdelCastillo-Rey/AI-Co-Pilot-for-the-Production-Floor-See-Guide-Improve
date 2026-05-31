"use client";

import { useEffect, useState } from "react";
import { Button } from "@/components/ui/Button";
import { Icon } from "@/components/ui/Icon";
import type { AlertActionApi, AlertCaseType, AlertRuleApi, EmailTemplateApi } from "@/lib/api";
import { cn } from "@/lib/cn";

export type AlertRuleFormValues = {
  title: string;
  description: string;
  zone: string;
  caseType: AlertCaseType;
  severity: "CRITICAL" | "WARNING";
  enabled: boolean;
  notifyEmail: boolean;
  icon: string;
  emailTemplateId?: string;
};

interface AlertRuleModalProps {
  open: boolean;
  mode: "create" | "edit";
  actions: AlertActionApi[];
  emailTemplates: EmailTemplateApi[];
  initial?: AlertRuleApi | null;
  onClose: () => void;
  onSubmit: (values: AlertRuleFormValues) => Promise<boolean>;
}

function defaultsFromAction(action: AlertActionApi, templates: EmailTemplateApi[]): AlertRuleFormValues {
  const tpl = templates.find((t) => t.caseType === action.caseType && t.isBuiltin);
  return {
    title: `${action.label} Detection`,
    description: action.description,
    zone: action.defaultZone,
    caseType: action.caseType,
    severity: action.defaultSeverity as "CRITICAL" | "WARNING",
    enabled: action.defaultEnabled,
    notifyEmail: true,
    icon: action.icon,
    emailTemplateId: tpl?.id,
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
    actions[0] ? defaultsFromAction(actions[0], emailTemplates) : {
      title: "",
      description: "",
      zone: "",
      caseType: "unknown",
      severity: "WARNING",
      enabled: true,
      notifyEmail: true,
      icon: "warning",
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
        caseType: (initial.caseType ?? "unknown") as AlertCaseType,
        severity: initial.severity === "CRITICAL" ? "CRITICAL" : "WARNING",
        enabled: initial.enabled,
        notifyEmail: initial.notifyEmail ?? true,
        icon: initial.icon,
        emailTemplateId: initial.emailTemplateId ?? undefined,
      });
    } else if (actions[0]) {
      setValues(defaultsFromAction(actions[0], emailTemplates));
    }
    setError(null);
  }, [open, mode, initial, actions, emailTemplates]);

  const templatesForCase = emailTemplates.filter(
    (t) => t.caseType === values.caseType && t.enabled,
  );

  if (!open) return null;

  const handleActionChange = (caseType: AlertCaseType) => {
    const action = actions.find((a) => a.caseType === caseType);
    if (!action) return;
    setValues((v) => ({
      ...v,
      caseType,
      icon: action.icon,
      emailTemplateId:
        emailTemplates.find((t) => t.caseType === caseType && t.isBuiltin)?.id ??
        emailTemplates.find((t) => t.caseType === caseType)?.id,
      ...(mode === "create"
        ? {
            title: `${action.label} Detection`,
            description: action.description,
            zone: action.defaultZone,
            severity: action.defaultSeverity as "CRITICAL" | "WARNING",
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
    setSaving(true);
    setError(null);
    const ok = await onSubmit(values);
    setSaving(false);
    if (ok) onClose();
    else setError("Could not save rule. Check alerting service is running.");
  };

  return (
    <div className="fixed inset-0 z-[100] flex items-center justify-center p-4">
      <button
        type="button"
        className="absolute inset-0 bg-on-surface/40 backdrop-blur-sm"
        onClick={onClose}
        aria-label="Close"
      />
      <div className="relative z-10 w-full max-w-lg rounded-card border border-outline-variant bg-surface-container-lowest shadow-overlay">
        <div className="flex items-center justify-between border-b border-outline-variant px-6 py-4">
          <div>
            <h2 className="font-headline text-headline-md text-on-surface">
              {mode === "create" ? "Create Alert Rule" : "Edit Alert Rule"}
            </h2>
            <p className="text-body-sm text-outline">Maps vision events to email notifications</p>
          </div>
          <button type="button" onClick={onClose} className="rounded-lg p-2 text-outline hover:bg-surface-container-low">
            <Icon name="close" size={20} />
          </button>
        </div>

        <form onSubmit={(e) => void handleSubmit(e)} className="space-y-4 px-6 py-5">
          <div>
            <span className="mb-2 block font-label text-label-sm text-outline">Action type</span>
            <div className="grid grid-cols-2 gap-2">
              {actions.map((action) => (
                <button
                  key={action.caseType}
                  type="button"
                  onClick={() => handleActionChange(action.caseType)}
                  className={cn(
                    "flex items-center gap-2 rounded-lg border px-3 py-2 text-left text-body-sm transition-colors",
                    values.caseType === action.caseType
                      ? "border-primary bg-primary-fixed/40 text-primary"
                      : "border-outline-variant/70 hover:bg-surface-container-low",
                  )}
                >
                  <Icon name={action.icon} size={16} />
                  {action.label}
                </button>
              ))}
            </div>
          </div>

          <label className="block">
            <span className="mb-1.5 block font-label text-label-sm text-outline">Rule title</span>
            <input
              value={values.title}
              onChange={(e) => setValues((v) => ({ ...v, title: e.target.value }))}
              className="w-full rounded-lg border border-outline-variant/70 bg-surface-container-low px-3 py-2 text-body-sm focus:border-primary focus:outline-none focus:ring-1 focus:ring-primary/30"
            />
          </label>

          <label className="block">
            <span className="mb-1.5 block font-label text-label-sm text-outline">Description</span>
            <textarea
              value={values.description}
              onChange={(e) => setValues((v) => ({ ...v, description: e.target.value }))}
              rows={3}
              className="w-full rounded-lg border border-outline-variant/70 bg-surface-container-low px-3 py-2 text-body-sm focus:border-primary focus:outline-none focus:ring-1 focus:ring-primary/30"
            />
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
              <span className="mb-1.5 block font-label text-label-sm text-outline">Severity</span>
              <select
                value={values.severity}
                onChange={(e) =>
                  setValues((v) => ({
                    ...v,
                    severity: e.target.value as "CRITICAL" | "WARNING",
                  }))
                }
                className="w-full rounded-lg border border-outline-variant/70 bg-surface-container-low px-3 py-2 text-body-sm"
              >
                <option value="WARNING">Warning</option>
                <option value="CRITICAL">Critical</option>
              </select>
            </label>
          </div>

          <label className="block">
            <span className="mb-1.5 block font-label text-label-sm text-outline">Email template</span>
            <select
              value={values.emailTemplateId ?? ""}
              onChange={(e) =>
                setValues((v) => ({ ...v, emailTemplateId: e.target.value || undefined }))
              }
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
            <label className="flex items-center gap-2 text-body-sm">
              <input
                type="checkbox"
                checked={values.notifyEmail}
                onChange={(e) => setValues((v) => ({ ...v, notifyEmail: e.target.checked }))}
                className="rounded border-outline-variant"
              />
              Email notifications
            </label>
          </div>

          {error && <p className="text-body-sm text-error">{error}</p>}

          <div className="flex justify-end gap-3 border-t border-outline-variant/60 pt-4">
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
