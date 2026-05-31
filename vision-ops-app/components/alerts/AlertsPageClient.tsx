"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import { AppShell } from "@/components/layout/AppShell";
import { AlertRuleModal, type AlertRuleFormValues } from "@/components/alerts/AlertRuleModal";
import { Badge } from "@/components/ui/Badge";
import { Button } from "@/components/ui/Button";
import { Icon } from "@/components/ui/Icon";
import { SectionHeader } from "@/components/ui/SectionHeader";
import { Switch } from "@/components/ui/Switch";
import { EmailTemplateModal, type EmailTemplateFormValues } from "@/components/alerts/EmailTemplateModal";
import {
  createAlertRule,
  createEmailTemplate,
  deleteAlertRule,
  deleteEmailTemplate,
  fetchAlertActions,
  fetchAlertRules,
  fetchEmailNotificationStatus,
  fetchEmailTemplates,
  fetchTelemetry,
  previewEmailTemplate,
  sendTestAlertEmail,
  toggleAlertRule,
  updateAlertRule,
  updateEmailTemplate,
  type AlertActionApi,
  type AlertCaseType,
  type AlertRuleApi,
  type EmailNotificationStatus,
  type EmailTemplateApi,
  type TelemetryMetric,
} from "@/lib/api";
import { cn } from "@/lib/cn";

export function AlertsPageClient() {
  const [rules, setRules] = useState<AlertRuleApi[]>([]);
  const [actions, setActions] = useState<AlertActionApi[]>([]);
  const [emailTemplates, setEmailTemplates] = useState<EmailTemplateApi[]>([]);
  const [loading, setLoading] = useState(true);
  const [search, setSearch] = useState("");
  const [modalOpen, setModalOpen] = useState(false);
  const [editRule, setEditRule] = useState<AlertRuleApi | null>(null);
  const [templateModalOpen, setTemplateModalOpen] = useState(false);
  const [editTemplate, setEditTemplate] = useState<EmailTemplateApi | null>(null);
  const [previewHtml, setPreviewHtml] = useState<string | null>(null);
  const [emailTemplatesOpen, setEmailTemplatesOpen] = useState(false);

  const load = useCallback(async () => {
    const [rulesData, actionsData, templatesData] = await Promise.all([
      fetchAlertRules(),
      fetchAlertActions(),
      fetchEmailTemplates(),
    ]);
    setRules(rulesData);
    setActions(actionsData);
    setEmailTemplates(templatesData);
    setLoading(false);
  }, []);

  useEffect(() => {
    void load();
    const id = setInterval(() => void load(), 30_000);
    return () => clearInterval(id);
  }, [load]);

  const filteredRules = useMemo(() => {
    const q = search.trim().toLowerCase();
    if (!q) return rules;
    return rules.filter(
      (r) =>
        r.title.toLowerCase().includes(q) ||
        r.zone.toLowerCase().includes(q) ||
        (r.caseType ?? "").toLowerCase().includes(q) ||
        r.description.toLowerCase().includes(q),
    );
  }, [rules, search]);

  const activeCount = rules.filter((r) => r.enabled).length;
  const actionLabel = (caseType?: string) =>
    actions.find((a) => a.caseType === caseType)?.label ?? caseType ?? "Unknown";

  const handleToggle = async (id: string) => {
    const updated = await toggleAlertRule(id);
    if (updated) setRules((prev) => prev.map((r) => (r.id === id ? updated : r)));
  };

  const handleDelete = async (rule: AlertRuleApi) => {
    if (!window.confirm(`Delete rule “${rule.title}”?`)) return;
    const ok = await deleteAlertRule(rule.id);
    if (ok) setRules((prev) => prev.filter((r) => r.id !== rule.id));
  };

  const handleSave = async (values: AlertRuleFormValues) => {
    if (editRule) {
      const updated = await updateAlertRule(editRule.id, values);
      if (updated) {
        setRules((prev) => prev.map((r) => (r.id === editRule.id ? updated : r)));
        return true;
      }
      return false;
    }
    const created = await createAlertRule(values);
    if (created) {
      setRules((prev) => [...prev, created]);
      return true;
    }
    return false;
  };

  const openCreate = () => {
    setEditRule(null);
    setModalOpen(true);
  };

  const openEdit = (rule: AlertRuleApi) => {
    setEditRule(rule);
    setModalOpen(true);
  };

  const handleTemplateSave = async (values: EmailTemplateFormValues) => {
    if (editTemplate && !editTemplate.isBuiltin) {
      const updated = await updateEmailTemplate(editTemplate.id, values);
      if (updated) {
        setEmailTemplates((prev) => prev.map((t) => (t.id === editTemplate.id ? updated : t)));
        return true;
      }
      return false;
    }
    const created = await createEmailTemplate({
      name: values.name,
      baseTemplateId: values.baseTemplateId,
      subject: values.subject,
      headline: values.headline,
      body: values.body,
      category: values.category,
      footerReason: values.footerReason,
    });
    if (created) {
      setEmailTemplates((prev) => [...prev, created]);
      return true;
    }
    return false;
  };

  const handlePreviewTemplate = async (template: EmailTemplateApi) => {
    const preview = await previewEmailTemplate(template.id);
    if (preview) setPreviewHtml(preview.html);
  };

  const handleDeleteTemplate = async (template: EmailTemplateApi) => {
    if (template.isBuiltin) return;
    if (!window.confirm(`Delete template “${template.name}”?`)) return;
    const ok = await deleteEmailTemplate(template.id);
    if (ok) setEmailTemplates((prev) => prev.filter((t) => t.id !== template.id));
  };

  return (
    <>
      <AppShell
        searchPlaceholder="Search rules…"
        searchValue={search}
        onSearchChange={setSearch}
      >
        <div className="mx-auto max-w-[1600px] space-y-6 p-6 lg:p-8">
          <header className="flex flex-col gap-4 sm:flex-row sm:items-end sm:justify-between">
            <div>
              <h2 className="font-headline text-headline-lg text-on-surface">
                Alert Rules & Integrations
              </h2>
              <p className="mt-1 text-body-md text-outline">
                Enable vision actions — operator idle, left position, forklift zones — and route to
                email.
              </p>
            </div>
            <Button icon="add" className="rounded-lg" onClick={openCreate}>
              Create New Rule
            </Button>
          </header>

          <div className="grid grid-cols-2 gap-4 sm:grid-cols-4">
            <StatChip label="Active rules" value={String(activeCount)} />
            <StatChip label="Total rules" value={String(rules.length)} />
            <StatChip label="Action types" value={String(actions.length)} />
            <StatChip
              label="Configured actions"
              value={String(new Set(rules.map((r) => r.caseType)).size)}
            />
          </div>

          <div className="grid grid-cols-12 gap-6">
            <section className="col-span-12 space-y-4 lg:col-span-8">
              <SectionHeader
                title="Vision Action Rules"
                action={
                  <span className="font-label text-label-sm font-bold text-primary">
                    {activeCount} active
                  </span>
                }
              />
              {loading ? (
                <p className="text-body-sm text-outline">Loading rules…</p>
              ) : filteredRules.length === 0 ? (
                <p className="text-body-sm text-outline">No rules match your search.</p>
              ) : (
                <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
                  {filteredRules.map((rule) => (
                    <RuleCard
                      key={rule.id}
                      rule={rule}
                      actionLabel={actionLabel(rule.caseType)}
                      onToggle={() => void handleToggle(rule.id)}
                      onEdit={() => openEdit(rule)}
                      onDelete={() => void handleDelete(rule)}
                    />
                  ))}
                </div>
              )}
            </section>

            <section className="col-span-12 space-y-4 lg:col-span-4">
              <SectionHeader title="External Notifications" />
              <EmailPanel rules={rules} actions={actions} />
            </section>

            <section className="col-span-12 space-y-4">
              <SectionHeader title="System Telemetry" />
              <TelemetryPanel />
            </section>

            <section className="col-span-12">
              <EmailTemplatesSection
                open={emailTemplatesOpen}
                onToggle={() => setEmailTemplatesOpen((prev) => !prev)}
                templates={emailTemplates}
                onNewTemplate={() => {
                  setEditTemplate(null);
                  setTemplateModalOpen(true);
                }}
                onPreview={(template) => void handlePreviewTemplate(template)}
                onEdit={(template) => {
                  setEditTemplate(template);
                  setTemplateModalOpen(true);
                }}
                onDelete={(template) => void handleDeleteTemplate(template)}
              />
            </section>
          </div>
        </div>
      </AppShell>

      <AlertRuleModal
        open={modalOpen}
        mode={editRule ? "edit" : "create"}
        actions={actions}
        emailTemplates={emailTemplates}
        initial={editRule}
        onClose={() => {
          setModalOpen(false);
          setEditRule(null);
        }}
        onSubmit={handleSave}
      />

      <EmailTemplateModal
        open={templateModalOpen}
        mode={editTemplate ? "edit" : "create"}
        templates={emailTemplates}
        initial={editTemplate}
        onClose={() => {
          setTemplateModalOpen(false);
          setEditTemplate(null);
        }}
        onSubmit={handleTemplateSave}
      />

      {previewHtml && (
        <div className="fixed inset-0 z-[110] flex items-center justify-center p-4">
          <button
            type="button"
            className="absolute inset-0 bg-on-surface/50 backdrop-blur-sm"
            onClick={() => setPreviewHtml(null)}
            aria-label="Close preview"
          />
          <div className="relative z-10 max-h-[90vh] w-full max-w-3xl overflow-hidden rounded-card border border-outline-variant bg-surface-container-lowest shadow-overlay">
            <div className="flex items-center justify-between border-b border-outline-variant px-4 py-3">
              <h3 className="font-headline text-body-md font-semibold">Email preview</h3>
              <button type="button" onClick={() => setPreviewHtml(null)} className="rounded p-2 hover:bg-surface-container-low">
                <Icon name="close" />
              </button>
            </div>
            <iframe
              title="Email preview"
              srcDoc={previewHtml}
              className="h-[70vh] w-full bg-[#eef0f3]"
            />
          </div>
        </div>
      )}
    </>
  );
}

function EmailTemplatesSection({
  open,
  onToggle,
  templates,
  onNewTemplate,
  onPreview,
  onEdit,
  onDelete,
}: {
  open: boolean;
  onToggle: () => void;
  templates: EmailTemplateApi[];
  onNewTemplate: () => void;
  onPreview: (template: EmailTemplateApi) => void;
  onEdit: (template: EmailTemplateApi) => void;
  onDelete: (template: EmailTemplateApi) => void;
}) {
  const customCount = templates.filter((t) => !t.isBuiltin).length;

  return (
    <div className="overflow-hidden rounded-card border border-outline-variant/60 bg-surface-container-lowest">
      <div className="flex items-center gap-3 border-b border-outline-variant bg-surface-container-highest px-4 py-2">
        <button
          type="button"
          className="flex min-w-0 flex-1 items-center gap-2 text-left"
          onClick={onToggle}
          aria-expanded={open}
        >
          <Icon
            name="expand_more"
            size={20}
            className={cn("shrink-0 text-outline transition-transform", open && "rotate-180")}
          />
          <span className="text-label-sm uppercase tracking-wider text-on-surface-variant">
            Email Templates
          </span>
          <span className="font-label text-label-sm text-outline">
            {templates.length} total
            {customCount > 0 ? ` · ${customCount} custom` : ""}
          </span>
        </button>
        {open && (
          <Button variant="outline" icon="mail" className="shrink-0 rounded-lg" onClick={onNewTemplate}>
            New Template
          </Button>
        )}
      </div>

      {open && (
        <div className="grid grid-cols-1 gap-3 p-4 md:grid-cols-2 xl:grid-cols-4">
          {templates.map((template) => (
            <article
              key={template.id}
              className="rounded-card border border-outline-variant/60 bg-surface-container-low p-4"
            >
              <div className="mb-2 flex items-start justify-between gap-2">
                <div>
                  <p className="font-label text-label-sm text-primary">{template.slug}</p>
                  <h4 className="text-body-sm font-semibold text-on-surface">{template.name}</h4>
                </div>
                <Badge variant={template.severityLevel === "critical" ? "critical" : "warning"}>
                  {template.severityLevel}
                </Badge>
              </div>
              <p className="line-clamp-2 text-body-sm text-outline">{template.headline}</p>
              <div className="mt-3 flex flex-wrap gap-2">
                {template.isBuiltin && <Badge variant="info">Built-in</Badge>}
                <Badge variant="zone">{template.caseType}</Badge>
              </div>
              <div className="mt-4 flex gap-2">
                <Button
                  variant="ghost"
                  className="flex-1 py-1 text-label-sm"
                  onClick={() => onPreview(template)}
                >
                  Preview
                </Button>
                {!template.isBuiltin && (
                  <>
                    <button
                      type="button"
                      className="rounded p-1.5 text-outline hover:text-primary"
                      onClick={() => onEdit(template)}
                    >
                      <Icon name="edit" size={16} />
                    </button>
                    <button
                      type="button"
                      className="rounded p-1.5 text-outline hover:text-error"
                      onClick={() => onDelete(template)}
                    >
                      <Icon name="delete" size={16} />
                    </button>
                  </>
                )}
              </div>
            </article>
          ))}
        </div>
      )}
    </div>
  );
}

function StatChip({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-card border border-outline-variant/60 bg-surface-container-lowest px-4 py-3">
      <p className="font-label text-label-sm uppercase text-outline">{label}</p>
      <p className="font-headline text-[22px] font-bold text-on-surface">{value}</p>
    </div>
  );
}

function RuleCard({
  rule,
  actionLabel,
  onToggle,
  onEdit,
  onDelete,
}: {
  rule: AlertRuleApi;
  actionLabel: string;
  onToggle: () => void;
  onEdit: () => void;
  onDelete: () => void;
}) {
  const severityVariant =
    rule.severity === "CRITICAL"
      ? "critical"
      : rule.severity === "WARNING"
        ? "warning"
        : "disabled";

  return (
    <article
      className={cn(
        "flex flex-col justify-between rounded-card border bg-surface-container-lowest p-4 transition-colors",
        rule.enabled ? "border-outline-variant/60 hover:border-primary/40" : "border-outline-variant/40 opacity-80",
      )}
    >
      <div className="mb-4 flex items-start justify-between gap-2">
        <div className="rounded-lg bg-primary-fixed/30 p-2">
          <Icon name={rule.icon} className="text-primary" size={20} />
        </div>
        <div className="flex items-center gap-2">
          <Switch checked={rule.enabled} onChange={onToggle} label={`Toggle ${rule.title}`} />
          <button
            type="button"
            onClick={onEdit}
            className="rounded p-1.5 text-outline hover:bg-surface-container-low hover:text-primary"
            aria-label="Edit rule"
          >
            <Icon name="edit" size={18} />
          </button>
          <button
            type="button"
            onClick={onDelete}
            className="rounded p-1.5 text-outline hover:bg-error-container hover:text-error"
            aria-label="Delete rule"
          >
            <Icon name="delete" size={18} />
          </button>
        </div>
      </div>
      <div>
        <p className="mb-1 font-label text-label-sm text-primary">{actionLabel}</p>
        <h4 className="text-body-md font-semibold text-on-surface">{rule.title}</h4>
        <p className="mt-1 text-body-sm text-outline">{rule.description}</p>
        {rule.updatedBy && (
          <p className="mt-2 text-body-sm text-outline/80">
            Last updated by <span className="font-medium text-on-surface">{rule.updatedBy}</span>
          </p>
        )}
        <div className="mt-4 flex flex-wrap gap-2">
          <Badge variant="zone">{rule.zone}</Badge>
          <Badge variant={severityVariant}>{rule.enabled ? rule.severity : "DISABLED"}</Badge>
          {rule.notifyEmail && (
            <Badge variant="info">
              <Icon name="mail" size={12} className="mr-1 inline" />
              Email
            </Badge>
          )}
        </div>
      </div>
    </article>
  );
}

function EmailPanel({
  rules,
  actions,
}: {
  rules: AlertRuleApi[];
  actions: AlertActionApi[];
}) {
  const [status, setStatus] = useState<EmailNotificationStatus | null>(null);
  const [message, setMessage] = useState<string | null>(null);
  const [sending, setSending] = useState<AlertCaseType | null>(null);

  useEffect(() => {
    void fetchEmailNotificationStatus().then(setStatus);
  }, []);

  const isRuleEnabled = (caseType: AlertCaseType) =>
    rules.some((r) => r.caseType === caseType && r.enabled);

  const handleTest = async (caseType: AlertCaseType) => {
    if (!isRuleEnabled(caseType)) {
      setMessage(`Enable a “${actions.find((a) => a.caseType === caseType)?.label}” rule first.`);
      return;
    }
    setSending(caseType);
    setMessage(null);
    const result = await sendTestAlertEmail(caseType);
    setMessage(result.message);
    setSending(null);
  };

  const connected = status?.status === "ready";
  const dryRun = status?.dryRun;

  return (
    <article className="space-y-5 rounded-card border border-outline-variant/60 bg-surface-container-lowest p-5">
      <div className="flex items-center gap-4">
        <div className="flex h-12 w-12 items-center justify-center rounded-lg bg-primary-fixed/40">
          <Icon name="mail" className="text-primary" />
        </div>
        <div>
          <h4 className="text-body-md font-semibold text-on-surface">Email (MailerSend)</h4>
          <p className="text-body-sm text-outline">Test alerts respect enabled rules</p>
        </div>
      </div>

      <div className="space-y-2 text-body-sm">
        <Row label="From" value={status?.fromEmail ?? "—"} />
        <Row label="To" value={status?.toEmails?.join(", ") ?? "—"} />
        <Row label="Mode" value={dryRun ? "Dry run (no send)" : "Live send"} />
      </div>

      <div>
        <p className="mb-2 font-label text-label-sm font-bold uppercase text-outline">
          Send test alerts
        </p>
        <div className="grid grid-cols-1 gap-2">
          {actions.map((action) => {
            const enabled = isRuleEnabled(action.caseType);
            return (
              <Button
                key={action.caseType}
                variant="outline"
                icon={action.icon}
                className={cn(
                  "justify-start rounded-lg py-2 text-left",
                  !enabled && "opacity-50",
                )}
                disabled={sending !== null}
                onClick={() => void handleTest(action.caseType)}
              >
                {sending === action.caseType
                  ? "Sending…"
                  : `${action.label}${enabled ? "" : " (disabled)"}`}
              </Button>
            );
          })}
        </div>
      </div>

      {message && (
        <p className="rounded-lg border border-outline-variant bg-surface-container-low p-3 text-body-sm">
          {message}
        </p>
      )}

      <div className="border-t border-outline-variant pt-4">
        <div className="flex items-center justify-between">
          <span className="text-body-sm font-medium">Status</span>
          <span
            className={cn(
              "flex items-center gap-1.5 font-label text-label-sm font-bold",
              connected || dryRun ? "text-success" : "text-error",
            )}
          >
            <span
              className={cn(
                "h-2 w-2 rounded-full",
                connected || dryRun ? "bg-success" : "bg-error",
              )}
            />
            {connected ? "CONNECTED" : dryRun ? "DRY-RUN" : "NOT CONFIGURED"}
          </span>
        </div>
      </div>
    </article>
  );
}

function Row({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex justify-between gap-4">
      <span className="text-outline">{label}</span>
      <span className="truncate text-right font-medium text-on-surface">{value}</span>
    </div>
  );
}

function TelemetryPanel() {
  const [metrics, setMetrics] = useState<TelemetryMetric[]>([]);
  const [loading, setLoading] = useState(true);
  const [collectedAt, setCollectedAt] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    setLoading(true);
    const data = await fetchTelemetry();
    if (data) {
      setMetrics(data.metrics);
      setCollectedAt(data.collectedAt);
    }
    setLoading(false);
  }, []);

  useEffect(() => {
    void refresh();
    const id = setInterval(() => void refresh(), 30_000);
    return () => clearInterval(id);
  }, [refresh]);

  if (loading && metrics.length === 0) {
    return (
      <article className="rounded-card border border-outline-variant/60 bg-surface-container-lowest p-5">
        <p className="text-body-sm text-outline">Collecting telemetry…</p>
      </article>
    );
  }

  return (
    <article className="space-y-4 rounded-card border border-outline-variant/60 bg-surface-container-lowest p-5">
      <div className="flex items-center justify-between">
        <p className="font-label text-label-sm text-outline">
          {collectedAt ? `Updated ${new Date(collectedAt).toLocaleTimeString()}` : "Live probes"}
        </p>
        <Button variant="ghost" icon="refresh" onClick={() => void refresh()}>
          Refresh
        </Button>
      </div>
      <div className="grid grid-cols-1 gap-6 md:grid-cols-3">
        {metrics.map((m) => (
          <div key={m.service} className="space-y-3">
            <div className="flex items-center justify-between">
              <span className="text-body-sm text-on-surface-variant">{m.label}</span>
              <span
                className={cn(
                  "font-label text-label-sm font-bold",
                  m.status === "ok" ? "text-primary" : m.status === "degraded" ? "text-warning" : "text-error",
                )}
              >
                {m.value}
              </span>
            </div>
            <div className="flex h-16 items-end gap-1">
              {m.bars.map((h, i) => (
                <div
                  key={i}
                  className={cn(
                    "w-full rounded-t-sm",
                    m.highlight && i === m.bars.length - 1 ? "bg-primary" : "bg-outline-variant/30",
                  )}
                  style={{ height: `${h}%` }}
                />
              ))}
            </div>
            <p className="font-label text-label-sm text-outline">{m.detail}</p>
          </div>
        ))}
      </div>
    </article>
  );
}
