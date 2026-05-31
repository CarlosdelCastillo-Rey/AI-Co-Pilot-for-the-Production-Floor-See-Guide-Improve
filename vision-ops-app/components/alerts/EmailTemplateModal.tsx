"use client";

import { useEffect, useState } from "react";
import { Button } from "@/components/ui/Button";
import { Icon } from "@/components/ui/Icon";
import type { EmailTemplateApi } from "@/lib/api";
import { cn } from "@/lib/cn";

export type EmailTemplateFormValues = {
  name: string;
  baseTemplateId: string;
  subject: string;
  headline: string;
  body: string;
  category: string;
  footerReason: string;
};

interface EmailTemplateModalProps {
  open: boolean;
  mode: "create" | "edit";
  templates: EmailTemplateApi[];
  initial?: EmailTemplateApi | null;
  onClose: () => void;
  onSubmit: (values: EmailTemplateFormValues) => Promise<boolean>;
}

export function EmailTemplateModal({
  open,
  mode,
  templates,
  initial,
  onClose,
  onSubmit,
}: EmailTemplateModalProps) {
  const builtins = templates.filter((t) => t.isBuiltin);
  const [values, setValues] = useState<EmailTemplateFormValues>({
    name: "",
    baseTemplateId: builtins[0]?.id ?? "",
    subject: "",
    headline: "",
    body: "",
    category: "",
    footerReason: "",
  });
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!open) return;
    if (mode === "edit" && initial) {
      setValues({
        name: initial.name,
        baseTemplateId: initial.id,
        subject: initial.subject,
        headline: initial.headline,
        body: initial.body,
        category: initial.category,
        footerReason: initial.footerReason,
      });
    } else if (builtins[0]) {
      const b = builtins[0];
      setValues({
        name: `${b.name} (Custom)`,
        baseTemplateId: b.id,
        subject: b.subject,
        headline: b.headline,
        body: b.body,
        category: b.category,
        footerReason: b.footerReason,
      });
    }
    setError(null);
  }, [open, mode, initial, builtins]);

  if (!open) return null;

  const handleBaseChange = (id: string) => {
    const base = templates.find((t) => t.id === id);
    if (!base) return;
    setValues((v) => ({
      ...v,
      baseTemplateId: id,
      ...(mode === "create"
        ? {
            subject: base.subject,
            headline: base.headline,
            body: base.body,
            category: base.category,
            footerReason: base.footerReason,
          }
        : {}),
    }));
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!values.name.trim()) {
      setError("Template name is required.");
      return;
    }
    setSaving(true);
    setError(null);
    const ok = await onSubmit(values);
    setSaving(false);
    if (ok) onClose();
    else setError("Could not save template.");
  };

  const readOnlyContent = mode === "edit" && initial?.isBuiltin;

  return (
    <div className="fixed inset-0 z-[100] flex items-center justify-center p-4">
      <button type="button" className="absolute inset-0 bg-on-surface/40 backdrop-blur-sm" onClick={onClose} aria-label="Close" />
      <div className="relative z-10 max-h-[90vh] w-full max-w-2xl overflow-y-auto rounded-card border border-outline-variant bg-surface-container-lowest shadow-overlay">
        <div className="sticky top-0 flex items-center justify-between border-b border-outline-variant bg-surface-container-lowest px-6 py-4">
          <div>
            <h2 className="font-headline text-headline-md text-on-surface">
              {mode === "create" ? "Create Email Template" : "Edit Email Template"}
            </h2>
            <p className="text-body-sm text-outline">VisionOps branded transactional layout</p>
          </div>
          <button type="button" onClick={onClose} className="rounded-lg p-2 text-outline hover:bg-surface-container-low">
            <Icon name="close" size={20} />
          </button>
        </div>

        <form onSubmit={(e) => void handleSubmit(e)} className="space-y-4 px-6 py-5">
          {mode === "create" && (
            <label className="block">
              <span className="mb-1.5 block font-label text-label-sm text-outline">Base template</span>
              <select
                value={values.baseTemplateId}
                onChange={(e) => handleBaseChange(e.target.value)}
                className="w-full rounded-lg border border-outline-variant/70 bg-surface-container-low px-3 py-2 text-body-sm"
              >
                {builtins.map((t) => (
                  <option key={t.id} value={t.id}>
                    {t.name} ({t.slug})
                  </option>
                ))}
              </select>
            </label>
          )}

          <label className="block">
            <span className="mb-1.5 block font-label text-label-sm text-outline">Template name</span>
            <input
              value={values.name}
              onChange={(e) => setValues((v) => ({ ...v, name: e.target.value }))}
              className="w-full rounded-lg border border-outline-variant/70 bg-surface-container-low px-3 py-2 text-body-sm"
            />
          </label>

          <label className="block">
            <span className="mb-1.5 block font-label text-label-sm text-outline">Email subject</span>
            <input
              value={values.subject}
              disabled={readOnlyContent}
              onChange={(e) => setValues((v) => ({ ...v, subject: e.target.value }))}
              className={cn("w-full rounded-lg border border-outline-variant/70 bg-surface-container-low px-3 py-2 font-label text-label-sm", readOnlyContent && "opacity-60")}
            />
          </label>

          <div className="grid gap-4 sm:grid-cols-2">
            <label className="block">
              <span className="mb-1.5 block font-label text-label-sm text-outline">Category label</span>
              <input
                value={values.category}
                disabled={readOnlyContent}
                onChange={(e) => setValues((v) => ({ ...v, category: e.target.value }))}
                className="w-full rounded-lg border border-outline-variant/70 bg-surface-container-low px-3 py-2 text-body-sm"
              />
            </label>
            <label className="block">
              <span className="mb-1.5 block font-label text-label-sm text-outline">Footer reason</span>
              <input
                value={values.footerReason}
                disabled={readOnlyContent}
                onChange={(e) => setValues((v) => ({ ...v, footerReason: e.target.value }))}
                className="w-full rounded-lg border border-outline-variant/70 bg-surface-container-low px-3 py-2 text-body-sm"
              />
            </label>
          </div>

          <label className="block">
            <span className="mb-1.5 block font-label text-label-sm text-outline">Headline</span>
            <input
              value={values.headline}
              disabled={readOnlyContent}
              onChange={(e) => setValues((v) => ({ ...v, headline: e.target.value }))}
              className="w-full rounded-lg border border-outline-variant/70 bg-surface-container-low px-3 py-2 text-body-sm"
            />
          </label>

          <label className="block">
            <span className="mb-1.5 block font-label text-label-sm text-outline">
              Body (use **bold** markers)
            </span>
            <textarea
              value={values.body}
              disabled={readOnlyContent}
              onChange={(e) => setValues((v) => ({ ...v, body: e.target.value }))}
              rows={4}
              className="w-full rounded-lg border border-outline-variant/70 bg-surface-container-low px-3 py-2 text-body-sm"
            />
          </label>

          {readOnlyContent && (
            <p className="text-body-sm text-outline">
              Built-in templates cannot be edited. Clone by creating a new template from this base.
            </p>
          )}

          {error && <p className="text-body-sm text-error">{error}</p>}

          <div className="flex justify-end gap-3 border-t border-outline-variant/60 pt-4">
            <Button type="button" variant="ghost" onClick={onClose}>Cancel</Button>
            {!readOnlyContent && (
              <Button type="submit" icon="mail" disabled={saving}>
                {saving ? "Saving…" : mode === "create" ? "Create Template" : "Save Template"}
              </Button>
            )}
          </div>
        </form>
      </div>
    </div>
  );
}
