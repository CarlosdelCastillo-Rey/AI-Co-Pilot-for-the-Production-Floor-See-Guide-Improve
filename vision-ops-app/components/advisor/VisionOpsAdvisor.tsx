"use client";

import { usePathname } from "next/navigation";
import { useCallback, useEffect, useRef, useState } from "react";
import { Icon } from "@/components/ui/Icon";
import { advisorChat, fetchAdvisorWelcome, fetchRealtimeEvents } from "@/lib/api";
import type { RealtimeEvent } from "@/lib/types";
import { cn } from "@/lib/cn";
import { DEFAULT_ROUTE } from "@/lib/navigation";

type ChatMessage = { role: "user" | "assistant"; text: string };

const PAGE_ROUTES: { path: string; page: string; title: string }[] = [
  { path: "/live", page: "live", title: "Live Streams" },
  { path: "/har-hitl", page: "har-hitl", title: "Person HITL" },
  { path: "/timeline", page: "timeline", title: "Timeline" },
  { path: "/analytics", page: "analytics", title: "Analytics" },
  { path: "/alerts", page: "alerts", title: "Alerts & Rules" },
  { path: "/settings", page: "settings", title: "Plant Settings" },
];

function pageFromPath(pathname: string): { page: string; title: string } {
  const match = PAGE_ROUTES.find((r) => pathname.startsWith(r.path));
  if (match) return { page: match.page, title: match.title };
  return { page: "dashboard", title: "Dashboard" };
}

export function VisionOpsAdvisor() {
  const pathname = usePathname();
  const [open, setOpen] = useState(false);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [quickPrompts, setQuickPrompts] = useState<string[]>([]);
  const [welcomeLoading, setWelcomeLoading] = useState(false);
  const [screenAlerts, setScreenAlerts] = useState<RealtimeEvent[]>([]);
  const listRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);
  const welcomedForPage = useRef<string | null>(null);

  const { page, title } = pageFromPath(pathname ?? DEFAULT_ROUTE);

  const loadWelcome = useCallback(async () => {
    setWelcomeLoading(true);
    const [welcome, alerts] = await Promise.all([
      fetchAdvisorWelcome(page, title),
      fetchRealtimeEvents(6),
    ]);
    setScreenAlerts(alerts);
    if (welcome) {
      const text = [welcome.intro, welcome.pageContext, welcome.status]
        .filter(Boolean)
        .join("\n\n");
      setMessages([{ role: "assistant", text }]);
      setQuickPrompts(welcome.quickPrompts);
      welcomedForPage.current = `${page}:${title}`;
    } else {
      setMessages([
        {
          role: "assistant",
          text: `VisionOps AI — ops advisor for ${title}. Ask about safety, rules, cameras, KPIs, or cost impact on this screen.`,
        },
      ]);
      setQuickPrompts(["What can you help with here?", "What needs attention now?"]);
    }
    setWelcomeLoading(false);
  }, [page, title]);

  useEffect(() => {
    if (!open) return;
    const key = `${page}:${title}`;
    if (welcomedForPage.current !== key) {
      void loadWelcome();
    }
    inputRef.current?.focus();
  }, [open, page, title, loadWelcome]);

  useEffect(() => {
    if (!open) {
      welcomedForPage.current = null;
    }
  }, [open]);

  useEffect(() => {
    listRef.current?.scrollTo({ top: listRef.current.scrollHeight, behavior: "smooth" });
  }, [messages, loading, welcomeLoading]);

  const send = useCallback(
    async (text: string) => {
      const msg = text.trim();
      if (!msg || loading) return;
      setInput("");
      setMessages((m) => [...m, { role: "user", text: msg }]);
      setLoading(true);
      try {
        const res = await advisorChat({
          message: msg,
          page,
          pageTitle: title,
          alerts: screenAlerts.map((e) => ({
            id: e.id,
            title: e.title,
            severity: e.severity,
            time: e.time,
            description: e.description,
          })),
        });
        setMessages((m) => [...m, { role: "assistant", text: res.reply }]);
      } catch (err) {
        const detail = err instanceof Error ? err.message : "Advisor request failed";
        setMessages((m) => [
          ...m,
          {
            role: "assistant",
            text: detail.startsWith("ERROR:") ? detail : `Advisor unavailable: ${detail}`,
          },
        ]);
      } finally {
        setLoading(false);
      }
    },
    [loading, page, title, screenAlerts],
  );

  if (pathname?.startsWith("/login")) return null;

  const isAnalytics = pathname?.startsWith("/analytics");
  const anchorClass = isAnalytics
    ? "fixed bottom-[7.5rem] right-6 z-50 sm:right-8"
    : "fixed bottom-6 right-6 z-50 sm:bottom-8 sm:right-8";

  const showQuickPrompts = !loading && !welcomeLoading && quickPrompts.length > 0;

  return (
    <>
      {!open && (
        <button
          type="button"
          onClick={() => setOpen(true)}
          className={cn(
            "advisor-fab group flex items-center gap-2 rounded-full border border-primary/30 bg-primary px-4 py-3 text-on-primary shadow-overlay transition-transform hover:scale-[1.02] active:scale-[0.98]",
            anchorClass,
          )}
          aria-label="Talk with VisionOps AI"
        >
          <span className="advisor-sonar pointer-events-none absolute inset-0 rounded-full" aria-hidden />
          <span className="advisor-sonar advisor-sonar-delay pointer-events-none absolute inset-0 rounded-full" aria-hidden />
          <Icon name="smart_toy" size={22} className="relative z-10" />
          <span className="relative z-10 hidden font-label text-label-sm font-semibold sm:inline">
            VisionOps AI
          </span>
        </button>
      )}

      {open && (
        <div
          className={cn(
            "flex w-[min(22rem,calc(100vw-1.5rem))] flex-col overflow-hidden rounded-2xl border border-outline-variant bg-surface-container-lowest shadow-overlay",
            anchorClass,
          )}
          role="dialog"
          aria-label="VisionOps AI advisor"
        >
          <header className="flex items-center justify-between border-b border-outline-variant bg-primary px-4 py-3 text-on-primary">
            <div className="flex items-center gap-2">
              <Icon name="smart_toy" size={20} />
              <div>
                <p className="font-headline text-body-sm font-semibold">VisionOps AI</p>
                <p className="font-label text-[10px] opacity-80">
                  Viewing: {title}
                </p>
              </div>
            </div>
            <button
              type="button"
              onClick={() => setOpen(false)}
              className="rounded-lg p-1 hover:bg-on-primary/10"
              aria-label="Close"
            >
              <Icon name="close" size={20} />
            </button>
          </header>

          <div ref={listRef} className="max-h-72 min-h-[10rem] flex-1 space-y-3 overflow-y-auto px-4 py-3">
            {welcomeLoading && messages.length === 0 && (
              <p className="font-label text-label-sm text-outline animate-pulse">Loading context…</p>
            )}
            {messages.map((m, i) => (
              <div
                key={i}
                className={cn(
                  "rounded-lg px-3 py-2 text-body-sm leading-snug whitespace-pre-wrap",
                  m.role === "user"
                    ? "ml-6 bg-primary-container/20 text-on-surface"
                    : "mr-1 bg-surface-container-low text-on-surface",
                )}
              >
                {m.text}
              </div>
            ))}
            {loading && (
              <p className="font-label text-label-sm text-outline animate-pulse">Thinking…</p>
            )}
          </div>

          {showQuickPrompts && (
            <div className="flex flex-wrap gap-1.5 border-t border-outline-variant/50 px-3 py-2">
              {quickPrompts.map((q) => (
                <button
                  key={q}
                  type="button"
                  onClick={() => void send(q)}
                  className="rounded-full border border-outline-variant/60 px-2.5 py-1 font-label text-[10px] text-on-surface-variant hover:border-primary/40 hover:text-primary"
                >
                  {q}
                </button>
              ))}
            </div>
          )}

          <form
            className="flex gap-2 border-t border-outline-variant p-3"
            onSubmit={(e) => {
              e.preventDefault();
              void send(input);
            }}
          >
            <input
              ref={inputRef}
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder={`Ask about ${title}…`}
              disabled={loading || welcomeLoading}
              className="min-w-0 flex-1 rounded-lg border border-outline-variant/70 bg-surface-container-low px-3 py-2 text-body-sm focus:border-primary focus:outline-none focus:ring-1 focus:ring-primary/30"
            />
            <button
              type="submit"
              disabled={loading || welcomeLoading || !input.trim()}
              className="rounded-lg bg-primary px-3 py-2 text-on-primary disabled:opacity-50"
              aria-label="Send"
            >
              <Icon name="send" size={18} />
            </button>
          </form>
        </div>
      )}
    </>
  );
}
