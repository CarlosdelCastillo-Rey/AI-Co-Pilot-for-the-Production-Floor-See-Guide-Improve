"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { cameraAdvisorChat } from "@/lib/api";
import { Icon } from "@/components/ui/Icon";
import { cn } from "@/lib/cn";

type ChatMessage = { role: "user" | "assistant"; text: string };

type Props = {
  cameraId: string;
  sessionId?: string;
  accent?: string;
  /** When true, fills the right column card (no outer border-t). */
  embedded?: boolean;
};

export function CameraHarChat({ cameraId, sessionId, accent = "#81C784", embedded = false }: Props) {
  const [message, setMessage] = useState("");
  const [messages, setMessages] = useState<ChatMessage[]>([
    {
      role: "assistant",
      text: "Ask about this feed — current activity, inference log, or alerts for this camera.",
    },
  ]);
  const [loading, setLoading] = useState(false);
  const [expanded, setExpanded] = useState(true);
  const listRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    if (!expanded || !listRef.current) return;
    listRef.current.scrollTo({ top: listRef.current.scrollHeight, behavior: "smooth" });
  }, [messages, loading, expanded]);

  useEffect(() => {
    if (expanded) inputRef.current?.focus({ preventScroll: true });
  }, [expanded]);

  const send = useCallback(async () => {
    const text = message.trim();
    if (!text || loading) return;
    setMessage("");
    setMessages((m) => [...m, { role: "user", text }]);
    setLoading(true);
    setExpanded(true);
    const res = await cameraAdvisorChat(cameraId, text, sessionId);
    setMessages((m) => [
      ...m,
      {
        role: "assistant",
        text: res?.reply ?? "Could not reach the camera advisor. Is alerting (:8001) running?",
      },
    ]);
    setLoading(false);
  }, [cameraId, loading, message, sessionId]);

  const handleListWheel = useCallback(
    (e: React.WheelEvent<HTMLDivElement>) => {
      e.stopPropagation();
      const el = e.currentTarget;
      const atTop = el.scrollTop <= 0;
      const atBottom = el.scrollTop + el.clientHeight >= el.scrollHeight - 2;
      if (atTop && e.deltaY < 0 && expanded) {
        setExpanded(false);
        return;
      }
      if ((atTop && e.deltaY < 0) || (atBottom && e.deltaY > 0)) {
        e.preventDefault();
      }
    },
    [expanded],
  );

  return (
    <div
      className={cn(
        "flex min-h-0 flex-col bg-surface-container-low",
        embedded
          ? "h-full overflow-hidden rounded-lg border border-outline-variant/50"
          : "border-t border-outline-variant/50",
      )}
    >
      <button
        type="button"
        onClick={() => setExpanded((e) => !e)}
        className="flex w-full items-center justify-between px-4 py-3 text-left transition-colors hover:bg-surface-container-high/40"
        aria-expanded={expanded}
      >
        <span className="flex items-center gap-2 text-body-sm font-semibold text-on-surface">
          <span
            className="flex h-8 w-8 items-center justify-center rounded-full"
            style={{ backgroundColor: `${accent}22`, color: accent }}
          >
            <Icon name="smart_toy" size={18} />
          </span>
          Camera assistant
          {messages.length > 1 && !expanded && (
            <span className="rounded-full bg-primary/15 px-2 py-0.5 font-label text-[10px] font-bold text-primary">
              {messages.length - 1} msgs
            </span>
          )}
        </span>
        <span className="flex items-center gap-2 text-outline">
          <span className="hidden font-label text-[10px] sm:inline">
            {expanded ? "Scroll up to minimize" : "Tap to open"}
          </span>
          <Icon name={expanded ? "expand_less" : "expand_more"} size={22} />
        </span>
      </button>

      {expanded && (
        <div className="flex min-h-0 flex-1 flex-col border-t border-outline-variant/40 px-3 pb-3 pt-2">
          <div
            ref={listRef}
            onWheel={handleListWheel}
            className={cn(
              "min-h-[10rem] flex-1 space-y-3 overflow-y-auto overscroll-y-contain rounded-xl border border-outline-variant/50 bg-surface-container-lowest p-3 scroll-smooth",
              embedded ? "max-h-none" : "h-80 max-h-[min(50vh,28rem)]",
            )}
          >
            {messages.map((m, i) => (
              <div
                key={i}
                className={cn(
                  "flex",
                  m.role === "user" ? "justify-end" : "justify-start",
                )}
              >
                <div
                  className={cn(
                    "max-w-[85%] rounded-2xl px-4 py-2.5 text-body-sm leading-relaxed whitespace-pre-wrap shadow-sm",
                    m.role === "user"
                      ? "rounded-br-md text-on-primary"
                      : "rounded-bl-md border border-outline-variant/40 bg-surface text-on-surface",
                  )}
                  style={m.role === "user" ? { backgroundColor: accent } : undefined}
                >
                  {m.text}
                </div>
              </div>
            ))}
            {loading && (
              <div className="flex justify-start">
                <div className="rounded-2xl rounded-bl-md border border-outline-variant/40 bg-surface px-4 py-2.5">
                  <span className="inline-flex items-center gap-2 font-label text-label-sm text-outline">
                    <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-outline" />
                    <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-outline [animation-delay:150ms]" />
                    <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-outline [animation-delay:300ms]" />
                    Thinking…
                  </span>
                </div>
              </div>
            )}
          </div>

          <div className="mt-2 flex shrink-0 gap-2">
            <input
              ref={inputRef}
              type="text"
              value={message}
              onChange={(e) => setMessage(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && void send()}
              placeholder="Ask about this feed and activity log…"
              className="min-w-0 flex-1 rounded-xl border border-outline-variant/60 bg-surface px-4 py-3 text-body-sm text-on-surface placeholder:text-outline focus:border-primary focus:outline-none focus:ring-1 focus:ring-primary/30"
              disabled={loading}
            />
            <button
              type="button"
              onClick={() => void send()}
              disabled={loading || !message.trim()}
              className="flex h-12 w-12 shrink-0 items-center justify-center rounded-xl text-on-primary disabled:opacity-50"
              style={{ backgroundColor: accent }}
              aria-label="Send message"
            >
              <Icon name="send" size={20} />
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
