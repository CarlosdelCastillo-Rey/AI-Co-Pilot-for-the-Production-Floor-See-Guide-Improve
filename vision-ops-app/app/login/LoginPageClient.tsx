"use client";

import { FormEvent, useEffect, useState } from "react";
import { useRouter, useSearchParams } from "next/navigation";
import { Icon } from "@/components/ui/Icon";
import { clearToken, getToken, setToken } from "@/lib/auth";
import { fetchCurrentUser, loginUser, registerUser } from "@/lib/api";
import { cn } from "@/lib/cn";
import { DEFAULT_ROUTE } from "@/lib/navigation";

type Mode = "login" | "register";

const inputClass =
  "box-border block w-full min-w-0 rounded-[10px] border border-outline-variant bg-white px-3 py-2.5 text-body-md text-on-surface placeholder:text-outline/60 focus:border-primary focus:outline-none focus:ring-2 focus:ring-primary/20";

export default function LoginPageClient() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const next = searchParams.get("next") || DEFAULT_ROUTE;

  const [mode, setMode] = useState<Mode>("login");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [name, setName] = useState("");
  const [role, setRole] = useState("Supervisor");
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    const token = getToken();
    if (!token) return;
    void fetchCurrentUser().then((u) => {
      if (u) router.replace(next);
      else clearToken();
    });
  }, [next, router]);

  const handleSubmit = async (e: FormEvent) => {
    e.preventDefault();
    setError(null);
    setBusy(true);
    try {
      const result =
        mode === "login"
          ? await loginUser(email, password)
          : await registerUser(email, password, name, role);
      if (!result) {
        setError(
          mode === "login"
            ? "Invalid email or password"
            : "Could not create account. Check your details or try another email.",
        );
        return;
      }
      setToken(result.token);
      router.replace(next);
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="relative flex min-h-screen items-center justify-center bg-background px-4 py-10 blueprint-grid">
      <div className="pointer-events-none absolute inset-0 bg-gradient-to-b from-primary-fixed/20 via-transparent to-transparent" />

      <div className="relative w-full max-w-[420px] shrink-0">
        <div className="mb-6 text-center">
          <div className="mx-auto mb-4 flex h-12 w-12 items-center justify-center rounded-xl bg-primary text-on-primary shadow-sm">
            <Icon name="visibility" size={26} />
          </div>
          <h1 className="font-headline text-headline-md text-on-background">VisionOps</h1>
          <p className="mt-1 text-body-sm text-outline">Sign in to the production floor</p>
        </div>

        <div className="rounded-card border border-outline-variant/80 bg-surface-container-lowest p-8 shadow-overlay">
          <div className="mb-6 grid grid-cols-2 gap-1 rounded-lg bg-surface-container-low p-1">
            {(["login", "register"] as const).map((tab) => (
              <button
                key={tab}
                type="button"
                onClick={() => setMode(tab)}
                className={cn(
                  "rounded-md py-2.5 text-body-sm font-medium transition-colors",
                  mode === tab
                    ? "bg-white text-primary shadow-sm"
                    : "text-outline hover:text-on-surface",
                )}
              >
                {tab === "login" ? "Sign in" : "Create account"}
              </button>
            ))}
          </div>

          <form onSubmit={handleSubmit} className="space-y-4">
            {mode === "register" && (
              <>
                <label className="block">
                  <span className="mb-1.5 block text-label-sm text-on-surface-variant">
                    Full name
                  </span>
                  <input
                    type="text"
                    required
                    value={name}
                    onChange={(e) => setName(e.target.value)}
                    autoComplete="name"
                    className={inputClass}
                    placeholder="Jane Supervisor"
                  />
                </label>
                <label className="block">
                  <span className="mb-1.5 block text-label-sm text-on-surface-variant">
                    Role / title
                  </span>
                  <input
                    type="text"
                    required
                    value={role}
                    onChange={(e) => setRole(e.target.value)}
                    autoComplete="organization-title"
                    className={inputClass}
                    placeholder="Ops Lead, Supervisor, Engineer…"
                  />
                </label>
              </>
            )}

            <label className="block">
              <span className="mb-1.5 block text-label-sm text-on-surface-variant">Email</span>
              <input
                type="email"
                required
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                autoComplete="email"
                className={inputClass}
                placeholder="you@plant.com"
              />
            </label>

            <label className="block">
              <span className="mb-1.5 block text-label-sm text-on-surface-variant">
                Password
              </span>
              <input
                type="password"
                required
                minLength={6}
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                autoComplete={mode === "login" ? "current-password" : "new-password"}
                className={inputClass}
                placeholder="At least 6 characters"
              />
            </label>

            {error && (
              <p className="rounded-lg border border-error-container bg-error-container/50 px-3 py-2 text-body-sm text-error">
                {error}
              </p>
            )}

            <button
              type="submit"
              disabled={busy}
              className="mt-2 flex w-full min-h-touch items-center justify-center rounded-[10px] bg-primary text-label-md font-semibold text-on-primary transition-colors hover:bg-primary-container disabled:opacity-50"
            >
              {busy ? "Please wait…" : mode === "login" ? "Sign in" : "Create account"}
            </button>
          </form>

          {mode === "login" && (
            <p className="mt-6 border-t border-outline-variant/60 pt-5 text-center text-body-sm text-outline">
              Demo account{" "}
              <span className="font-label text-label-sm text-on-surface">
                admin@visionops.local
              </span>{" "}
              /{" "}
              <span className="font-label text-label-sm text-on-surface">admin123</span>
            </p>
          )}
        </div>
      </div>
    </div>
  );
}
