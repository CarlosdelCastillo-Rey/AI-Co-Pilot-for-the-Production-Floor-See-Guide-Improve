"use client";

import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useState,
  type ReactNode,
} from "react";
import { usePathname, useRouter } from "next/navigation";
import { clearToken, getToken, setToken } from "@/lib/auth";
import { fetchCurrentUser, loginUser, registerUser, type UserApi } from "@/lib/api";

type AuthContextValue = {
  user: UserApi | null;
  loading: boolean;
  login: (email: string, password: string) => Promise<string | null>;
  register: (email: string, password: string, name: string) => Promise<string | null>;
  logout: () => void;
};

const AuthContext = createContext<AuthContextValue | null>(null);

export function AuthProvider({ children }: { children: ReactNode }) {
  const router = useRouter();
  const pathname = usePathname();
  const [user, setUser] = useState<UserApi | null>(null);
  const [loading, setLoading] = useState(true);

  const loadUser = useCallback(async () => {
    const token = getToken();
    if (!token) {
      setUser(null);
      setLoading(false);
      router.replace(`/login?next=${encodeURIComponent(pathname)}`);
      return;
    }
    const me = await fetchCurrentUser();
    if (!me) {
      clearToken();
      setUser(null);
      setLoading(false);
      router.replace(`/login?next=${encodeURIComponent(pathname)}`);
      return;
    }
    setUser(me);
    setLoading(false);
  }, [pathname, router]);

  useEffect(() => {
    void loadUser();
  }, [loadUser]);

  const login = useCallback(async (email: string, password: string) => {
    const result = await loginUser(email, password);
    if (!result) return "Invalid email or password";
    setToken(result.token);
    setUser(result.user);
    return null;
  }, []);

  const register = useCallback(async (email: string, password: string, name: string) => {
    const result = await registerUser(email, password, name);
    if (!result) return "Could not create account (email may already exist)";
    setToken(result.token);
    setUser(result.user);
    return null;
  }, []);

  const logout = useCallback(() => {
    clearToken();
    setUser(null);
    router.replace("/login");
  }, [router]);

  const value = useMemo(
    () => ({ user, loading, login, register, logout }),
    [user, loading, login, register, logout],
  );

  if (loading) {
    return (
      <div className="flex min-h-screen items-center justify-center bg-[#F5F6F8]">
        <p className="text-body-sm text-[#5A626C]">Loading…</p>
      </div>
    );
  }

  if (!user) return null;

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

export function useAuth(): AuthContextValue {
  const ctx = useContext(AuthContext);
  if (!ctx) throw new Error("useAuth must be used within AuthProvider");
  return ctx;
}
