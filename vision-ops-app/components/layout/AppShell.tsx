"use client";

import { Sidebar } from "./Sidebar";
import { TopNav } from "./TopNav";

interface AppShellProps {
  children: React.ReactNode;
  searchPlaceholder?: string;
  searchValue?: string;
  onSearchChange?: (value: string) => void;
  fullBleed?: boolean;
}

export function AppShell({
  children,
  searchPlaceholder,
  searchValue,
  onSearchChange,
  fullBleed = false,
}: AppShellProps) {
  return (
    <>
      <Sidebar />
      <TopNav
        searchPlaceholder={searchPlaceholder}
        searchValue={searchValue}
        onSearchChange={onSearchChange}
      />
      <main
        className={
          fullBleed
            ? "ml-[240px] min-h-screen bg-background pt-16"
            : "ml-[240px] min-h-screen bg-background pt-16"
        }
      >
        {children}
      </main>
    </>
  );
}
