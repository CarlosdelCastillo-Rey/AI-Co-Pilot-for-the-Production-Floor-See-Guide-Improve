import { Sidebar } from "./Sidebar";
import { TopNav } from "./TopNav";

interface AppShellProps {
  children: React.ReactNode;
  searchPlaceholder?: string;
  fullBleed?: boolean;
}

export function AppShell({
  children,
  searchPlaceholder,
  fullBleed = false,
}: AppShellProps) {
  return (
    <>
      <Sidebar />
      <TopNav searchPlaceholder={searchPlaceholder} />
      <main
        className={
          fullBleed
            ? "ml-[240px] min-h-screen pt-16"
            : "ml-[240px] min-h-screen bg-background pt-16"
        }
      >
        {children}
      </main>
    </>
  );
}
