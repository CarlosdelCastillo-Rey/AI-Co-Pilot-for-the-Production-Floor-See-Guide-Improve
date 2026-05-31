import { Suspense } from "react";
import LoginPage from "./LoginPageClient";

export default function Page() {
  return (
    <Suspense
      fallback={
        <div className="flex min-h-screen items-center justify-center bg-[#F5F6F8]">
          <p className="text-body-sm text-[#5A626C]">Loading…</p>
        </div>
      }
    >
      <LoginPage />
    </Suspense>
  );
}
