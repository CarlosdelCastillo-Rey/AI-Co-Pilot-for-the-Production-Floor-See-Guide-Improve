"use client";

import { AuthProvider } from "@/components/auth/AuthProvider";
import { VisionOpsAdvisor } from "@/components/advisor/VisionOpsAdvisor";

export default function DashboardLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <AuthProvider>
      {children}
      <VisionOpsAdvisor />
    </AuthProvider>
  );
}
