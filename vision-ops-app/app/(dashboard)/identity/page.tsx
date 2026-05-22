import { AppShell } from "@/components/layout/AppShell";
import { IdentityEnrollmentPanel } from "@/components/identity/IdentityEnrollmentPanel";

export default function IdentityPage() {
  return (
    <AppShell searchPlaceholder="Search…">
      <div className="p-lg">
        <div className="mb-lg">
          <h2 className="font-headline text-headline-md text-on-surface">
            Identity & face enrollment
          </h2>
          <p className="mt-1 text-body-sm text-outline">
            Register your name and face so Live Streams shows your name on the green box.
          </p>
        </div>
        <IdentityEnrollmentPanel />
      </div>
    </AppShell>
  );
}
