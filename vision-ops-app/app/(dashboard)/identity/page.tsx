import { redirect } from "next/navigation";
import { DEFAULT_ROUTE } from "@/lib/navigation";

export default function IdentityPage() {
  redirect(DEFAULT_ROUTE);
}
