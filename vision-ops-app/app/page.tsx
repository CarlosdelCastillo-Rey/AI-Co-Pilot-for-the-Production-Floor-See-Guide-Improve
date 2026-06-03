import { redirect } from "next/navigation";
import { DEFAULT_ROUTE } from "@/lib/navigation";

export default function Home() {
  redirect(DEFAULT_ROUTE);
}
