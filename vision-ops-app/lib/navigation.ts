export type NavId =
  | "live"
  | "harPersonHitl"
  | "analytics"
  | "timeline"
  | "alerts";

/** Default landing route after login and `/`. */
export const DEFAULT_ROUTE = "/analytics";

export interface NavItem {
  id: NavId;
  label: string;
  href: string;
  icon: string;
}

export const NAV_ITEMS: NavItem[] = [
  { id: "live", label: "Live Streams", href: "/live", icon: "videocam" },
  { id: "harPersonHitl", label: "Person HITL", href: "/har-hitl", icon: "groups" },
  { id: "analytics", label: "Dashboard", href: "/analytics", icon: "analytics" },
  { id: "timeline", label: "Timeline", href: "/timeline", icon: "timeline" },
  { id: "alerts", label: "Alerts & Notifications", href: "/alerts", icon: "rule" },
];

export function getNavItem(id: NavId): NavItem {
  return NAV_ITEMS.find((item) => item.id === id)!;
}
