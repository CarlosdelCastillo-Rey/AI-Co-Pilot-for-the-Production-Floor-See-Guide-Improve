export type NavId = "live" | "identity" | "vision" | "analytics" | "timeline" | "alerts";

/** Routes hidden from sidebar and redirected when visited directly. */
export const HIDDEN_NAV_IDS: NavId[] = ["identity", "vision"];

/** Default landing route after login and `/`. */
export const DEFAULT_ROUTE = "/analytics";

export interface NavItem {
  id: NavId;
  label: string;
  href: string;
  icon: string;
}

const ALL_NAV_ITEMS: NavItem[] = [
  { id: "live", label: "Live Streams", href: "/live", icon: "videocam" },
  { id: "identity", label: "My Identity", href: "/identity", icon: "face" },
  { id: "vision", label: "Vision Lab", href: "/vision-lab", icon: "science" },
  { id: "analytics", label: "Analytics", href: "/analytics", icon: "analytics" },
  { id: "timeline", label: "Timeline", href: "/timeline", icon: "timeline" },
  { id: "alerts", label: "Alert Rules", href: "/alerts", icon: "rule" },
];

export const NAV_ITEMS: NavItem[] = ALL_NAV_ITEMS.filter(
  (item) => !HIDDEN_NAV_IDS.includes(item.id),
);

export function getNavItem(id: NavId): NavItem {
  return NAV_ITEMS.find((item) => item.id === id)!;
}
