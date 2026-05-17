export type NavId = "live" | "analytics" | "timeline" | "alerts";

export interface NavItem {
  id: NavId;
  label: string;
  href: string;
  icon: string;
}

export const NAV_ITEMS: NavItem[] = [
  { id: "live", label: "Live Streams", href: "/live", icon: "videocam" },
  { id: "analytics", label: "Analytics", href: "/analytics", icon: "analytics" },
  { id: "timeline", label: "Timeline", href: "/timeline", icon: "timeline" },
  { id: "alerts", label: "Alert Rules", href: "/alerts", icon: "rule" },
];

export function getNavItem(id: NavId): NavItem {
  return NAV_ITEMS.find((item) => item.id === id)!;
}
