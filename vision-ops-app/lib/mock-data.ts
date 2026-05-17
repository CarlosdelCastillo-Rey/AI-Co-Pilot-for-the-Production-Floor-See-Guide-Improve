export type Severity = "critical" | "warning" | "info" | "normal";

export interface CameraFeed {
  id: string;
  name: string;
  location: string;
  image: string;
  coords: string;
  status: "live" | "offline";
  overlays: {
    type: "person" | "machine" | "forklift";
    label: string;
    top: string;
    left: string;
    width: string;
    height: string;
    variant?: "primary" | "tertiary" | "error";
  }[];
}

export interface TimelineEvent {
  id: string;
  time: string;
  severity: Severity;
  title: string;
  description: string;
  meta: { icon: string; text: string }[];
  thumbnail: string;
  clipDuration: string;
}

export interface AlertRule {
  id: string;
  icon: string;
  title: string;
  description: string;
  zone: string;
  severity: "CRITICAL" | "WARNING" | "DISABLED";
  enabled: boolean;
}

export interface RealtimeEvent {
  time: string;
  title: string;
  description: string;
  severity: "critical" | "primary" | "neutral";
}

export const CAMERA_FEEDS: CameraFeed[] = [
  {
    id: "cam-01",
    name: "Camera 01 - Assembly",
    location: "Main Hall / Line 4",
    image:
      "https://lh3.googleusercontent.com/aida-public/AB6AXuCtdXg1qgVaATzDFV4GlsmN6CkUoyf1Z5phhagAyhKszH_SM-XO_97YtvK6_rhFO1EC5ny-HEVEIP1Wz2oRu_LYR5IOJVdWCFu0csqXHHopNJFR5fD0-ooCwFJKB6q8aDm0yPLzbPKtGYY7AQThGRta6LJSy3krV7Ze8hd6UnLyT7J6eiI11S6664PLbZ9IWYxq4SeOpkEwSm2g-eCVrZOwtq7YjtLw8HV8C_23jAB7xWqoV3X1prHnLVBcL0GHSMS0ayxsVG6QrqY",
    coords: "42.3601° N, 71.0589° W | 12:44:03:22",
    status: "live",
    overlays: [
      {
        type: "person",
        label: "Operator [ID_04] - 98%",
        top: "30%",
        left: "45%",
        width: "12%",
        height: "40%",
        variant: "primary",
      },
      {
        type: "machine",
        label: "Assembly_Station_A - ACTIVE",
        top: "20%",
        left: "10%",
        width: "30%",
        height: "60%",
        variant: "tertiary",
      },
    ],
  },
  {
    id: "cam-02",
    name: "Camera 02 - Warehouse",
    location: "Loading Dock / Zone B",
    image:
      "https://lh3.googleusercontent.com/aida-public/AB6AXuD0WaxmzB30i4UvnXB1kC5UAjuno45jZ0-lYANMKEPRwQpqZH639_Ac7yPq9EJwxynwUcUc8jWfLtP6TuMgSHCc4R8QV2j8GXrckY0OSBfzsbliQXwp7qGaM_dgRn_CJ_-YN2M84FIR_4mTkNuSeOUqYZv-zFb-PGGtCtruaN4-mtsBu_sa6AvIDb6JnHCMjexxBix8FdInNk_8IbvGQsjiq1a0uDuXJABCY-cv8XDEYCM9YPSBwMnKCs_vAm8Ksn_xYUDxWv9393I",
    coords: "42.3610° N, 71.0595° W | 12:44:03:22",
    status: "live",
    overlays: [
      {
        type: "forklift",
        label: "Forklift [FL-02] - SPEED ALERT",
        top: "60%",
        left: "20%",
        width: "25%",
        height: "30%",
        variant: "error",
      },
    ],
  },
];

export const REALTIME_EVENTS: RealtimeEvent[] = [
  {
    time: "12:43:55",
    title: "Proximity Warning",
    description: "Forklift [FL-02] entered restricted Zone B",
    severity: "critical",
  },
  {
    time: "12:42:10",
    title: "Shift Change Detected",
    description: "Operator [ID_04] assigned to Station A",
    severity: "primary",
  },
  {
    time: "12:40:02",
    title: "System Diagnostic",
    description: "IP Node [192.168.1.14] latency stable",
    severity: "neutral",
  },
];

export const TIMELINE_EVENTS: TimelineEvent[] = [
  {
    id: "evt-1",
    time: "14:23:12",
    severity: "critical",
    title: "Conveyor 2 Blockage Detected",
    description:
      "Flow stopped for 3:12 due to material buildup at Transfer Point Alpha. Automated e-stop triggered. Operator recalibration required before restart.",
    meta: [
      { icon: "person", text: "J. Miller" },
      { icon: "location_on", text: "Zone 4 / Floor 2" },
    ],
    thumbnail:
      "https://lh3.googleusercontent.com/aida-public/AB6AXuBp3bQPzQYw62zX_2pzW3Le1-ZFn9mAobM6DAzL7jaK90xOBNPRzHhfUesgAadlpjhz9ex-CPUQ5sujRcXQAIgBKm_a9sfLVCFlKWtFal_Ps3Ql17YDYsuq1rxGtAqG9lxV813sMd2V513vwwM-x8gSQiQs4E05Vl_Jue-RdN-Bgwigs3Gzmxx05yuZnqXQfPmQv-p3s0a5zanhbJrj6VS7CXsn3uaciPGI4A99y25oo14uwZeMFrSi3ZHR_s-8iJfIBR5cvjRpftM",
    clipDuration: "00:14",
  },
  {
    id: "evt-2",
    time: "13:45:05",
    severity: "warning",
    title: "Thermal Deviation: Motor M-09",
    description:
      "Operating temperature exceeded threshold of 85°C. Current reading: 92.4°C. Cooling sequence initiated manually. Scheduling maintenance for next shift.",
    meta: [{ icon: "sensors", text: "Heat Sensor HS-41" }],
    thumbnail:
      "https://lh3.googleusercontent.com/aida-public/AB6AXuCnsrmrUduGm-QCjh10fyqvme8ZMCbKRjwABPesR4oxCruGBfqNCi8SUe8X2TVRiF052P7j7ocQUD_ulTVZFPgvNErB3NZhu8bMos0pAljzxOEeKp0mF7_2YiTO1fNesT5MfbEyN4omirInz5n7s-GZfd9_YxgIfi0LR6OLVg4_zDhWPyY5uBxxKy3RSjg3GN9oNUgXPmGUoxS7APDt78Q7icUBckGq9rCyR07l19id848lmVgAbTquisW7ZDDz7cEbsSUdwgSkPGA",
    clipDuration: "00:08",
  },
  {
    id: "evt-3",
    time: "12:10:44",
    severity: "info",
    title: "Operator Handoff: Shift B",
    description:
      "Standard procedure completed. All systems verified at 98.4% efficiency. Inventory levels marked as sufficient for the next 8-hour cycle.",
    meta: [{ icon: "sync_alt", text: "Auto-logged" }],
    thumbnail:
      "https://lh3.googleusercontent.com/aida-public/AB6AXuA8QSJuWijkQiH0Uh59JTlgi2pmkroNvjYpqGsPSWBQhVy75kVNp29LsbVxQkkcOhr0_IjPjn1wvj7BhNLU-wKALbVVHiFd1KUJWO73VC_a8ygEtXF1PNwZVDH7Tk4y3nkmIryRMuV-lozrr-d-Ruv5Zfzf7vOtXbw8iHfBZWSB-VALVnyaBxmbV3boSV7Ht9yFVnJP13RoXH4IchlJ6WexsoAo_gS76TxFoCPJPGZ-JBuSl57Kl3PeIyaFzhF52BANco8nFQ166do",
    clipDuration: "00:32",
  },
  {
    id: "evt-4",
    time: "09:12:00",
    severity: "critical",
    title: "Unidentified Personnel in Restricted Zone",
    description:
      "Vision AI detected unauthorized human presence in Robot Cell G-2. Safety lockdown engaged. Supervisor notification sent via secure link.",
    meta: [{ icon: "visibility", text: "Vision Node 12" }],
    thumbnail:
      "https://lh3.googleusercontent.com/aida-public/AB6AXuDzVywH68hQHg65TTMaExnZY0KmdNovr8Et6ZtjbO4O0uCheM1Y1cK3yDjdY1LdmwOYCpZ8d3UGHokqIfh2EsPI7Y939h06x8lJqzumKo_-_wusiCZL6Ze3hbOXpn0ie1l5LH6LF52MyCcCUV3t_8ajOi4a-3ZSU1jKSbO1oJ51Ky-7aXwNZal7Zcezfqn00xCQNSu756VMRgpyfjIM2GAQqPF7i1Hc9cb6LrNVpFlxAyDITx3nLSG3zYqeyEjYl9gMB8wygpgGckw",
    clipDuration: "00:45",
  },
];

export const ALERT_RULES: AlertRule[] = [
  {
    id: "rule-1",
    icon: "polyline",
    title: "Geofencing Intrusion (North Dock)",
    description:
      "Triggers if unauthorized personnel enter Zone A-14 after 22:00.",
    zone: "ZONE A-14",
    severity: "CRITICAL",
    enabled: true,
  },
  {
    id: "rule-2",
    icon: "groups",
    title: "Crowd Density Threshold",
    description: "Main Lobby monitoring. Trigger when person count exceeds 15.",
    zone: "MAIN LOBBY",
    severity: "WARNING",
    enabled: true,
  },
  {
    id: "rule-3",
    icon: "warning",
    title: "PPE Compliance Check",
    description:
      "Automated hard-hat and high-vis vest detection at Gate 3.",
    zone: "GATE 3",
    severity: "DISABLED",
    enabled: false,
  },
  {
    id: "rule-4",
    icon: "fire_extinguisher",
    title: "Fire/Smoke Detection",
    description:
      "Thermal imaging integration for early fire detection in Warehouse B.",
    zone: "WHSE B",
    severity: "CRITICAL",
    enabled: true,
  },
];

export const SHIFT_SUMMARY = {
  date: "Tuesday, Oct 24, 2023",
  incidentCount: 2,
  incidentDelta: "+100% from prev.",
  uptime: "94.2%",
  assets: [
    { name: "Conveyor Line Alpha", events: 2 },
    { name: "Motor M-09", events: 1 },
    { name: "Robot Cell G-2", events: 1 },
  ],
};
