export type Severity = "critical" | "warning" | "info" | "normal";

export interface CameraFeed {
  id: string;
  name: string;
  location: string;
  zone?: string | null;
  sourceType?: string;
  image: string;
  streamUrl?: string;
  heatmapUrl?: string;
  previewUrl?: string;
  visionProbe?: Record<string, unknown> | null;
  error?: string | null;
  coords: string;
  status: "live" | "offline";
  enabled?: boolean;
  inferenceModel?: string | null;
  inferenceTask?: string | null;
  modelBadge?: { model: string; task: string } | null;
  anomalyScore?: number | null;
  backendCameraId?: string | null;
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

export interface LiveStats {
  camerasOnline: { current: number; total: number; healthy: boolean };
  inferencesPerMin: { value: number; trend: string };
  eventsToday: { value: number; delta: string };
  avgEdgeLatencyMs: { value: number; delta: string };
}

export interface CameraCreateInput {
  name: string;
  location: string;
  zone?: string;
  sourceType: "rtsp" | "onvif" | "webcam";
  streamUrl?: string;
  coords?: string;
  inferenceModel?: string;
  inferenceTask?: string;
  imageUrl?: string;
  backendCameraId?: string;
}

export interface RealtimeEvent {
  id: string;
  time: string;
  title: string;
  description: string;
  severity: "critical" | "primary" | "neutral";
  resolutionStatus?: "OPEN" | "ACKNOWLEDGED" | "RESOLVED" | "FALSE_POSITIVE";
}
