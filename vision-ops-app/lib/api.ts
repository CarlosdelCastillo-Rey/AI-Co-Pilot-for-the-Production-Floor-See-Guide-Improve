import { CAMERA_FEEDS, type CameraFeed } from "@/lib/mock-data";

const DEFAULT_API_URL = "http://localhost:8000";

export function getApiBase(): string {
  return process.env.NEXT_PUBLIC_API_URL ?? DEFAULT_API_URL;
}

/** Browser / client calls via Next.js rewrite */
export function getProxyApiBase(): string {
  return "/vision-api";
}

export function useMockData(): boolean {
  return process.env.NEXT_PUBLIC_USE_MOCK_DATA === "true";
}

export type FaceStatus = {
  enrolled: boolean;
  ready: boolean;
  name: string;
  error: string | null;
  has_preview: boolean;
  previewUrl: string | null;
};

export type FaceStorageInfo = {
  summary: string;
  paths: Record<string, string | boolean>;
  stored_items: {
    id: string;
    file: string | null;
    contents: string;
    used_for_recognition: boolean;
  }[];
  gitignored: boolean;
};

function proxyBackendUrl(url: string | undefined): string | undefined {
  if (!url) return url;
  if (url.startsWith("/vision-api") || url.startsWith("http://localhost:3000")) {
    return url;
  }
  const base = getApiBase().replace(/\/$/, "");
  if (url.startsWith(base)) {
    return `/vision-api${url.slice(base.length)}`;
  }
  if (url.startsWith("/api/")) {
    return `/vision-api${url}`;
  }
  return url;
}

/** Webcam + industrial cameras from backend when vision_enabled. */
export async function getLiveCameraFeeds(): Promise<CameraFeed[]> {
  if (useMockData()) {
    return CAMERA_FEEDS;
  }

  const base = getApiBase();
  try {
    const res = await fetch(`${base}/api/cameras`, {
      cache: "no-store",
    });
    if (!res.ok) {
      return CAMERA_FEEDS;
    }
    const apiCameras = (await res.json()) as CameraFeed[];
    if (!apiCameras.length) {
      return CAMERA_FEEDS;
    }
    return apiCameras.map((cam) => ({
      ...cam,
      streamUrl: proxyBackendUrl(cam.streamUrl),
      heatmapUrl: proxyBackendUrl(cam.heatmapUrl),
      previewUrl: proxyBackendUrl(cam.previewUrl),
      image: cam.image?.startsWith("/api/") ? proxyBackendUrl(cam.image) ?? cam.image : cam.image,
    }));
  } catch {
    return CAMERA_FEEDS;
  }
}

export type VisionStatus = {
  ready: boolean;
  cameras: Record<
    string,
    {
      label: string;
      last_probe?: boolean;
      has_heatmap?: boolean;
      heatmapUrl?: string | null;
      last_severity?: string;
      anomaly_score?: number;
    }
  >;
};

export async function fetchVisionStatus(): Promise<VisionStatus | null> {
  try {
    const res = await fetch(`${getProxyApiBase()}/api/vision/status`, { cache: "no-store" });
    if (!res.ok) return null;
    const data = (await res.json()) as VisionStatus;
    for (const key of Object.keys(data.cameras)) {
      const cam = data.cameras[key];
      if (cam.heatmapUrl) {
        cam.heatmapUrl = proxyBackendUrl(cam.heatmapUrl) ?? cam.heatmapUrl;
      }
    }
    return data;
  } catch {
    return null;
  }
}

export async function runVisionProbe(
  cameraId: string,
  options?: { setBaseline?: boolean; mode?: string },
): Promise<{ ok: boolean; message: string; data?: unknown }> {
  const res = await fetch(`${getProxyApiBase()}/api/vision/probe`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      camera_id: cameraId,
      mode: options?.mode ?? "auto",
      set_baseline: options?.setBaseline ?? false,
    }),
  });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) {
    const detail = typeof data.detail === "string" ? data.detail : "Vision probe failed";
    return { ok: false, message: detail };
  }
  return { ok: true, message: "Probe completed.", data };
}

export async function fetchFaceStatus(): Promise<FaceStatus | null> {
  try {
    const res = await fetch(`${getProxyApiBase()}/api/faces/status`, { cache: "no-store" });
    if (!res.ok) return null;
    const data = (await res.json()) as FaceStatus;
    if (data.previewUrl) {
      data.previewUrl = `${getProxyApiBase()}/api/faces/preview`;
    }
    return data;
  } catch {
    return null;
  }
}

export async function fetchFaceStorage(): Promise<FaceStorageInfo | null> {
  try {
    const res = await fetch(`${getProxyApiBase()}/api/faces/storage`, { cache: "no-store" });
    if (!res.ok) return null;
    return (await res.json()) as FaceStorageInfo;
  } catch {
    return null;
  }
}

export async function enrollFace(name: string): Promise<{ ok: boolean; message: string; data?: unknown }> {
  const res = await fetch(`${getProxyApiBase()}/api/faces/enroll`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ name: name.trim() }),
  });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) {
    const detail = typeof data.detail === "string" ? data.detail : "Enrollment failed";
    return { ok: false, message: detail };
  }
  return { ok: true, message: `Registered as “${(data as { name?: string }).name ?? name}”.`, data };
}

export async function deleteFaceEnrollment(): Promise<{ ok: boolean; message: string }> {
  const res = await fetch(`${getProxyApiBase()}/api/faces/enroll`, { method: "DELETE" });
  if (!res.ok) {
    const data = await res.json().catch(() => ({}));
    return { ok: false, message: (data as { detail?: string }).detail ?? "Delete failed" };
  }
  return { ok: true, message: "Enrollment removed." };
}
