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

/** First card from backend webcam; remaining cards stay mock. */
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
    const webcam = apiCameras[0];
    if (!webcam) {
      return CAMERA_FEEDS;
    }
    if (webcam.streamUrl) {
      webcam.streamUrl = `/vision-api/api/cameras/${webcam.id}/stream`;
    }
    return [webcam, ...CAMERA_FEEDS.slice(1)];
  } catch {
    return CAMERA_FEEDS;
  }
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
