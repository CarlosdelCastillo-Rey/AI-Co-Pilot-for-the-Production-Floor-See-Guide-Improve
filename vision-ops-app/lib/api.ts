import type { CameraCreateInput, CameraFeed, LiveStats, RealtimeEvent } from "@/lib/types";
import { authHeaders } from "@/lib/auth";

const DEFAULT_API_URL = "http://localhost:8000";

function apiAuthHeaders(): HeadersInit {
  return { "Content-Type": "application/json", ...authHeaders() };
}

export type UserApi = {
  id: string;
  email: string;
  name: string;
  role: string;
  createdAt?: string | null;
};

export type AuthResult = { token: string; user: UserApi };

export async function loginUser(email: string, password: string): Promise<AuthResult | null> {
  const res = await fetch(`${getApiFetchBase()}/api/auth/login`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ email, password }),
  });
  if (!res.ok) return null;
  return res.json();
}

export async function registerUser(
  email: string,
  password: string,
  name: string,
  role = "Supervisor",
): Promise<AuthResult | null> {
  const res = await fetch(`${getApiFetchBase()}/api/auth/register`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ email, password, name, role }),
  });
  if (!res.ok) return null;
  return res.json();
}

export async function fetchCurrentUser(): Promise<UserApi | null> {
  const headers = authHeaders();
  if (!headers.Authorization) return null;
  const res = await fetch(`${getApiFetchBase()}/api/auth/me`, {
    cache: "no-store",
    headers,
  });
  if (!res.ok) return null;
  return res.json();
}

export type DataResetResult = {
  ok: boolean;
  clearedByTable: Record<string, number>;
  totalRowsCleared: number;
  resetBy: string;
  preserved: string[];
};

export async function resetDynamicData(): Promise<DataResetResult | null> {
  const res = await fetch(`${getApiFetchBase()}/api/admin/reset-dynamic-data`, {
    method: "POST",
    headers: apiAuthHeaders(),
  });
  if (!res.ok) return null;
  return res.json();
}

export function getApiBase(): string {
  return process.env.NEXT_PUBLIC_API_URL ?? DEFAULT_API_URL;
}

/** SSR: direct backend URL. Browser: empty — paths are already `/api/...` (Next rewrite). */
export function getApiFetchBase(): string {
  if (typeof window === "undefined") {
    return getApiBase().replace(/\/$/, "");
  }
  return "";
}

const API_FETCH_RETRIES = 3;

export async function fetchApi(path: string, init?: RequestInit): Promise<Response | null> {
  const base = getApiFetchBase().replace(/\/$/, "");
  const url = path.startsWith("http") ? path : `${base}${path.startsWith("/") ? path : `/${path}`}`;
  for (let attempt = 0; attempt < API_FETCH_RETRIES; attempt++) {
    try {
      return await fetch(url, init);
    } catch {
      if (attempt >= API_FETCH_RETRIES - 1) return null;
      await new Promise((r) => setTimeout(r, 400 * (attempt + 1)));
    }
  }
  return null;
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

/** Route media URLs through the unified /api proxy when needed. */
export function proxyMediaUrl(url: string | null | undefined): string | undefined {
  if (!url?.trim()) return undefined;
  const trimmed = url.trim();
  if (trimmed.startsWith("/api/")) {
    return trimmed;
  }
  if (trimmed.startsWith("http://localhost:3000/")) {
    return trimmed.replace("http://localhost:3000", "");
  }

  let path = trimmed;
  if (trimmed.startsWith("http://") || trimmed.startsWith("https://")) {
    try {
      path = new URL(trimmed).pathname;
    } catch {
      return trimmed;
    }
  }

  if (path.startsWith("/api/")) {
    return path;
  }

  const apiBase = getApiBase().replace(/\/$/, "");
  if (trimmed.startsWith(`${apiBase}/`) || trimmed === apiBase) {
    return trimmed.slice(apiBase.length);
  }

  return trimmed;
}

/** HAR log/card thumbnails — JPEG snapshots via backend proxy, never mock-video MP4s. */
export function resolveHarPreviewUrl(url: string | null | undefined): string | undefined {
  if (!url?.trim()) return undefined;
  const lower = url.toLowerCase().split("?")[0];
  if (/\.(mp4|webm|mov|mkv|avi|m4v)$/.test(lower)) return undefined;
  return proxyMediaUrl(url);
}

/** Cameras from alerting DB merged with vision-backend runtime (streams, probes). */
export async function getLiveCameraFeeds(options?: {
  status?: string;
  model?: string;
  zone?: string;
  q?: string;
}): Promise<CameraFeed[]> {
  const params = new URLSearchParams();
  if (options?.status) params.set("status", options.status);
  if (options?.model) params.set("inferenceModel", options.model);
  if (options?.zone) params.set("zone", options.zone);
  if (options?.q) params.set("q", options.q);
  const qs = params.toString();
  const path = `/api/cameras${qs ? `?${qs}` : ""}`;
  const res = await fetchApi(path, { cache: "no-store" });
  if (!res?.ok) {
    return [];
  }
  const apiCameras = (await res.json()) as CameraFeed[];
  return apiCameras.map((cam) => ({
    ...cam,
    streamUrl: proxyMediaUrl(cam.streamUrl),
    videoUrl: cam.videoUrl,
    heatmapUrl: proxyMediaUrl(cam.heatmapUrl),
    previewUrl: proxyMediaUrl(cam.previewUrl),
    image: cam.image?.startsWith("/api/") ? proxyMediaUrl(cam.image) ?? cam.image : cam.image,
  }));
}

export async function fetchLiveStats(): Promise<LiveStats | null> {
  const res = await fetchApi("/api/cameras/stats/live", { cache: "no-store" });
  if (!res?.ok) return null;
  return res.json();
}

export async function createCamera(input: CameraCreateInput): Promise<CameraFeed | null> {
  const res = await fetch(`${getApiFetchBase()}/api/cameras`, {
    method: "POST",
    headers: apiAuthHeaders(),
    body: JSON.stringify(input),
  });
  if (!res.ok) return null;
  const cam = (await res.json()) as CameraFeed;
  return {
    ...cam,
    streamUrl: proxyMediaUrl(cam.streamUrl),
    videoUrl: cam.videoUrl,
    heatmapUrl: proxyMediaUrl(cam.heatmapUrl),
    previewUrl: proxyMediaUrl(cam.previewUrl),
  };
}

export async function deleteCamera(cameraId: string): Promise<boolean> {
  const res = await fetch(`${getApiFetchBase()}/api/cameras/${cameraId}`, {
    method: "DELETE",
  });
  return res.ok;
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
    const res = await fetch(`${getApiFetchBase()}/api/vision/status`, { cache: "no-store" });
    if (!res.ok) return null;
    const data = (await res.json()) as VisionStatus;
    for (const key of Object.keys(data.cameras)) {
      const cam = data.cameras[key];
      if (cam.heatmapUrl) {
        cam.heatmapUrl = proxyMediaUrl(cam.heatmapUrl) ?? cam.heatmapUrl;
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
  const res = await fetch(`${getApiFetchBase()}/api/vision/probe`, {
    method: "POST",
    headers: apiAuthHeaders(),
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

// --- HAR activity recognition (har-research) ---

export type HarModelId =
  | "v2-vjepa-powermean7"
  | "v2-vjepa2-ssv2"
  | "v2-vjepa"
  | "v2-dinov2";

export type HarModelInfo = {
  model_id: string;
  camera_id: string;
  label: string;
  ready: boolean;
  reason: string;
};

export type HarStatus = {
  ready: boolean;
  shared_clip: {
    resolved_path: string | null;
    source: string;
    frame_count_sampled: number;
    video_frame_count: number | null;
  };
  models: Record<
    string,
    {
      model_id: string;
      camera_id: string;
      label: string;
      last_probe?: boolean;
      prediction?: { label: string; confidence: number; top_k?: { label: string; prob: number }[] };
      source?: string;
      backend?: string;
    }
  >;
};

export async function fetchHarModels(): Promise<{
  models: HarModelInfo[];
  torch_available: boolean;
  checkpoint_dir_exists: boolean;
} | null> {
  try {
    const res = await fetch(`${getApiFetchBase()}/api/vision/har/models`, { cache: "no-store" });
    if (!res.ok) return null;
    const data = await res.json();
    return {
      models: data.models ?? [],
      torch_available: Boolean(data.torch_available),
      checkpoint_dir_exists: Boolean(data.checkpoint_dir_exists),
    };
  } catch {
    return null;
  }
}

export async function fetchHarStatus(): Promise<HarStatus | null> {
  try {
    const res = await fetch(`${getApiFetchBase()}/api/vision/har/status`, { cache: "no-store" });
    if (!res.ok) return null;
    return (await res.json()) as HarStatus;
  } catch {
    return null;
  }
}

export type HarTrackPrediction = {
  track_id: number;
  bbox: [number, number, number, number];
  det_conf: number;
  action_label: string | null;
  action_confidence: number | null;
  inferring?: boolean;
  n_frames?: number;
  global_person_id?: string | null;
  display_name?: string | null;
};

export type HarLiveCameraState = {
  camera_id: string;
  model_id: string;
  session_id?: string;
  video?: string;
  videoUrl?: string;
  inferring?: boolean;
  prediction?: {
    label: string;
    confidence: number;
    top_k?: { label: string; prob: number }[];
    track_id?: number;
  };
  track_predictions?: HarTrackPrediction[];
  person_count?: number;
  per_person_mode?: boolean;
  error?: string | null;
};

export async function fetchHarLiveCamera(cameraId: string): Promise<HarLiveCameraState | null> {
  try {
    const res = await fetch(`${getApiFetchBase()}/api/vision/har/live/${cameraId}`, {
      cache: "no-store",
    });
    if (!res.ok) return null;
    return (await res.json()) as HarLiveCameraState;
  } catch {
    return null;
  }
}

export async function setHarLivePlayback(cameraId: string, playing: boolean): Promise<void> {
  try {
    await fetch(`${getApiFetchBase()}/api/vision/har/live/${cameraId}/playback`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ playing }),
    });
  } catch {
    /* best-effort sync */
  }
}

export const HAR_BENCH_CAMERA_ID = "cam-har-bench";

export function mockSlotCameraId(slot: number): string {
  return `cam-har-mock-${slot}`;
}

export function mockSlotStreamUrl(slot: number): string {
  return `/api/cameras/${mockSlotCameraId(slot)}/stream`;
}

export type HarWallLayout = "full" | "dual" | "quad";

export async function syncHarMockWall(body: {
  layout: HarWallLayout;
  playing: boolean;
  model_id: HarModelId;
  active_video: string;
  full_view_index: number;
}): Promise<boolean> {
  try {
    const res = await fetch(`${getApiFetchBase()}/api/vision/har/wall/sync`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    return res.ok;
  } catch {
    return false;
  }
}

export type HarBenchConfig = {
  infer_every: number;
  buffer_frames: number;
  stream_fps: number;
  show_heatmap: boolean;
  show_yolo_boxes: boolean;
  top_k: number;
  ingest_logs: boolean;
  per_person_mode: boolean;
  dwell_windows: number;
  bbox_padding: number;
};

export type HarBenchSnapshot = {
  enabled: boolean;
  camera_id: string;
  stream_url: string;
  state: HarLiveCameraState & {
    config?: HarBenchConfig;
    backend?: string | null;
    device?: string | null;
    last_infer_ms?: number | null;
  };
  videos: { name: string; url: string }[];
  models: { model_id: HarModelId; label: string; ready: boolean }[];
  slots?: Record<string, HarLiveCameraState>;
  layout?: HarWallLayout;
  full_view_index?: number;
  primary_slot?: number;
};

export async function fetchHarBench(): Promise<HarBenchSnapshot | null> {
  try {
    const res = await fetch(`${getApiFetchBase()}/api/vision/har/bench`, { cache: "no-store" });
    if (!res.ok) return null;
    return (await res.json()) as HarBenchSnapshot;
  } catch {
    return null;
  }
}

export async function patchHarBenchConfig(
  config: Partial<HarBenchConfig>,
): Promise<HarBenchConfig | null> {
  try {
    const res = await fetch(`${getApiFetchBase()}/api/vision/har/bench/config`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(config),
    });
    if (!res.ok) return null;
    const data = await res.json();
    return data.config as HarBenchConfig;
  } catch {
    return null;
  }
}

export async function setHarBenchModel(modelId: HarModelId): Promise<boolean> {
  try {
    const res = await fetch(`${getApiFetchBase()}/api/vision/har/bench/model`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ model_id: modelId }),
    });
    return res.ok;
  } catch {
    return false;
  }
}

export async function setHarBenchVideo(videoName: string): Promise<boolean> {
  try {
    const res = await fetch(`${getApiFetchBase()}/api/vision/har/bench/video`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ video: videoName }),
    });
    return res.ok;
  } catch {
    return false;
  }
}

export async function resetHarBenchSession(): Promise<boolean> {
  try {
    const res = await fetch(`${getApiFetchBase()}/api/vision/har/bench/reset`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
    });
    return res.ok;
  } catch {
    return false;
  }
}

export function harBenchStreamUrl(slot = 0): string {
  return mockSlotStreamUrl(slot);
}

export async function setHarLivePlaybackAll(playing: boolean): Promise<boolean> {
  try {
    const res = await fetch(`${getApiFetchBase()}/api/vision/har/live/playback-all`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ playing }),
    });
    return res.ok;
  } catch {
    return false;
  }
}

export async function runHarProbe(
  modelId: HarModelId,
  options?: { clipPath?: string },
): Promise<{ ok: boolean; message: string; data?: unknown }> {
  const res = await fetch(`${getApiFetchBase()}/api/vision/har/${modelId}/probe`, {
    method: "POST",
    headers: apiAuthHeaders(),
    body: JSON.stringify({ clip_path: options?.clipPath ?? null }),
  });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) {
    const detail = typeof data.detail === "string" ? data.detail : "HAR probe failed";
    return { ok: false, message: detail };
  }
  return { ok: true, message: "HAR probe completed.", data };
}

type HarProbeAllJob = {
  id: string;
  status: string;
  error?: string | null;
  result?: {
    status?: string;
    errors?: { model_id: string; error: string }[];
    probes?: unknown[];
  };
};

async function pollHarProbeAllJob(
  jobId: string,
  onProgress?: (message: string) => void,
): Promise<HarProbeAllJob | null> {
  const deadline = Date.now() + 10 * 60 * 1000;
  while (Date.now() < deadline) {
    const res = await fetch(`${getApiFetchBase()}/api/vision/har/probe-all/jobs/${jobId}`, {
      cache: "no-store",
    });
    if (!res.ok) return null;
    const job = (await res.json()) as HarProbeAllJob;
    if (job.status === "running") {
      onProgress?.("Running HAR models on mock videos…");
      await new Promise((r) => setTimeout(r, 2000));
      continue;
    }
    return job;
  }
  return null;
}

export async function runAllHarProbes(
  options?: {
    clipPath?: string;
    reshuffleVideos?: boolean;
    onProgress?: (message: string) => void;
  },
): Promise<{ ok: boolean; message: string; data?: unknown }> {
  const base = getApiFetchBase();
  const startRes = await fetch(`${base}/api/vision/har/probe-all/async`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      clip_path: options?.clipPath ?? null,
      reshuffle_videos: options?.reshuffleVideos ?? true,
    }),
  });
  const startData = await startRes.json().catch(() => ({}));
  if (!startRes.ok) {
    const detail =
      typeof startData.detail === "string" ? startData.detail : "HAR probe-all failed to start";
    return { ok: false, message: detail };
  }

  const jobId = startData.job_id as string | undefined;
  if (!jobId) {
    return { ok: false, message: "No job id returned from probe-all/async" };
  }

  options?.onProgress?.("Started batch probe…");
  const job = await pollHarProbeAllJob(jobId, options?.onProgress);
  if (!job) {
    return { ok: false, message: "Probe-all timed out or job not found." };
  }
  if (job.status === "error" || job.error) {
    return { ok: false, message: job.error ?? "Probe-all failed", data: job.result };
  }

  const data = job.result ?? {};
  const errors = (data.errors as { model_id: string; error: string }[] | undefined) ?? [];
  if (errors.length > 0) {
    return {
      ok: true,
      message: `Completed with ${errors.length} error(s). See table for details.`,
      data,
    };
  }
  return { ok: true, message: "All HAR probes completed.", data };
}

export type HarRunSummary = {
  id: string;
  runType: string;
  clipSource: string;
  status: string;
  errorCount: number;
  createdAt: string | null;
};

export type HarResultRow = {
  id: string;
  runId: string;
  modelId: string;
  cameraId: string;
  predictedLabel: string | null;
  confidence: number | null;
  probedAt: string | null;
  status: string;
};

export async function fetchHarRuns(limit = 10): Promise<{ total: number; runs: HarRunSummary[] } | null> {
  try {
    const res = await fetch(`${getApiFetchBase()}/api/har/runs?limit=${limit}`, {
      cache: "no-store",
      headers: authHeaders(),
    });
    if (!res.ok) return null;
    return res.json();
  } catch {
    return null;
  }
}

export async function fetchHarResultsLatest(): Promise<HarResultRow[] | null> {
  try {
    const res = await fetch(`${getApiFetchBase()}/api/har/results/latest`, {
      cache: "no-store",
      headers: authHeaders(),
    });
    if (!res.ok) return null;
    const data = await res.json();
    return data.results ?? [];
  } catch {
    return null;
  }
}

export type HarActivityLogApi = {
  id: string;
  occurredAt: string | null;
  cameraId: string;
  modelId: string;
  modelLabel?: string | null;
  predictedLabel: string | null;
  confidence: number | null;
  topK?: { label: string; prob: number }[];
  isPrimaryAction?: boolean;
  personCount?: number;
  actorName?: string | null;
  source?: string;
  videoName?: string | null;
  clipUrl?: string | null;
  previewUrl?: string | null;
  snapshotUrl?: string | null;
  hyperparams?: Record<string, unknown>;
  hyperparamKey?: string;
  hyperparamLabel?: string;
  inferMs?: number | null;
  backend?: string | null;
};

export type HarModelPerformanceGroup = {
  comboKey?: string;
  modelId?: string;
  modelLabel?: string;
  hyperparamKey?: string;
  hyperparamLabel?: string;
  hyperparams?: Record<string, unknown>;
  videoName?: string | null;
  clipUrl?: string | null;
  previewUrl?: string | null;
  snapshotUrl?: string | null;
  source?: string;
  cameraId?: string;
  modelCount?: number;
  videoCount?: number;
  totalInferences: number;
  avgConfidence: number | null;
  primaryActionRatePct: number | null;
  avgInferMs: number | null;
  topLabel: string | null;
  topLabelPct: number | null;
  byLabel?: Record<string, number>;
};

export type HarModelPerformanceApi = {
  date: string;
  totalLogs: number;
  filteredCount?: number | null;
  hasData: boolean;
  byModel: HarModelPerformanceGroup[];
  byHyperparams: HarModelPerformanceGroup[];
  byCombo: HarModelPerformanceGroup[];
  recentLogs: HarActivityLogApi[];
};

export async function fetchHarModelPerformance(options?: {
  date?: string;
  modelId?: string;
  hyperparamKey?: string;
  comboKey?: string;
  source?: string;
  logsLimit?: number;
}): Promise<HarModelPerformanceApi | null> {
  try {
    const params = new URLSearchParams();
    if (options?.date) params.set("date", options.date);
    if (options?.modelId) params.set("modelId", options.modelId);
    if (options?.hyperparamKey) params.set("hyperparamKey", options.hyperparamKey);
    if (options?.comboKey) params.set("comboKey", options.comboKey);
    if (options?.source) params.set("source", options.source);
    if (options?.logsLimit != null) params.set("logsLimit", String(options.logsLimit));
    const q = params.toString();
    const res = await fetch(
      `${getApiFetchBase()}/api/har/analytics/model-performance${q ? `?${q}` : ""}`,
      { cache: "no-store", headers: authHeaders() },
    );
    if (!res.ok) return null;
    return (await res.json()) as HarModelPerformanceApi;
  } catch {
    return null;
  }
}

export type HarSnapshotTrack = {
  track_id: number;
  display_name?: string | null;
  action_label?: string | null;
  action_confidence?: number | null;
  inferring?: boolean;
};

export type HarSnapshotResult = {
  snapshot_url?: string | null;
  track_count?: number;
  tracks?: HarSnapshotTrack[];
  error?: string;
};

export async function captureHarSnapshot(cameraId: string): Promise<HarSnapshotResult | null> {
  try {
    const res = await fetch(
      `${getApiFetchBase()}/api/har/live/${encodeURIComponent(cameraId)}/snapshot`,
      { method: "POST", headers: authHeaders(), cache: "no-store" },
    );
    if (!res.ok) return null;
    return (await res.json()) as HarSnapshotResult;
  } catch {
    return null;
  }
}

export async function fetchHarActivityLogs(
  cameraId: string,
  limit = 80,
): Promise<HarActivityLogApi[]> {
  try {
    const res = await fetch(
      `${getApiFetchBase()}/api/har/activity?cameraId=${encodeURIComponent(cameraId)}&limit=${limit}`,
      { cache: "no-store", headers: authHeaders() },
    );
    if (!res.ok) return [];
    const data = await res.json();
    return data.logs ?? [];
  } catch {
    return [];
  }
}

export type HarAnalyticsDailyApi = {
  date: string;
  cameraId: string;
  totalInferences: number;
  byLabel: Record<string, number>;
  nonAssemblyRatePct: number;
  assembleSharePct: number;
  hourlyCounts: { hour: number; count: number }[];
  topDeviations: { label: string; count: number }[];
  hasData: boolean;
};

export type HarPlantAnalyticsApi = {
  date: string;
  cameraId?: string | null;
  cameraCount: number;
  totalInferences: number;
  assembleSharePct: number;
  nonAssemblyRatePct: number;
  productivityScore: number;
  primaryActionLabel: string;
  severityTags: { critical: number; warning: number; info: number };
  actionDowntimeMinutes: number;
  actionCostUsd: number;
  lineCostPerMinute: number;
  byCamera: {
    cameraId: string;
    name: string;
    totalInferences: number;
    assembleSharePct: number;
    nonAssemblyRatePct: number;
    productivityScore: number;
    topAction: string | null;
  }[];
  actionPareto: { label: string; count: number; pct: number }[];
  hourlyCounts: { hour: number; count: number }[];
  hasData: boolean;
};

export async function fetchHarAnalyticsPlant(
  cameraId?: string,
  date?: string,
): Promise<HarPlantAnalyticsApi | null> {
  try {
    const q = new URLSearchParams();
    if (cameraId) q.set("cameraId", cameraId);
    if (date) q.set("date", date);
    const res = await fetch(`${getApiFetchBase()}/api/har/analytics/plant?${q}`, {
      cache: "no-store",
      headers: authHeaders(),
    });
    if (!res.ok) return null;
    return res.json();
  } catch {
    return null;
  }
}

export async function fetchHarAnalyticsDaily(
  cameraId: string,
  date?: string,
): Promise<HarAnalyticsDailyApi | null> {
  try {
    const q = new URLSearchParams({ cameraId });
    if (date) q.set("date", date);
    const res = await fetch(`${getApiFetchBase()}/api/har/analytics/daily?${q}`, {
      cache: "no-store",
      headers: authHeaders(),
    });
    if (!res.ok) return null;
    return res.json();
  } catch {
    return null;
  }
}

export async function fetchHarAnalyticsRealtime(cameraId: string, minutes = 30) {
  try {
    const res = await fetch(
      `${getApiFetchBase()}/api/har/analytics/realtime?cameraId=${encodeURIComponent(cameraId)}&minutes=${minutes}`,
      { cache: "no-store", headers: authHeaders() },
    );
    if (!res.ok) return null;
    return res.json();
  } catch {
    return null;
  }
}

export async function cameraAdvisorChat(
  cameraId: string,
  message: string,
  sessionId?: string,
): Promise<{ reply: string; usedFallback: boolean } | null> {
  try {
    const res = await fetch(`${getApiFetchBase()}/api/advisor/camera-chat`, {
      method: "POST",
      headers: apiAuthHeaders(),
      body: JSON.stringify({ cameraId, message, sessionId }),
    });
    if (!res.ok) return null;
    const data = await res.json();
    return { reply: data.reply ?? "", usedFallback: Boolean(data.usedFallback) };
  } catch {
    return null;
  }
}

export async function fetchFaceStatus(): Promise<FaceStatus | null> {
  try {
    const res = await fetch(`${getApiFetchBase()}/api/faces/status`, { cache: "no-store" });
    if (!res.ok) return null;
    const data = (await res.json()) as FaceStatus;
    if (data.previewUrl) {
      data.previewUrl = `${getApiFetchBase()}/api/faces/preview`;
    }
    return data;
  } catch {
    return null;
  }
}

export async function fetchFaceStorage(): Promise<FaceStorageInfo | null> {
  try {
    const res = await fetch(`${getApiFetchBase()}/api/faces/storage`, { cache: "no-store" });
    if (!res.ok) return null;
    return (await res.json()) as FaceStorageInfo;
  } catch {
    return null;
  }
}

export async function enrollFace(name: string): Promise<{ ok: boolean; message: string; data?: unknown }> {
  const res = await fetch(`${getApiFetchBase()}/api/faces/enroll`, {
    method: "POST",
    headers: apiAuthHeaders(),
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
  const res = await fetch(`${getApiFetchBase()}/api/faces/enroll`, { method: "DELETE" });
  if (!res.ok) {
    const data = await res.json().catch(() => ({}));
    return { ok: false, message: (data as { detail?: string }).detail ?? "Delete failed" };
  }
  return { ok: true, message: "Enrollment removed." };
}

// --- Ops API (auth, alerts, timeline, analytics — unified backend :8000) ---

export type AlertCaseType =
  | "user_not_working"
  | "user_left_position"
  | "forklift_in_zone"
  | "har_action_detected"
  | "unknown";

export type AlertRuleApi = {
  id: string;
  icon: string;
  title: string;
  description: string;
  zone: string;
  caseType?: AlertCaseType;
  severity: "CRITICAL" | "WARNING" | "INFO" | "DISABLED";
  enabled: boolean;
  notifyEmail?: boolean;
  notifyTelegram?: boolean;
  notifyInApp?: boolean;
  emailTemplateId?: string | null;
  updatedAt?: string | null;
  updatedBy?: string | null;
  confidenceThreshold?: number;
  harActionLabel?: string;
  eventSeverity?: "info" | "warning" | "critical";
};

export type EmailTemplateApi = {
  id: string;
  slug: string;
  name: string;
  caseType: AlertCaseType;
  category: string;
  severityLevel: "warning" | "critical" | "info";
  headline: string;
  body: string;
  subject: string;
  footerReason: string;
  layout: string;
  snapshotUrl?: string | null;
  isBuiltin: boolean;
  enabled: boolean;
};

export type EmailTemplatePreview = {
  templateId: string;
  subject: string;
  html: string;
  text: string;
};

export type AlertActionApi = {
  caseType: AlertCaseType;
  harActionLabel?: string;
  label: string;
  description: string;
  icon: string;
  defaultZone: string;
  defaultSeverity: string;
  defaultEnabled: boolean;
};

export type AlertRuleInput = Omit<AlertRuleApi, "id">;

export type ResolutionStatus = "OPEN" | "ACKNOWLEDGED" | "RESOLVED" | "FALSE_POSITIVE";

export type TimelineEventApi = {
  id: string;
  time: string;
  severity: "critical" | "warning" | "info" | "normal";
  title: string;
  description: string;
  meta: { icon: string; text: string }[];
  thumbnail: string;
  clipDuration: string;
  clipUrl?: string;
  cameraId?: string;
  caseType?: string;
  occurredAt?: string;
  resolutionStatus: ResolutionStatus;
  acknowledgedAt?: string | null;
  acknowledgedBy?: string | null;
  resolvedAt?: string | null;
  resolvedBy?: string | null;
  industrialReasonCode?: string | null;
  downtimeCausedSeconds?: number;
  scrapCausedUnits?: number;
  closureNotes?: string | null;
  hiddenFromPanel?: boolean;
  harSource?: boolean;
};

function normalizeTimelineEvent(event: TimelineEventApi): TimelineEventApi {
  return {
    ...event,
    thumbnail: proxyMediaUrl(event.thumbnail) ?? event.thumbnail,
  };
}

export type ReasonCodeApi = { code: string; label: string; category: string };

export type ShiftSummaryApi = {
  date: string;
  incidentCount: number;
  incidentDelta: string;
  uptime: string;
  uptimePct?: number;
  assets: { name: string; events: number }[];
  totalEvents?: number;
  openCount?: number;
  acknowledgedCount?: number;
  resolvedCount?: number;
  falsePositiveCount?: number;
  openCriticalCount?: number;
  openWarningCount?: number;
  totalIncidents?: number;
  infoActivityCount?: number;
  allClear?: boolean;
  avgAckSeconds?: number | null;
  topReasonCodes?: { code: string; count: number }[];
};

export type ShiftAiSummaryApi = {
  generatedAt: string;
  date: string;
  shift: string;
  shiftLabel: string;
  siteName: string;
  currentStatus: "all_clear" | "action_needed" | "critical";
  statusHeadline: string;
  statusDetail: string;
  allClear: boolean;
  narrative: string;
  highlights: string[];
  suggestions: string[];
  recommendations: string[];
  metrics: {
    totalEvents: number;
    openCount: number;
    openCriticalCount: number;
    acknowledgedCount: number;
    resolvedCount: number;
    falsePositiveCount: number;
    uptime: string;
    uptimePct?: number;
    oee: number;
    availability: number;
    performance: number;
    quality: number;
    flowEfficiency: number;
    coqTotalUsd: number;
    avgAckSeconds: number | null;
  };
  severityCounts: {
    critical: number;
    warning: number;
    info: number;
    normal: number;
  };
};

export type AnalyticsSummaryApi = {
  date: string;
  shift: string;
  cameraId?: string | null;
  flowEfficiency: string;
  flowEfficiencyTrend: string;
  incidentCount: number;
  uptime: string;
  oee?: string;
  openCriticalCount?: number;
};

export type AnalyticsOeeApi = {
  date: string;
  shift: string;
  cameraId?: string | null;
  availability: number;
  performance: number;
  quality: number;
  oee: number;
};

export type AnalyticsCoqApi = {
  date: string;
  shift: string;
  downtimeMinutes: number;
  scrapUnits: number;
  downtimeCostUsd: number;
  scrapCostUsd: number;
  totalCostUsd: number;
  lineCostPerMinute: number;
  materialCostPerUnit: number;
  actionDeviationCostUsd?: number;
  actionDowntimeMinutes?: number;
};

export type AnalyticsParetoApi = {
  date: string;
  shift: string;
  items: { code: string; label: string; count: number; pct: number; cumulativePct: number }[];
  totalTagged: number;
};

export type TimelineStatsApi = {
  date: string;
  openCount: number;
  acknowledgedCount: number;
  resolvedCount: number;
  falsePositiveCount: number;
  openCriticalCount: number;
  allClear: boolean;
  avgAckSeconds: number | null;
  topReasonCodes: { code: string; count: number }[];
};

export type AnalyticsHeatmapApi = {
  cameraId: string;
  date: string;
  shift: string;
  grid: { width: number; height: number; cells: number[][]; hotspots?: unknown[] };
  anomalyCount: number;
  sensorsActive: number;
  source?: string;
  sourceLabel?: string;
  hasData?: boolean;
  severityTags?: { critical: number; warning: number; info: number };
};

export type AnalyticsInsightsApi = {
  date: string;
  shift: string;
  cameraId?: string | null;
  flowEfficiency?: string;
  flowHistory?: { date: string; value: number }[];
  downtimeByStation: { name: string; minutes: number; widthPct: string; critical: boolean }[];
  bottlenecks: { id: string; title: string; severity: string; description: string; critical: boolean }[];
  recommendation: string;
  severityTags?: { critical: number; warning: number; info: number };
  harActions?: HarPlantAnalyticsApi | null;
};

export type PlantSettingsApi = {
  siteName: string;
  lineCostPerMinute: number;
  materialCostPerUnit: number;
  targetCycleSec: number;
  shiftHours: number;
  uptimeCriticalPenalty: number;
  uptimeWarningPenalty: number;
  uptimeFloorPct: number;
  uptimeCeilingPct: number;
  performanceFloorPct: number;
  performanceCeilingPct: number;
  qualityFloorPct: number;
  qualityCeilingPct: number;
  defaultClipDurationSec: number;
  downtimeCriticalThresholdPct: number;
  inferenceBasePerCamera: number;
  inferenceProbeBonus: number;
  inferenceEventMultiplier: number;
  inferenceMinPerCamera: number;
  updatedAt?: string | null;
  updatedBy?: string | null;
};

export type KpiDefinitionApi = {
  id: string;
  label: string;
  formula: string;
  description: string;
  settingsKeys: string[];
};

const EMPTY_SHIFT_SUMMARY: ShiftSummaryApi = {
  date: new Date().toLocaleDateString("en-US", { weekday: "long", month: "short", day: "numeric", year: "numeric" }),
  incidentCount: 0,
  incidentDelta: "No incidents",
  uptime: "—",
  assets: [],
};

function mapRealtimeSeverity(severity: TimelineEventApi["severity"]): RealtimeEvent["severity"] {
  if (severity === "critical") return "critical";
  if (severity === "warning") return "primary";
  return "neutral";
}

export async function fetchEmailTemplates(caseType?: AlertCaseType): Promise<EmailTemplateApi[]> {
  const qs = caseType ? `?caseType=${caseType}` : "";
  const res = await fetchApi(`/api/alerts/email-templates${qs}`, { cache: "no-store" });
  if (!res?.ok) return [];
  return res.json();
}

export async function createEmailTemplate(input: {
  name: string;
  baseTemplateId: string;
  subject?: string;
  headline?: string;
  body?: string;
  category?: string;
  footerReason?: string;
}): Promise<EmailTemplateApi | null> {
  const res = await fetch(`${getApiFetchBase()}/api/alerts/email-templates`, {
    method: "POST",
    headers: apiAuthHeaders(),
    body: JSON.stringify(input),
  });
  if (!res.ok) return null;
  return res.json();
}

export async function updateEmailTemplate(
  id: string,
  patch: Partial<{
    name: string;
    subject: string;
    headline: string;
    body: string;
    category: string;
    footerReason: string;
    enabled: boolean;
  }>,
): Promise<EmailTemplateApi | null> {
  const res = await fetch(`${getApiFetchBase()}/api/alerts/email-templates/${id}`, {
    method: "PATCH",
    headers: apiAuthHeaders(),
    body: JSON.stringify(patch),
  });
  if (!res.ok) return null;
  return res.json();
}

export async function deleteEmailTemplate(id: string): Promise<boolean> {
  const res = await fetch(`${getApiFetchBase()}/api/alerts/email-templates/${id}`, {
    method: "DELETE",
    headers: authHeaders(),
  });
  return res.ok;
}

export async function previewEmailTemplate(id: string): Promise<EmailTemplatePreview | null> {
  const res = await fetch(`${getApiFetchBase()}/api/alerts/email-templates/${id}/preview`, {
    method: "POST",
  });
  if (!res.ok) return null;
  return res.json();
}

export async function fetchAlertActions(): Promise<AlertActionApi[]> {
  const res = await fetchApi("/api/alerts/actions", { cache: "no-store" });
  if (!res?.ok) return [];
  return res.json();
}

export async function fetchAlertRules(): Promise<AlertRuleApi[]> {
  const res = await fetchApi("/api/alerts/rules", { cache: "no-store" });
  if (!res?.ok) return [];
  return res.json();
}

export async function createAlertRule(rule: AlertRuleInput): Promise<AlertRuleApi | null> {
  const res = await fetch(`${getApiFetchBase()}/api/alerts/rules`, {
    method: "POST",
    headers: apiAuthHeaders(),
    body: JSON.stringify(rule),
  });
  if (!res.ok) return null;
  return res.json();
}

export async function updateAlertRule(
  ruleId: string,
  patch: Partial<AlertRuleInput>,
): Promise<AlertRuleApi | null> {
  const res = await fetch(`${getApiFetchBase()}/api/alerts/rules/${ruleId}`, {
    method: "PATCH",
    headers: apiAuthHeaders(),
    body: JSON.stringify(patch),
  });
  if (!res.ok) return null;
  return res.json();
}

export async function deleteAlertRule(ruleId: string): Promise<boolean> {
  const res = await fetch(`${getApiFetchBase()}/api/alerts/rules/${ruleId}`, {
    method: "DELETE",
    headers: authHeaders(),
  });
  return res.ok;
}

export async function toggleAlertRule(ruleId: string): Promise<AlertRuleApi | null> {
  const res = await fetch(`${getApiFetchBase()}/api/alerts/rules/${ruleId}/toggle`, {
    method: "POST",
    headers: authHeaders(),
  });
  if (!res.ok) return null;
  return res.json();
}

export async function fetchTimelineEvents(
  options?: {
    limit?: number;
    severity?: string;
    cameraId?: string;
    resolutionStatus?: ResolutionStatus;
    harOnly?: boolean;
    caseType?: string;
    /** When true (default), hides events removed from the timeline panel. */
    panelOnly?: boolean;
  },
): Promise<TimelineEventApi[]> {
  const params = new URLSearchParams();
  params.set("limit", String(options?.limit ?? 50));
  params.set("panelOnly", String(options?.panelOnly !== false));
  if (options?.severity) params.set("severity", options.severity);
  if (options?.cameraId) params.set("cameraId", options.cameraId);
  if (options?.resolutionStatus) params.set("resolutionStatus", options.resolutionStatus);
  if (options?.harOnly) params.set("harOnly", "true");
  if (options?.caseType) params.set("caseType", options.caseType);
  const res = await fetchApi(`/api/timeline?${params}`, { cache: "no-store" });
  if (!res?.ok) return [];
  const events = (await res.json()) as TimelineEventApi[];
  return events.map(normalizeTimelineEvent);
}

export async function fetchTimelineStats(): Promise<TimelineStatsApi | null> {
  const res = await fetchApi("/api/timeline/stats", { cache: "no-store" });
  if (!res?.ok) return null;
  return res.json();
}

export async function fetchReasonCodes(): Promise<ReasonCodeApi[]> {
  const res = await fetch(`${getApiFetchBase()}/api/timeline/reason-codes`, { cache: "no-store" });
  if (!res.ok) return [];
  return res.json();
}

export async function acknowledgeTimelineEvent(eventId: string): Promise<TimelineEventApi | null> {
  const res = await fetch(`${getApiFetchBase()}/api/timeline/${eventId}/acknowledge`, {
    method: "PATCH",
    headers: apiAuthHeaders(),
  });
  if (!res.ok) return null;
  return normalizeTimelineEvent(await res.json());
}

export async function resolveTimelineEvent(
  eventId: string,
  body: {
    status: "RESOLVED" | "FALSE_POSITIVE";
    reasonCode?: string;
    downtimeSeconds?: number;
    scrapUnits?: number;
    notes?: string;
  },
): Promise<TimelineEventApi | null> {
  const res = await fetch(`${getApiFetchBase()}/api/timeline/${eventId}/resolve`, {
    method: "PATCH",
    headers: apiAuthHeaders(),
    body: JSON.stringify({
      status: body.status,
      reasonCode: body.reasonCode,
      downtimeSeconds: body.downtimeSeconds ?? 0,
      scrapUnits: body.scrapUnits ?? 0,
      notes: body.notes,
    }),
  });
  if (!res.ok) return null;
  return normalizeTimelineEvent(await res.json());
}

export async function dismissTimelineEvent(eventId: string): Promise<TimelineEventApi | null> {
  const res = await fetch(`${getApiFetchBase()}/api/timeline/${eventId}/dismiss`, {
    method: "PATCH",
    headers: apiAuthHeaders(),
  });
  if (!res.ok) return null;
  return normalizeTimelineEvent(await res.json());
}

export async function fetchShiftSummary(): Promise<ShiftSummaryApi> {
  const res = await fetch(`${getApiFetchBase()}/api/timeline/summary`, { cache: "no-store" });
  if (!res.ok) return EMPTY_SHIFT_SUMMARY;
  return res.json();
}

export async function fetchShiftAiSummary(): Promise<ShiftAiSummaryApi | null> {
  const res = await fetch(`${getApiFetchBase()}/api/timeline/ai-summary`, { cache: "no-store" });
  if (!res.ok) return null;
  return res.json();
}

export async function fetchRealtimeEvents(limit = 8): Promise<RealtimeEvent[]> {
  const events = await fetchTimelineEvents({ limit, resolutionStatus: "OPEN" });
  return events.map((e) => ({
    id: e.id,
    time: e.time,
    title: e.title,
    description: e.description,
    severity: mapRealtimeSeverity(e.severity),
    resolutionStatus: e.resolutionStatus,
  }));
}

export async function fetchAnalyticsSummary(
  shift = "morning",
  cameraId?: string,
): Promise<AnalyticsSummaryApi | null> {
  const params = new URLSearchParams({ shift });
  if (cameraId) params.set("cameraId", cameraId);
  const res = await fetch(`${getApiFetchBase()}/api/analytics/summary?${params}`, {
    cache: "no-store",
  });
  if (!res.ok) return null;
  return res.json();
}

export async function fetchAnalyticsHeatmap(
  shift = "morning",
  cameraId = "cam-01",
): Promise<AnalyticsHeatmapApi | null> {
  const params = new URLSearchParams({ shift, cameraId });
  const res = await fetch(`${getApiFetchBase()}/api/analytics/heatmap?${params}`, {
    cache: "no-store",
  });
  if (!res.ok) return null;
  return res.json();
}

export async function fetchAnalyticsInsights(
  shift = "morning",
  cameraId?: string,
): Promise<AnalyticsInsightsApi | null> {
  const params = new URLSearchParams({ shift });
  if (cameraId) params.set("cameraId", cameraId);
  const res = await fetch(`${getApiFetchBase()}/api/analytics/insights?${params}`, {
    cache: "no-store",
  });
  if (!res.ok) return null;
  return res.json();
}

export async function fetchAnalyticsOee(
  shift = "morning",
  cameraId?: string,
): Promise<AnalyticsOeeApi | null> {
  const params = new URLSearchParams({ shift });
  if (cameraId) params.set("cameraId", cameraId);
  const res = await fetch(`${getApiFetchBase()}/api/analytics/oee?${params}`, { cache: "no-store" });
  if (!res.ok) return null;
  return res.json();
}

export async function fetchAnalyticsCoq(
  shift = "morning",
  cameraId?: string,
): Promise<AnalyticsCoqApi | null> {
  const params = new URLSearchParams({ shift });
  if (cameraId) params.set("cameraId", cameraId);
  const res = await fetch(`${getApiFetchBase()}/api/analytics/coq?${params}`, { cache: "no-store" });
  if (!res.ok) return null;
  return res.json();
}

export async function fetchAnalyticsPareto(
  shift = "morning",
  cameraId?: string,
): Promise<AnalyticsParetoApi | null> {
  const params = new URLSearchParams({ shift });
  if (cameraId) params.set("cameraId", cameraId);
  const res = await fetch(`${getApiFetchBase()}/api/analytics/pareto?${params}`, {
    cache: "no-store",
  });
  if (!res.ok) return null;
  return res.json();
}

export type EmailNotificationStatus = {
  provider: string;
  channel: string;
  configured: boolean;
  dryRun: boolean;
  fromEmail: string | null;
  toEmails: string[];
  status: "ready" | "dry_run" | "not_configured";
};

export type TelegramNotificationStatus = {
  provider: string;
  channel: string;
  configured: boolean;
  dryRun: boolean;
  chatIds: string[];
  chatCount: number;
  status: "ready" | "dry_run" | "not_configured";
};

export type TelemetryMetric = {
  service: string;
  label: string;
  status: string;
  value: string;
  detail: string;
  latencyMs: number | null;
  bars: number[];
  highlight: boolean;
};

export type TelemetryResponse = {
  collectedAt: string;
  metrics: TelemetryMetric[];
};


export async function fetchEmailNotificationStatus(): Promise<EmailNotificationStatus | null> {
  const res = await fetch(`${getApiFetchBase()}/api/notifications/email/status`, {
    cache: "no-store",
  });
  if (!res.ok) return null;
  return res.json();
}

export async function fetchTelegramNotificationStatus(): Promise<TelegramNotificationStatus | null> {
  const res = await fetch(`${getApiFetchBase()}/api/notifications/telegram/status`, {
    cache: "no-store",
  });
  if (!res.ok) return null;
  return res.json();
}

export async function fetchTelemetry(): Promise<TelemetryResponse | null> {
  const res = await fetchApi("/api/telemetry", { cache: "no-store" });
  if (!res?.ok) return null;
  return res.json();
}

export async function fetchPlantSettings(): Promise<PlantSettingsApi | null> {
  const res = await fetch(`${getApiFetchBase()}/api/settings/plant`, { cache: "no-store" });
  if (!res.ok) return null;
  return res.json();
}

export async function updatePlantSettings(
  data: Partial<PlantSettingsApi>,
): Promise<PlantSettingsApi | null> {
  const res = await fetch(`${getApiFetchBase()}/api/settings/plant`, {
    method: "PATCH",
    headers: apiAuthHeaders(),
    body: JSON.stringify(data),
  });
  if (!res.ok) return null;
  return res.json();
}

let kpiDefinitionsCache: KpiDefinitionApi[] | null = null;

export async function fetchKpiDefinitions(): Promise<KpiDefinitionApi[]> {
  if (kpiDefinitionsCache) return kpiDefinitionsCache;
  const res = await fetch(`${getApiFetchBase()}/api/settings/kpi-definitions`, { cache: "no-store" });
  if (!res.ok) return [];
  const data = (await res.json()) as { items: KpiDefinitionApi[] };
  kpiDefinitionsCache = data.items;
  return data.items;
}

export function clearKpiDefinitionsCache() {
  kpiDefinitionsCache = null;
}

export type AdvisorChatRequest = {
  message: string;
  page: string;
  pageTitle?: string;
  alerts?: { id: string; title: string; severity: string; time: string; description: string }[];
};

export type AdvisorChatResponse = {
  reply: string;
  model: string;
  usedFallback: boolean;
  snapshot: {
    openCriticalCount: number;
    openCount: number;
    camerasLive: number;
  };
};

export type AdvisorWelcomeResponse = {
  intro: string;
  pageContext: string;
  status: string;
  welcome: string;
  quickPrompts: string[];
  page: string;
  pageTitle: string;
};

export async function fetchAdvisorWelcome(
  page: string,
  pageTitle?: string,
): Promise<AdvisorWelcomeResponse | null> {
  const params = new URLSearchParams({ page });
  if (pageTitle) params.set("pageTitle", pageTitle);
  const res = await fetch(`${getApiFetchBase()}/api/advisor/welcome?${params}`, {
    cache: "no-store",
  });
  if (!res.ok) return null;
  return res.json();
}

export async function advisorChat(body: AdvisorChatRequest): Promise<AdvisorChatResponse> {
  const res = await fetch(`${getApiFetchBase()}/api/advisor/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json", ...authHeaders() },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    const detail = typeof err.detail === "string" ? err.detail : `HTTP ${res.status}`;
    throw new Error(detail);
  }
  return res.json();
}


export async function fetchRecentNotifications(limit = 12): Promise<RealtimeEvent[]> {
  const res = await fetch(`${getApiFetchBase()}/api/notifications/recent?limit=${limit}`, {
    cache: "no-store",
  });
  if (!res.ok) return [];
  const data = (await res.json()) as { events?: TimelineEventApi[] };
  return (data.events ?? []).map((e) => ({
    id: e.id,
    time: e.time,
    title: e.title,
    description: e.description,
    severity: mapRealtimeSeverity(e.severity),
    resolutionStatus: e.resolutionStatus,
  }));
}

export async function sendTestHarAlertEmail(
  harActionLabel: string,
): Promise<{ ok: boolean; message: string }> {
  const res = await fetch(`${getApiFetchBase()}/api/alerting/har/test-email`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ harActionLabel }),
  });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) {
    const detail = typeof data.detail === "string" ? data.detail : "Send failed";
    return { ok: false, message: detail };
  }
  return { ok: true, message: (data.message as string) ?? `Test email sent for ${harActionLabel}.` };
}

export async function sendTestHarAlertTelegram(
  harActionLabel: string,
): Promise<{ ok: boolean; message: string }> {
  const res = await fetch(`${getApiFetchBase()}/api/alerting/har/test-telegram`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ harActionLabel }),
  });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) {
    const detail = typeof data.detail === "string" ? data.detail : "Send failed";
    return { ok: false, message: detail };
  }
  return { ok: true, message: (data.message as string) ?? `Test Telegram sent for ${harActionLabel}.` };
}

export async function sendTestAlertEmail(
  caseType: AlertCaseType,
): Promise<{ ok: boolean; message: string }> {
  const res = await fetch(`${getApiFetchBase()}/api/alerting/email/test/${caseType}`, {
    method: "POST",
  });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) {
    const detail = typeof data.detail === "string" ? data.detail : "Send failed";
    return { ok: false, message: detail };
  }
  const event = data.event as {
    dryRun?: boolean;
    caseType?: string;
    messageIds?: string[];
    telegramDryRun?: boolean;
    telegramMessageIds?: string[];
  };
  if (event.dryRun && event.telegramDryRun) {
    return { ok: true, message: `Dry-run: ${caseType} rendered (no email or Telegram sent).` };
  }
  if (event.dryRun) {
    return { ok: true, message: `Dry-run: ${caseType} email rendered (no email sent).` };
  }
  if (event.telegramDryRun) {
    return { ok: true, message: `Sent ${caseType} email; Telegram dry-run only.` };
  }
  return { ok: true, message: `Sent ${caseType} alert via enabled channels.` };
}

export async function sendTestAlertTelegram(
  caseType: AlertCaseType,
): Promise<{ ok: boolean; message: string }> {
  const res = await fetch(`${getApiFetchBase()}/api/alerting/telegram/test/${caseType}`, {
    method: "POST",
  });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) {
    const detail = typeof data.detail === "string" ? data.detail : "Send failed";
    return { ok: false, message: detail };
  }
  const event = data.event as {
    dryRun?: boolean;
    telegramDryRun?: boolean;
    telegramMessageIds?: string[];
    telegramError?: string | null;
  };
  if (event.telegramDryRun) {
    return { ok: true, message: `Dry-run: ${caseType} Telegram message rendered (not sent).` };
  }
  if (event.telegramMessageIds?.length) {
    const warn = event.telegramError ? ` Warning: ${event.telegramError}` : "";
    return { ok: true, message: `Sent ${caseType} alert to Telegram (${event.telegramMessageIds.length} message(s)).${warn}` };
  }
  if (event.telegramError) {
    return { ok: false, message: event.telegramError };
  }
  return { ok: true, message: `Sent ${caseType} alert via enabled channels.` };
}
