/** har-research parity — /api/har/v2 client */

import { authHeaders } from "@/lib/auth";
import { getApiBase, getApiFetchBase } from "@/lib/api";

export function harV2Base(): string {
  return getApiFetchBase();
}

async function harV2Fetch<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${harV2Base()}${path}`, {
    ...init,
    headers: {
      "Content-Type": "application/json",
      ...authHeaders(),
      ...(init?.headers ?? {}),
    },
    cache: "no-store",
  });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(text || `HAR v2 ${path} failed (${res.status})`);
  }
  return res.json() as Promise<T>;
}

export type HarAuditSession = {
  session_id: string;
  source: string;
  camera_id?: string | null;
  video_name?: string | null;
  model_id?: string | null;
  model_tag?: string | null;
  started_at?: string | null;
  ended_at?: string | null;
  n_events: number;
  n_confirmed: number;
  n_tracks: number;
  hyperparams?: Record<string, unknown>;
};

export type HarSessionEvent = {
  event_id: string;
  session_id: string;
  track_id: number;
  global_person_id?: string | null;
  reid_match_score?: number | null;
  frame_idx?: number | null;
  action_label?: string | null;
  raw_label?: string | null;
  confidence?: number | null;
  label_changed?: boolean;
  uncertain?: boolean;
  occurred_at?: string | null;
  crop_url?: string | null;
  frame_url?: string | null;
};

export type HarTrackSummary = {
  session_id: string;
  track_id: number;
  global_person_id?: string | null;
  display_name?: string | null;
  /** True when this track has any HITL label saved (including false-detection rejections). */
  has_label?: boolean;
  n_inferences: number;
  frame_first?: number | null;
  frame_last?: number | null;
  dominant_action?: string;
  dominant_confidence?: number;
  sample_crop_url?: string | null;
};

export type HarPerson = {
  global_person_id: string;
  display_name?: string | null;
  n_appearances: number;
  n_tracks?: number;
  first_seen_at?: string | null;
  last_seen_at?: string | null;
  thumbnail_url?: string | null;
};

export type HarPersonSessionRow = {
  session_id: string;
  track_id: number;
  video?: string | null;
  n_appearances?: number;
  frame_first?: number | null;
  frame_last?: number | null;
  dominant_action?: string | null;
  match_score?: number | null;
  sample_crop_url?: string | null;
};

export type HarPersonHistoryRow = {
  session_id: string;
  track_id: number;
  event_id?: string | null;
  video_name?: string | null;
  frame_idx?: number | null;
  action_label?: string | null;
  confidence?: number | null;
  match_score?: number | null;
  occurred_at?: string | null;
  crop_url?: string | null;
};

export type HarPersonMetrics = {
  n_events: number;
  n_sessions: number;
  n_tracks: number;
  n_videos: number;
  videos: string[];
  avg_confidence?: number | null;
  min_confidence?: number | null;
  max_confidence?: number | null;
  avg_reid_match?: number | null;
  min_reid_match?: number | null;
  max_reid_match?: number | null;
  n_uncertain: number;
  n_label_changes: number;
  dominant_action?: string | null;
  first_seen_at?: string | null;
  last_seen_at?: string | null;
  frame_first?: number | null;
  frame_last?: number | null;
};

export type HarPersonActionBreakdown = {
  action: string;
  count: number;
  share: number;
  avg_confidence?: number | null;
};

export type HarPersonSnapshot = {
  event_id: string;
  session_id: string;
  track_id: number;
  frame_idx?: number | null;
  action_label?: string | null;
  confidence?: number | null;
  occurred_at?: string | null;
  crop_url?: string | null;
  frame_url?: string | null;
  uncertain?: boolean;
  label_changed?: boolean;
};

export type HarPersonReport = HarPerson & {
  metrics: HarPersonMetrics;
  action_breakdown: HarPersonActionBreakdown[];
  sessions?: HarPersonSessionRow[];
  snapshots: HarPersonSnapshot[];
  events: HarSessionEvent[];
  human_labels: Array<Record<string, unknown>>;
  track_labels: Array<Record<string, unknown>>;
  events_shown: number;
  events_truncated: boolean;
  snapshots_shown: number;
  snapshots_truncated: boolean;
};

export async function fetchHarV2Sessions(limit = 50): Promise<HarAuditSession[]> {
  const data = await harV2Fetch<{ sessions: HarAuditSession[] }>(
    `/api/har/v2/sessions?limit=${limit}`,
  );
  return data.sessions ?? [];
}

export async function fetchHarV2Session(sessionId: string): Promise<HarAuditSession & { summary?: Record<string, unknown> }> {
  return harV2Fetch(`/api/har/v2/sessions/${encodeURIComponent(sessionId)}`);
}

export async function fetchHarV2Tracks(sessionId: string): Promise<HarTrackSummary[]> {
  const data = await harV2Fetch<{ tracks: HarTrackSummary[] }>(
    `/api/har/v2/sessions/${encodeURIComponent(sessionId)}/tracks`,
  );
  return data.tracks ?? [];
}

export async function fetchHarV2Events(sessionId: string, trackId?: number): Promise<HarSessionEvent[]> {
  const qs = trackId != null ? `?track_id=${trackId}` : "";
  const data = await harV2Fetch<{ events: HarSessionEvent[] }>(
    `/api/har/v2/sessions/${encodeURIComponent(sessionId)}/events${qs}`,
  );
  return data.events ?? [];
}

export async function fetchHarV2Persons(limit = 100): Promise<HarPerson[]> {
  const data = await harV2Fetch<{ persons: HarPerson[] }>(`/api/har/v2/persons?limit=${limit}`);
  return data.persons ?? [];
}

export async function fetchHarV2Person(globalPersonId: string) {
  return harV2Fetch<HarPerson & { sessions?: HarPersonSessionRow[]; history?: HarPersonHistoryRow[] }>(
    `/api/har/v2/persons/${encodeURIComponent(globalPersonId)}`,
  );
}

export async function fetchHarV2PersonHistory(
  globalPersonId: string,
  limit = 500,
): Promise<HarPersonHistoryRow[]> {
  const data = await harV2Fetch<{ history: HarPersonHistoryRow[] }>(
    `/api/har/v2/persons/${encodeURIComponent(globalPersonId)}/history?limit=${limit}`,
  );
  return data.history ?? [];
}

export async function fetchHarV2PersonReport(globalPersonId: string): Promise<HarPersonReport> {
  return harV2Fetch<HarPersonReport>(
    `/api/har/v2/persons/${encodeURIComponent(globalPersonId)}/report?snapshot_limit=60&event_limit=1000`,
  );
}

export async function saveHarV2PersonName(globalPersonId: string, displayName: string): Promise<void> {
  await harV2Fetch(`/api/har/v2/persons/${encodeURIComponent(globalPersonId)}`, {
    method: "PATCH",
    body: JSON.stringify({ display_name: displayName }),
  });
}

export async function splitHarV2TrackIdentity(
  sessionId: string,
  trackId: number,
): Promise<{ new_global_person_id: string }> {
  return harV2Fetch("/api/har/v2/persons/split-track", {
    method: "POST",
    body: JSON.stringify({ session_id: sessionId, track_id: trackId }),
  });
}

export async function mergeHarV2Persons(
  targetGlobalPersonId: string,
  sourceGlobalPersonIds: string[],
): Promise<{
  target_global_person_id: string;
  merged_global_person_ids: string[];
  appearances_moved: number;
  events_moved: number;
}> {
  return harV2Fetch("/api/har/v2/persons/merge", {
    method: "POST",
    body: JSON.stringify({
      target_global_person_id: targetGlobalPersonId,
      source_global_person_ids: sourceGlobalPersonIds,
    }),
  });
}

export async function saveHarV2TrackLabel(payload: {
  session_id: string;
  track_id: number;
  person_verdict: string;
  display_name?: string;
  global_person_id?: string;
  action_notes?: string;
  video?: string;
  dominant_action?: string;
  dominant_confidence?: number;
  n_events?: number;
}): Promise<{ global_person_id?: string | null; registry_updated?: boolean }> {
  const data = await harV2Fetch<{ label?: { global_person_id?: string; registry_updated?: boolean } }>(
    "/api/har/v2/track-labels",
    {
      method: "POST",
      body: JSON.stringify({ ...payload, reviewer: "supervisor" }),
    },
  );
  return {
    global_person_id: data.label?.global_person_id,
    registry_updated: data.label?.registry_updated,
  };
}

export async function fetchHarV2ReviewQueue(limit = 30): Promise<HarSessionEvent[]> {
  const data = await harV2Fetch<{ queue: HarSessionEvent[] }>(`/api/har/v2/review-queue?limit=${limit}`);
  return data.queue ?? [];
}

export type HarActionLabels = {
  model_labels: string[];
  custom_labels: string[];
  all_labels: string[];
};

export async function fetchHarV2ActionLabels(): Promise<HarActionLabels> {
  return harV2Fetch<HarActionLabels>("/api/har/v2/action-labels");
}

export async function saveHarV2HumanLabel(payload: {
  session_id?: string;
  event_id?: string;
  track_id?: number;
  person_verdict?: string;
  action_verdict?: string;
  correct_label?: string;
  usable_for_training?: boolean;
  notes?: string;
}): Promise<void> {
  await harV2Fetch("/api/har/v2/human-labels", {
    method: "POST",
    body: JSON.stringify({ ...payload, reviewer: "supervisor" }),
  });
}

export async function downloadHarV2PersonPdf(globalPersonId: string): Promise<void> {
  const res = await fetch(
    `${harV2Base()}/api/har/v2/persons/${encodeURIComponent(globalPersonId)}/pdf`,
    { headers: { ...authHeaders() }, cache: "no-store" },
  );
  if (!res.ok) {
    const text = await res.text();
    throw new Error(text || `PDF generation failed (${res.status})`);
  }
  const blob = await res.blob();
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = `visionops-report-${globalPersonId}.pdf`;
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}

export function harV2ArtifactUrl(relativeOrAbsolute: string | null | undefined): string | null {
  if (!relativeOrAbsolute) return null;
  if (relativeOrAbsolute.startsWith("http")) return relativeOrAbsolute;
  return `${harV2Base()}${relativeOrAbsolute}`;
}

export type HarEvalJob = {
  job_id: string;
  status: string;
  progress: number;
  frames_done: number;
  max_frames: number;
  session_id?: string | null;
  error?: string | null;
  track_stats?: Array<Record<string, unknown>>;
};

export type HarPreviewFrame = {
  frame_idx: number;
  url: string;
  n_tracks: number;
  tracks: Array<Record<string, unknown>>;
};

function visionBase(): string {
  return getApiFetchBase();
}

export async function startHarEval(payload: {
  model_id: string;
  video: string;
  max_frames?: number;
  infer_every?: number;
}): Promise<{ job_id: string }> {
  const res = await fetch(`${visionBase()}/api/vision/har/eval/run`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function fetchHarEvalJob(jobId: string): Promise<HarEvalJob> {
  const res = await fetch(`${visionBase()}/api/vision/har/eval/jobs/${encodeURIComponent(jobId)}`, {
    cache: "no-store",
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function fetchHarPreviewFrames(sessionId: string): Promise<{
  frames: HarPreviewFrame[];
  summary: Record<string, unknown>;
}> {
  const data = await harV2Fetch<{ frames: HarPreviewFrame[]; summary: Record<string, unknown> }>(
    `/api/har/v2/sessions/${encodeURIComponent(sessionId)}/preview-frames`,
  );
  return { frames: data.frames ?? [], summary: data.summary ?? {} };
}

// ── Retrain API ───────────────────────────────────────────────────────────────

export type RetrainIdentity = {
  global_person_id: string;
  display_name: string;
  n_appearances: number;
  last_seen_at: string | null;
};

export type RetrainActionLabel = {
  label: string;
  samples: number;
};

export type RetrainDatasetStats = {
  sessions: number;
  persons: number;
  identities: RetrainIdentity[];
  hitl_human_labels: number;
  hitl_track_labels: number;
  total_hitl_samples: number;
  action_labels: {
    known: RetrainActionLabel[];
    new: RetrainActionLabel[];
  };
  base_class_names: string[];
  base_embeddings_available: { vjepa: boolean; dinov2: boolean };
  last_retrain_at: string | null;
  last_retrain_status: string | null;
  last_retrain_result: Record<string, unknown>;
};

export type RetrainJobState = {
  status: "idle" | "running" | "done" | "failed";
  started_at: string | null;
  finished_at: string | null;
  logs: string[];
  result: {
    val_accuracy?: number;
    n_samples?: number;
    n_classes?: number;
    class_names?: string[];
    n_hitl_samples?: number;
    output_dir?: string;
    backbone?: string;
  };
  error: string | null;
};

export async function fetchRetrainDatasetStats(): Promise<RetrainDatasetStats> {
  return harV2Fetch<RetrainDatasetStats>("/api/har/retrain/dataset-stats");
}

export async function fetchRetrainStatus(): Promise<RetrainJobState> {
  return harV2Fetch<RetrainJobState>("/api/har/retrain/status");
}

export async function triggerRetrain(backbone: "vjepa" | "dinov2", epochs: number): Promise<{ status: string; message?: string }> {
  return harV2Fetch("/api/har/retrain", {
    method: "POST",
    body: JSON.stringify({ backbone, epochs }),
  });
}

// ── Worker productivity summary ───────────────────────────────────────────────

export type PersonSummary = {
  global_person_id: string;
  display_name: string | null;
  n_events: number;
  n_idle_events: number;
  idle_pct: number;
  productive_pct: number;
  n_uncertain: number;
  n_low_conf_events: number;
  avg_confidence: number | null;
  avg_infer_ms: number | null;
  n_appearances: number;
  n_tracks: number;
  n_sessions: number;
  dominant_action: string | null;
  first_seen_at: string | null;
  last_seen_at: string | null;
  thumbnail_url: string | null;
  anomaly_count: number;
};

export async function fetchPersonsSummary(limit = 100): Promise<PersonSummary[]> {
  const data = await harV2Fetch<{ persons: PersonSummary[] }>(`/api/har/v2/persons/summary?limit=${limit}`);
  return data.persons ?? [];
}

export type PlantConfig = {
  siteName: string;
  lineCostPerMinute: number;
  shiftHours: number;
  materialCostPerUnit: number;
  targetCycleSec: number;
};

export async function fetchPlantConfig(): Promise<PlantConfig | null> {
  try {
    const res = await fetch(`${harV2Base()}/api/settings/plant`, { cache: "no-store" });
    if (!res.ok) return null;
    return res.json() as Promise<PlantConfig>;
  } catch {
    return null;
  }
}
