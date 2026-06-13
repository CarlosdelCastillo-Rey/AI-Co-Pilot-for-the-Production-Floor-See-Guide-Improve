"use client";

import { useCallback, useEffect, useState } from "react";
import {
  Button,
  HarV2Badge,
  HarV2Empty,
  HarV2ListButton,
  HarV2Message,
  HarV2Panel,
  HarV2Table,
  HarV2TableCell,
  HarV2TableHead,
  HarV2TableRow,
  HarV2Th,
  fieldLabel,
  inputClass,
  selectClass,
  textareaClass,
} from "@/components/har-v2/har-v2-ui";
import { PersonCropThumb } from "@/components/har-v2/PersonCropThumb";
import {
  fetchHarV2ActionLabels,
  fetchHarV2Events,
  fetchHarV2ReviewQueue,
  fetchHarV2Sessions,
  fetchHarV2Tracks,
  harV2ArtifactUrl,
  saveHarV2HumanLabel,
  saveHarV2TrackLabel,
  type HarActionLabels,
  type HarAuditSession,
  type HarSessionEvent,
  type HarTrackSummary,
} from "@/lib/har-v2-api";

type Props = {
  onOpenRegistry?: (globalPersonId: string) => void;
  initialFocus?: ReviewFocus;
};

type ReviewFocus = "session" | "queue";

export function HarPersonHitlSessionsTab({ onOpenRegistry, initialFocus = "session" }: Props) {
  const [sessions, setSessions] = useState<HarAuditSession[]>([]);
  const [selectedId, setSelectedId] = useState<string>("");
  const [tracks, setTracks] = useState<HarTrackSummary[]>([]);
  const [trackIdx, setTrackIdx] = useState(0);
  const [queue, setQueue] = useState<HarSessionEvent[]>([]);
  const [queueIdx, setQueueIdx] = useState(0);
  const [focus, setFocus] = useState<ReviewFocus>(initialFocus);
  const [personVerdict, setPersonVerdict] = useState("yes");
  const [displayName, setDisplayName] = useState("");
  const [notes, setNotes] = useState("");
  const [actionVerdict, setActionVerdict] = useState("yes");
  const [queuePersonVerdict, setQueuePersonVerdict] = useState("yes");
  const [correctLabel, setCorrectLabel] = useState("");
  const [actionLabels, setActionLabels] = useState<HarActionLabels | null>(null);
  const [loading, setLoading] = useState(true);
  const [msg, setMsg] = useState("");
  const [msgError, setMsgError] = useState(false);

  const loadSessions = useCallback(async () => {
    const rows = await fetchHarV2Sessions(80);
    setSessions(rows);
    if (rows.length && !selectedId) setSelectedId(rows[0].session_id);
    return rows;
  }, [selectedId]);

  const loadQueue = useCallback(async () => {
    const rows = await fetchHarV2ReviewQueue(40);
    setQueue(rows);
    setQueueIdx((prev) => Math.min(prev, Math.max(0, rows.length - 1)));
    if (initialFocus === "queue" && rows.length > 0) {
      setFocus("queue");
      setQueueIdx(0);
    }
    return rows;
  }, [initialFocus]);

  const loadActionLabels = useCallback(async () => {
    const labels = await fetchHarV2ActionLabels();
    setActionLabels(labels);
    return labels;
  }, []);

  const load = useCallback(async () => {
    setLoading(true);
    try {
      await Promise.all([loadSessions(), loadQueue(), loadActionLabels()]);
    } catch (e) {
      setMsg(String(e));
      setMsgError(true);
    } finally {
      setLoading(false);
    }
  }, [loadSessions, loadQueue, loadActionLabels]);

  useEffect(() => {
    void load();
  }, [load]);

  useEffect(() => {
    if (!selectedId) return;
    void (async () => {
      try {
        const t = await fetchHarV2Tracks(selectedId);
        setTracks(t);
        setTrackIdx(0);
      } catch (e) {
        setMsg(String(e));
        setMsgError(true);
      }
    })();
  }, [selectedId]);

  const current = tracks[trackIdx];
  const queueItem = queue[queueIdx];

  useEffect(() => {
    setDisplayName(current?.display_name?.trim() ?? "");
  }, [current?.track_id, current?.display_name]);

  const cropUrl = harV2ArtifactUrl(current?.sample_crop_url);
  const queueCropUrl = harV2ArtifactUrl(queueItem?.crop_url);

  async function refreshTracks() {
    if (!selectedId) return;
    const t = await fetchHarV2Tracks(selectedId);
    setTracks(t);
    return t;
  }

  function selectSession(sessionId: string) {
    setFocus("session");
    setSelectedId(sessionId);
  }

  function selectQueueItem(index: number) {
    setFocus("queue");
    setQueueIdx(index);
  }

  async function saveTag() {
    if (!current || !selectedId) return;
    try {
      const result = await saveHarV2TrackLabel({
        session_id: selectedId,
        track_id: current.track_id,
        person_verdict: personVerdict,
        display_name: displayName.trim() || undefined,
        global_person_id: current.global_person_id ?? undefined,
        action_notes: notes.trim() || undefined,
        video: sessions.find((s) => s.session_id === selectedId)?.video_name ?? undefined,
        dominant_action: current.dominant_action,
        dominant_confidence: current.dominant_confidence,
        n_events: current.n_inferences,
      });
      const updated = await refreshTracks();
      const savedName = displayName.trim();
      if (!result.registry_updated && savedName) {
        setMsg(
          result.global_person_id
            ? `Saved label, but registry name may not have updated. Open the Registry tab to verify.`
            : `Saved label only — this track has no registry person id yet (wait for inference events with embeddings).`,
        );
        setMsgError(true);
      } else {
        setMsg(
          savedName ? `Saved “${savedName}” for track #${current.track_id}.` : "Track tag saved.",
        );
        setMsgError(false);
      }
      setNotes("");
      const nextIdx = Math.min(trackIdx + 1, (updated?.length ?? tracks.length) - 1);
      if (trackIdx < tracks.length - 1) setTrackIdx(nextIdx);
    } catch (e) {
      setMsg(String(e));
      setMsgError(true);
    }
  }

  useEffect(() => {
    setCorrectLabel("");
  }, [queueItem?.event_id]);

  async function saveQueueVerdict() {
    if (!queueItem) return;
    const trimmedLabel = correctLabel.trim();
    if (actionVerdict === "no" && !trimmedLabel) {
      setMsg("Pick or type the correct action label before saving.");
      setMsgError(true);
      return;
    }
    try {
      await saveHarV2HumanLabel({
        session_id: queueItem.session_id,
        event_id: queueItem.event_id,
        track_id: queueItem.track_id,
        person_verdict: queuePersonVerdict,
        action_verdict: actionVerdict,
        correct_label: actionVerdict === "no" ? trimmedLabel : undefined,
      });
      setMsg(
        actionVerdict === "no" && trimmedLabel
          ? `Saved “${trimmedLabel}” and expanded the action catalog.`
          : "Label saved for retraining export.",
      );
      setMsgError(false);
      setCorrectLabel("");
      await loadActionLabels();
      const rows = await loadQueue();
      if (queueIdx < rows.length - 1) {
        setQueueIdx(queueIdx + 1);
      } else if (rows.length > 0) {
        setQueueIdx(Math.min(queueIdx, rows.length - 1));
      } else {
        setFocus("session");
      }
    } catch (e) {
      setMsg(String(e));
      setMsgError(true);
    }
  }

  return (
    <div className="grid gap-6 lg:grid-cols-[minmax(260px,320px)_1fr]">
      <HarV2Panel title="Review worklist">
        {loading ? (
          <div className="h-32 animate-pulse rounded-md bg-surface-container-low" />
        ) : (
          <div className="max-h-[70vh] space-y-5 overflow-auto">
            <section>
              <h3 className="mb-2 font-label text-[10px] uppercase tracking-wide text-outline">
                Recorded sessions
              </h3>
              {sessions.length === 0 ? (
                <HarV2Empty>
                  No recorded sessions yet. Run Live Streams or Eval with session logging enabled.
                </HarV2Empty>
              ) : (
                <ul className="space-y-1">
                  {sessions.map((s) => (
                    <li key={s.session_id}>
                      <HarV2ListButton
                        active={focus === "session" && selectedId === s.session_id}
                        onClick={() => selectSession(s.session_id)}
                      >
                        <span className="block font-mono text-[11px]">{s.session_id}</span>
                        <span className="block text-on-surface">{s.video_name ?? s.source}</span>
                        <span className="block text-[11px] text-outline">
                          {s.n_events} events · {s.n_tracks} tracks
                        </span>
                      </HarV2ListButton>
                    </li>
                  ))}
                </ul>
              )}
            </section>

            <section>
              <h3 className="mb-2 font-label text-[10px] uppercase tracking-wide text-outline">
                Review queue ({queue.length})
              </h3>
              {queue.length === 0 ? (
                <p className="rounded-md border border-dashed border-outline-variant/60 px-3 py-4 text-center text-[11px] text-outline">
                  No uncertain crops right now.
                </p>
              ) : (
                <ul className="space-y-1">
                  {queue.map((item, i) => (
                    <li key={item.event_id}>
                      <HarV2ListButton active={focus === "queue" && queueIdx === i} onClick={() => selectQueueItem(i)}>
                        <div className="flex items-center gap-2">
                          <PersonCropThumb url={item.crop_url} size="sm" />
                          <div className="min-w-0 flex-1">
                            <span className="block text-on-surface">
                              Track #{item.track_id} · f{item.frame_idx ?? "—"}
                            </span>
                            <span className="block text-[11px] text-outline">
                              {item.raw_label ?? item.action_label ?? "—"}
                              {item.confidence != null ? ` · ${(item.confidence * 100).toFixed(0)}%` : ""}
                            </span>
                          </div>
                        </div>
                      </HarV2ListButton>
                    </li>
                  ))}
                </ul>
              )}
            </section>
          </div>
        )}
      </HarV2Panel>

      <div className="space-y-4">
        {focus === "queue" && queueItem ? (
          <div className="grid gap-6 lg:grid-cols-2">
            <HarV2Panel title={`Queue item ${queueIdx + 1} of ${queue.length}`}>
              <p className="mb-3 text-body-sm text-outline">
                Track #{queueItem.track_id} · frame {queueItem.frame_idx}
                {queueItem.global_person_id ? ` · ${queueItem.global_person_id}` : ""}
              </p>
              {queueCropUrl ? (
                // eslint-disable-next-line @next/next/no-img-element
                <img
                  src={queueCropUrl}
                  alt="Review crop"
                  className="max-h-96 w-full rounded-md border border-outline-variant/60 object-contain bg-surface-container-low"
                />
              ) : (
                <HarV2Empty>No crop image for this event.</HarV2Empty>
              )}
              <p className="mt-3 text-body-sm text-on-surface">
                Predicted:{" "}
                <strong>{queueItem.raw_label ?? queueItem.action_label ?? "—"}</strong>
                {queueItem.confidence != null && (
                  <span className="text-outline"> ({(queueItem.confidence * 100).toFixed(0)}%)</span>
                )}
              </p>
            </HarV2Panel>

            <HarV2Panel title="Crop verdict">
              <div className="space-y-4">
                <div>
                  <label className={fieldLabel}>Action correct?</label>
                  <select
                    value={actionVerdict}
                    onChange={(e) => setActionVerdict(e.target.value)}
                    className={selectClass}
                  >
                    <option value="yes">Yes</option>
                    <option value="no">No — wrong action</option>
                    <option value="dont_know">Don&apos;t know</option>
                  </select>
                </div>
                <div>
                  <label className={fieldLabel}>Real person?</label>
                  <select
                    value={queuePersonVerdict}
                    onChange={(e) => setQueuePersonVerdict(e.target.value)}
                    className={selectClass}
                  >
                    <option value="yes">Yes</option>
                    <option value="no">No</option>
                    <option value="unknown">Unknown</option>
                  </select>
                </div>
                <div>
                  <label className={fieldLabel}>Correct action</label>
                  {actionVerdict === "no" ? (
                    <>
                      <input
                        value={correctLabel}
                        onChange={(e) => setCorrectLabel(e.target.value)}
                        className={inputClass}
                        list="har-action-label-options"
                        placeholder="Pick a model class or type a new one (e.g. WALKING)"
                      />
                      <datalist id="har-action-label-options">
                        {(actionLabels?.all_labels ?? []).map((label) => (
                          <option key={label} value={label} />
                        ))}
                      </datalist>
                      <p className="mt-1 text-[11px] text-outline">
                        New labels are saved to the catalog for future reviews and retraining export.
                        {actionLabels?.custom_labels.length ? (
                          <>
                            {" "}
                            Custom: {actionLabels.custom_labels.join(", ")}
                          </>
                        ) : null}
                      </p>
                    </>
                  ) : (
                    <p className="rounded-md border border-outline-variant/50 bg-surface-container-low px-3 py-2 text-body-sm text-outline">
                      Only needed when the predicted action is wrong.
                    </p>
                  )}
                </div>
                <div className="flex flex-wrap gap-2 pt-2">
                  <Button variant="outline" type="button" onClick={() => setQueueIdx(Math.max(0, queueIdx - 1))}>
                    Prev
                  </Button>
                  <Button type="button" onClick={() => void saveQueueVerdict()}>
                    Save & next
                  </Button>
                </div>
              </div>
            </HarV2Panel>
          </div>
        ) : focus === "session" && current ? (
          <>
            <HarV2Panel
              title={
                current.display_name?.trim()
                  ? `${current.display_name} · track #${current.track_id} (${trackIdx + 1}/${tracks.length})`
                  : `Track #${current.track_id} (${trackIdx + 1}/${tracks.length})`
              }
            >
              <div className="mb-4 flex flex-wrap items-center gap-2">
                {current.display_name?.trim() ? (
                  <HarV2Badge>{current.display_name.trim()}</HarV2Badge>
                ) : null}
                {current.global_person_id ? (
                  <button
                    type="button"
                    className="cursor-pointer"
                    onClick={() => onOpenRegistry?.(current.global_person_id!)}
                    title="Open in registry"
                  >
                    <HarV2Badge>{current.global_person_id}</HarV2Badge>
                  </button>
                ) : (
                  <span className="text-[11px] text-outline">No registry id yet</span>
                )}
              </div>
              <div className="grid gap-6 md:grid-cols-2">
                <div>
                  {cropUrl ? (
                    // eslint-disable-next-line @next/next/no-img-element
                    <img
                      src={cropUrl}
                      alt="Person crop"
                      className="max-h-80 w-full rounded-md border border-outline-variant/60 object-contain bg-surface-container-low"
                    />
                  ) : (
                    <HarV2Empty>No crop saved for this track.</HarV2Empty>
                  )}
                </div>
                <div className="space-y-3 text-body-sm text-on-surface-variant">
                  <p>
                    Frames {current.frame_first}–{current.frame_last} · {current.n_inferences} inferences
                  </p>
                  <p>
                    Dominant action:{" "}
                    <strong className="text-on-surface">{current.dominant_action || "—"}</strong>
                    {current.dominant_confidence != null &&
                      ` (${(current.dominant_confidence * 100).toFixed(0)}%)`}
                  </p>
                  <div>
                    <label className={fieldLabel}>Real person?</label>
                    <select
                      value={personVerdict}
                      onChange={(e) => setPersonVerdict(e.target.value)}
                      className={selectClass}
                    >
                      <option value="yes">Yes — real person</option>
                      <option value="no">No — false detection</option>
                      <option value="unknown">Unsure</option>
                    </select>
                  </div>
                  <div>
                    <label className={fieldLabel}>Display name (registry)</label>
                    <input
                      value={displayName}
                      onChange={(e) => setDisplayName(e.target.value)}
                      placeholder="Worker name or ID"
                      className={inputClass}
                    />
                  </div>
                  <div>
                    <label className={fieldLabel}>Notes</label>
                    <textarea
                      value={notes}
                      onChange={(e) => setNotes(e.target.value)}
                      placeholder="Optional notes"
                      rows={2}
                      className={textareaClass}
                    />
                  </div>
                  <div className="flex flex-wrap gap-2 pt-1">
                    <Button variant="outline" type="button" onClick={() => setTrackIdx(Math.max(0, trackIdx - 1))}>
                      Prev
                    </Button>
                    <Button type="button" onClick={() => void saveTag()}>
                      Save tag
                    </Button>
                    <Button
                      variant="secondary"
                      type="button"
                      onClick={() => setTrackIdx(Math.min(tracks.length - 1, trackIdx + 1))}
                    >
                      Skip
                    </Button>
                  </div>
                </div>
              </div>
            </HarV2Panel>
            <TrackEventsPanel sessionId={selectedId} trackId={current.track_id} />
          </>
        ) : (
          <HarV2Empty>
            {queue.length > 0
              ? "Select a session track or a review-queue crop from the worklist."
              : "Select a session with recorded inference events."}
          </HarV2Empty>
        )}
        {msg ? <HarV2Message tone={msgError ? "error" : "info"}>{msg}</HarV2Message> : null}
      </div>
    </div>
  );
}

function TrackEventsPanel({ sessionId, trackId }: { sessionId: string; trackId: number }) {
  const [events, setEvents] = useState<Awaited<ReturnType<typeof fetchHarV2Events>>>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    setLoading(true);
    void fetchHarV2Events(sessionId, trackId)
      .then(setEvents)
      .catch(() => setEvents([]))
      .finally(() => setLoading(false));
  }, [sessionId, trackId]);

  return (
    <HarV2Panel title="Inference log">
      {loading ? (
        <div className="h-24 animate-pulse rounded-md bg-surface-container-low" />
      ) : events.length === 0 ? (
        <HarV2Empty>No events for this track.</HarV2Empty>
      ) : (
        <HarV2Table>
          <HarV2TableHead>
            <HarV2Th>Frame</HarV2Th>
            <HarV2Th>Action</HarV2Th>
            <HarV2Th>Conf</HarV2Th>
            <HarV2Th>Changed</HarV2Th>
          </HarV2TableHead>
          <tbody>
            {events.map((ev) => (
              <HarV2TableRow key={ev.event_id}>
                <HarV2TableCell>{ev.frame_idx}</HarV2TableCell>
                <HarV2TableCell>{ev.action_label ?? ev.raw_label ?? "—"}</HarV2TableCell>
                <HarV2TableCell>
                  {ev.confidence != null ? `${(ev.confidence * 100).toFixed(0)}%` : "—"}
                </HarV2TableCell>
                <HarV2TableCell>{ev.label_changed ? "✓" : ""}</HarV2TableCell>
              </HarV2TableRow>
            ))}
          </tbody>
        </HarV2Table>
      )}
    </HarV2Panel>
  );
}
