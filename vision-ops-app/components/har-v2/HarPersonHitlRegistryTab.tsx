"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
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
} from "@/components/har-v2/har-v2-ui";
import { PersonCropThumb } from "@/components/har-v2/PersonCropThumb";
import {
  downloadHarV2PersonPdf,
  fetchHarV2PersonReport,
  fetchHarV2Persons,
  harV2ArtifactUrl,
  mergeHarV2Persons,
  saveHarV2PersonName,
  splitHarV2TrackIdentity,
  type HarPerson,
  type HarPersonReport,
  type HarPersonSessionRow,
  type HarPersonSnapshot,
} from "@/lib/har-v2-api";

type Props = {
  initialPersonId?: string;
};

function pct(value: number | null | undefined): string {
  if (value == null) return "—";
  return `${(value * 100).toFixed(0)}%`;
}

function MetricCard({ label, value, hint }: { label: string; value: string; hint?: string }) {
  return (
    <div className="rounded-md border border-outline-variant/50 bg-surface-container-low px-3 py-2">
      <p className="font-label text-[10px] uppercase tracking-wide text-outline">{label}</p>
      <p className="mt-1 font-medium text-on-surface">{value}</p>
      {hint ? <p className="mt-0.5 text-[11px] text-outline">{hint}</p> : null}
    </div>
  );
}

export function HarPersonHitlRegistryTab({ initialPersonId }: Props) {
  const [persons, setPersons] = useState<HarPerson[]>([]);
  const [selected, setSelected] = useState<string>("");
  const [report, setReport] = useState<HarPersonReport | null>(null);
  const [selectedSnapshot, setSelectedSnapshot] = useState<HarPersonSnapshot | null>(null);
  const [name, setName] = useState("");
  const [msg, setMsg] = useState("");
  const [msgError, setMsgError] = useState(false);
  const [mergeSources, setMergeSources] = useState<Set<string>>(new Set());
  const [pdfLoading, setPdfLoading] = useState(false);

  const loadPersons = useCallback(async () => {
    const rows = await fetchHarV2Persons(100);
    setPersons(rows);
    return rows;
  }, []);

  const loadReport = useCallback(async (personId: string) => {
    const data = await fetchHarV2PersonReport(personId);
    setReport(data);
    setName(data.display_name ?? "");
    setSelectedSnapshot(data.snapshots[0] ?? null);
    return data;
  }, []);

  useEffect(() => {
    void loadPersons()
      .then((rows) => {
        if (initialPersonId && rows.some((p) => p.global_person_id === initialPersonId)) {
          setSelected(initialPersonId);
        } else if (rows.length && !selected) {
          setSelected(rows[0].global_person_id);
        }
      })
      .catch((e) => {
        setMsg(String(e));
        setMsgError(true);
      });
  }, [loadPersons, initialPersonId, selected]);

  useEffect(() => {
    if (initialPersonId) setSelected(initialPersonId);
  }, [initialPersonId]);

  useEffect(() => {
    if (!selected) {
      setReport(null);
      setSelectedSnapshot(null);
      return;
    }
    void loadReport(selected).catch((e) => {
      setMsg(String(e));
      setMsgError(true);
    });
    setMergeSources((prev) => {
      const next = new Set(prev);
      next.delete(selected);
      return next;
    });
  }, [selected, loadReport]);

  function toggleMergeSource(personId: string) {
    setMergeSources((prev) => {
      const next = new Set(prev);
      if (next.has(personId)) next.delete(personId);
      else next.add(personId);
      return next;
    });
  }

  async function mergeIntoSelected() {
    if (!selected || mergeSources.size === 0) return;
    try {
      const result = await mergeHarV2Persons(selected, [...mergeSources]);
      setMsg(
        `Merged ${result.merged_global_person_ids.length} duplicate${
          result.merged_global_person_ids.length === 1 ? "" : "s"
        } into ${selected}.`,
      );
      setMsgError(false);
      setMergeSources(new Set());
      await loadPersons();
      await loadReport(selected);
    } catch (e) {
      setMsg(String(e));
      setMsgError(true);
    }
  }

  async function saveName() {
    if (!selected) return;
    try {
      await saveHarV2PersonName(selected, name);
      setMsg("Display name saved.");
      setMsgError(false);
      const rows = await loadPersons();
      if (selected) await loadReport(selected);
      if (!rows.some((p) => p.global_person_id === selected) && rows.length) {
        setSelected(rows[0].global_person_id);
      }
    } catch (e) {
      setMsg(String(e));
      setMsgError(true);
    }
  }

  async function downloadPdf() {
    if (!selected) return;
    setPdfLoading(true);
    try {
      await downloadHarV2PersonPdf(selected);
    } catch (e) {
      setMsg(String(e));
      setMsgError(true);
    } finally {
      setPdfLoading(false);
    }
  }

  async function unlinkTrack(row: HarPersonSessionRow) {
    if (!report || (report.sessions?.length ?? 0) <= 1) return;
    try {
      const result = await splitHarV2TrackIdentity(row.session_id, row.track_id);
      setMsg(`Track #${row.track_id} moved to ${result.new_global_person_id}.`);
      setMsgError(false);
      const rows = await loadPersons();
      if (rows.some((p) => p.global_person_id === selected)) {
        await loadReport(selected);
      } else if (rows.length) {
        setSelected(rows[0].global_person_id);
      } else {
        setSelected("");
        setReport(null);
      }
    } catch (e) {
      setMsg(String(e));
      setMsgError(true);
    }
  }

  const sessions = (report?.sessions ?? []) as HarPersonSessionRow[];
  const multiTrack = sessions.length > 1;
  const selectedPerson = persons.find((p) => p.global_person_id === selected);
  const mergeCount = mergeSources.size;
  const metrics = report?.metrics;
  const snapshotPreviewUrl = useMemo(
    () => harV2ArtifactUrl(selectedSnapshot?.frame_url ?? selectedSnapshot?.crop_url),
    [selectedSnapshot],
  );

  return (
    <div className="grid gap-6 xl:grid-cols-[minmax(260px,300px)_1fr]">
      <HarV2Panel title="Known persons">
        {persons.length === 0 ? (
          <HarV2Empty>
            No persons registered yet. Tag tracks or confirm crops in the Review tab.
          </HarV2Empty>
        ) : (
          <>
            {selected && mergeCount > 0 ? (
              <div className="mb-3 space-y-2">
                <HarV2Message tone="info">
                  Merge {mergeCount} duplicate{mergeCount === 1 ? "" : "s"} into{" "}
                  {selectedPerson?.display_name || selected}.
                </HarV2Message>
                <Button type="button" onClick={() => void mergeIntoSelected()}>
                  Merge into selected
                </Button>
              </div>
            ) : selected && persons.length > 1 ? (
              <p className="mb-3 text-[11px] text-outline">
                Check duplicates below, then merge them into the selected person.
              </p>
            ) : null}
            <ul className="max-h-[70vh] space-y-1 overflow-auto">
              {persons.map((p) => {
                const isSelected = selected === p.global_person_id;
                const canMerge = selected && !isSelected;
                return (
                  <li key={p.global_person_id}>
                    <div className="flex items-start gap-2">
                      {canMerge ? (
                        <label className="mt-2 flex shrink-0 items-center">
                          <input
                            type="checkbox"
                            checked={mergeSources.has(p.global_person_id)}
                            onChange={() => toggleMergeSource(p.global_person_id)}
                            className="h-4 w-4 rounded border-outline-variant/60"
                            aria-label={`Merge ${p.display_name || p.global_person_id} into selected`}
                          />
                        </label>
                      ) : (
                        <span className="w-4 shrink-0" aria-hidden />
                      )}
                      <div className="min-w-0 flex-1">
                        <HarV2ListButton
                          active={isSelected}
                          onClick={() => setSelected(p.global_person_id)}
                        >
                          <div className="flex items-center gap-3">
                            <PersonCropThumb url={p.thumbnail_url} size="sm" />
                            <div className="min-w-0 flex-1">
                              <span className="block font-medium text-on-surface">
                                {p.display_name || "(unnamed)"}
                              </span>
                              <span className="block font-mono text-[11px] text-outline">{p.global_person_id}</span>
                              <span className="block text-[11px] text-outline">
                                {p.n_tracks ?? 1} track{(p.n_tracks ?? 1) === 1 ? "" : "s"} · {p.n_appearances}{" "}
                                events
                              </span>
                            </div>
                          </div>
                        </HarV2ListButton>
                      </div>
                    </div>
                  </li>
                );
              })}
            </ul>
          </>
        )}
      </HarV2Panel>

      <div className="space-y-4">
        {report && metrics ? (
          <>
            <HarV2Panel title="Person report">
              <div className="flex flex-wrap gap-4">
                <PersonCropThumb url={report.thumbnail_url} size="lg" />
                <div className="min-w-0 flex-1">
                  <HarV2Badge>{report.global_person_id}</HarV2Badge>
                  <p className="mt-3 text-body-sm text-on-surface">
                    {report.display_name?.trim() || "(unnamed worker)"}
                  </p>
                  <p className="mt-1 text-body-sm text-outline">
                    Dominant action: <strong className="text-on-surface">{metrics.dominant_action || "—"}</strong>
                    {" · "}
                    {metrics.first_seen_at?.slice(0, 19).replace("T", " ") ?? "—"} →{" "}
                    {metrics.last_seen_at?.slice(0, 19).replace("T", " ") ?? "—"}
                  </p>
                  {multiTrack ? (
                    <HarV2Message tone="info">
                      This identity spans multiple tracks. Use Unlink below if they are different workers.
                    </HarV2Message>
                  ) : null}
                  <div className="mt-4 flex flex-wrap gap-2">
                    <div className="min-w-[200px] flex-1">
                      <label className={fieldLabel}>Display name</label>
                      <input
                        value={name}
                        onChange={(e) => setName(e.target.value)}
                        className={inputClass}
                        placeholder="Worker name or ID"
                      />
                    </div>
                    <div className="flex items-end gap-2">
                      <Button type="button" onClick={() => void saveName()}>
                        Save name
                      </Button>
                      <Button
                        type="button"
                        variant="outline"
                        onClick={() => void downloadPdf()}
                        disabled={pdfLoading}
                      >
                        {pdfLoading ? "Generating..." : "Download Report"}
                      </Button>
                    </div>
                  </div>
                </div>
              </div>

              <h3 className="mb-2 mt-6 font-label text-[10px] uppercase tracking-wide text-outline">Metrics</h3>
              <div className="grid gap-2 sm:grid-cols-2 lg:grid-cols-4">
                <MetricCard label="Events" value={String(metrics.n_events)} />
                <MetricCard label="Tracks" value={String(metrics.n_tracks)} hint={`${metrics.n_sessions} sessions`} />
                <MetricCard
                  label="Videos"
                  value={String(metrics.n_videos)}
                  hint={metrics.videos.slice(0, 2).join(", ") || undefined}
                />
                <MetricCard
                  label="Frame span"
                  value={
                    metrics.frame_first != null && metrics.frame_last != null
                      ? `${metrics.frame_first}–${metrics.frame_last}`
                      : "—"
                  }
                />
                <MetricCard
                  label="Avg confidence"
                  value={pct(metrics.avg_confidence)}
                  hint={`${pct(metrics.min_confidence)} – ${pct(metrics.max_confidence)}`}
                />
                <MetricCard
                  label="Avg Re-ID match"
                  value={pct(metrics.avg_reid_match)}
                  hint={`${pct(metrics.min_reid_match)} – ${pct(metrics.max_reid_match)}`}
                />
                <MetricCard label="Uncertain" value={String(metrics.n_uncertain)} />
                <MetricCard label="Label changes" value={String(metrics.n_label_changes)} />
              </div>

              <h3 className="mb-2 mt-6 font-label text-[10px] uppercase tracking-wide text-outline">Action mix</h3>
              {report.action_breakdown.length === 0 ? (
                <HarV2Empty>No action predictions recorded yet.</HarV2Empty>
              ) : (
                <div className="space-y-2">
                  {report.action_breakdown.map((row) => (
                    <div key={row.action} className="grid grid-cols-[minmax(140px,1fr)_minmax(120px,180px)_auto] items-center gap-3 text-body-sm">
                      <span className="truncate text-on-surface">{row.action}</span>
                      <div className="h-2 overflow-hidden rounded-full bg-surface-container-high">
                        <div
                          className="h-full rounded-full bg-primary"
                          style={{ width: `${Math.max(row.share * 100, 2)}%` }}
                        />
                      </div>
                      <span className="whitespace-nowrap text-[11px] text-outline">
                        {row.count} · {(row.share * 100).toFixed(1)}%
                        {row.avg_confidence != null ? ` · ${pct(row.avg_confidence)}` : ""}
                      </span>
                    </div>
                  ))}
                </div>
              )}
            </HarV2Panel>

            <HarV2Panel
              title={`Snapshots${report.snapshots_truncated ? ` (showing ${report.snapshots_shown})` : ` (${report.snapshots_shown})`}`}
            >
              {report.snapshots.length === 0 ? (
                <HarV2Empty>No crop snapshots saved for this person.</HarV2Empty>
              ) : (
                <div className="grid gap-4 lg:grid-cols-[minmax(0,1fr)_220px]">
                  <div className="grid grid-cols-3 gap-2 sm:grid-cols-4 md:grid-cols-5 lg:grid-cols-6">
                    {report.snapshots.map((snap) => (
                      <button
                        key={snap.event_id}
                        type="button"
                        onClick={() => setSelectedSnapshot(snap)}
                        className={
                          selectedSnapshot?.event_id === snap.event_id
                            ? "rounded-md ring-2 ring-primary"
                            : "rounded-md ring-1 ring-outline-variant/40"
                        }
                      >
                        <PersonCropThumb url={snap.crop_url} size="md" />
                        <span className="mt-1 block truncate px-1 text-[10px] text-outline">
                          f{snap.frame_idx ?? "—"} · {snap.action_label ?? "—"}
                        </span>
                      </button>
                    ))}
                  </div>
                  {selectedSnapshot ? (
                    <div className="rounded-md border border-outline-variant/50 bg-surface-container-low p-3">
                      <p className="mb-2 text-[11px] text-outline">
                        Track #{selectedSnapshot.track_id} · f{selectedSnapshot.frame_idx ?? "—"}
                      </p>
                      {snapshotPreviewUrl ? (
                        // eslint-disable-next-line @next/next/no-img-element
                        <img
                          src={snapshotPreviewUrl}
                          alt="Selected snapshot"
                          className="max-h-56 w-full rounded-md object-contain"
                        />
                      ) : (
                        <HarV2Empty>No preview image.</HarV2Empty>
                      )}
                      <p className="mt-2 text-body-sm text-on-surface">
                        {selectedSnapshot.action_label ?? "—"}
                        {selectedSnapshot.confidence != null ? ` · ${pct(selectedSnapshot.confidence)}` : ""}
                      </p>
                      <p className="mt-1 text-[11px] text-outline">
                        {selectedSnapshot.uncertain ? "Uncertain · " : ""}
                        {selectedSnapshot.label_changed ? "Label changed · " : ""}
                        {selectedSnapshot.occurred_at?.slice(0, 19).replace("T", " ") ?? ""}
                      </p>
                    </div>
                  ) : null}
                </div>
              )}
            </HarV2Panel>

            <HarV2Panel title="Sessions & tracks">
              <HarV2Table>
                <HarV2TableHead>
                  <HarV2Th>Preview</HarV2Th>
                  <HarV2Th>Session</HarV2Th>
                  <HarV2Th>Track</HarV2Th>
                  <HarV2Th>Video</HarV2Th>
                  <HarV2Th>Events</HarV2Th>
                  <HarV2Th>Frames</HarV2Th>
                  <HarV2Th>Top action</HarV2Th>
                  <HarV2Th>Match</HarV2Th>
                  {multiTrack ? <HarV2Th>Unlink</HarV2Th> : null}
                </HarV2TableHead>
                <tbody>
                  {sessions.map((r) => (
                    <HarV2TableRow key={`${r.session_id}-${r.track_id}`}>
                      <HarV2TableCell>
                        <PersonCropThumb url={r.sample_crop_url} size="sm" />
                      </HarV2TableCell>
                      <HarV2TableCell className="font-mono text-[11px]">
                        {String(r.session_id).slice(0, 14)}…
                      </HarV2TableCell>
                      <HarV2TableCell>#{String(r.track_id)}</HarV2TableCell>
                      <HarV2TableCell>{String(r.video ?? "")}</HarV2TableCell>
                      <HarV2TableCell>{r.n_appearances ?? "—"}</HarV2TableCell>
                      <HarV2TableCell>
                        {r.frame_first != null && r.frame_last != null ? `${r.frame_first}–${r.frame_last}` : "—"}
                      </HarV2TableCell>
                      <HarV2TableCell>{String(r.dominant_action ?? "—")}</HarV2TableCell>
                      <HarV2TableCell className="text-[11px] text-outline">
                        {r.match_score != null ? `${(r.match_score * 100).toFixed(0)}%` : "—"}
                      </HarV2TableCell>
                      {multiTrack ? (
                        <HarV2TableCell>
                          <Button
                            variant="outline"
                            type="button"
                            className="text-xs"
                            onClick={() => void unlinkTrack(r)}
                          >
                            Unlink
                          </Button>
                        </HarV2TableCell>
                      ) : null}
                    </HarV2TableRow>
                  ))}
                </tbody>
              </HarV2Table>
            </HarV2Panel>

            {report.track_labels.length > 0 ? (
              <HarV2Panel title="Supervisor track tags">
                <HarV2Table>
                  <HarV2TableHead>
                    <HarV2Th>Session</HarV2Th>
                    <HarV2Th>Track</HarV2Th>
                    <HarV2Th>Name</HarV2Th>
                    <HarV2Th>Verdict</HarV2Th>
                    <HarV2Th>Action</HarV2Th>
                    <HarV2Th>Reviewed</HarV2Th>
                  </HarV2TableHead>
                  <tbody>
                    {report.track_labels.map((row, idx) => (
                      <HarV2TableRow key={`${String(row.session_id)}-${String(row.track_id)}-${idx}`}>
                        <HarV2TableCell className="font-mono text-[11px]">
                          {String(row.session_id).slice(0, 14)}…
                        </HarV2TableCell>
                        <HarV2TableCell>#{String(row.track_id)}</HarV2TableCell>
                        <HarV2TableCell>{String(row.display_name ?? "—")}</HarV2TableCell>
                        <HarV2TableCell>{String(row.person_verdict ?? "—")}</HarV2TableCell>
                        <HarV2TableCell>{String(row.dominant_action ?? "—")}</HarV2TableCell>
                        <HarV2TableCell className="text-[11px] text-outline">
                          {String(row.reviewed_at ?? "").slice(0, 19).replace("T", " ")}
                        </HarV2TableCell>
                      </HarV2TableRow>
                    ))}
                  </tbody>
                </HarV2Table>
              </HarV2Panel>
            ) : null}

            {report.human_labels.length > 0 ? (
              <HarV2Panel title="HITL crop reviews">
                <HarV2Table>
                  <HarV2TableHead>
                    <HarV2Th>Event</HarV2Th>
                    <HarV2Th>Predicted</HarV2Th>
                    <HarV2Th>Action verdict</HarV2Th>
                    <HarV2Th>Correct label</HarV2Th>
                    <HarV2Th>Person</HarV2Th>
                    <HarV2Th>Training</HarV2Th>
                  </HarV2TableHead>
                  <tbody>
                    {report.human_labels.map((row) => (
                      <HarV2TableRow key={String(row.label_id)}>
                        <HarV2TableCell className="font-mono text-[11px]">
                          {String(row.event_id ?? "").slice(0, 12)}…
                        </HarV2TableCell>
                        <HarV2TableCell>{String(row.predicted_label ?? "—")}</HarV2TableCell>
                        <HarV2TableCell>{String(row.action_verdict ?? "—")}</HarV2TableCell>
                        <HarV2TableCell>{String(row.correct_label ?? "—")}</HarV2TableCell>
                        <HarV2TableCell>{String(row.person_verdict ?? "—")}</HarV2TableCell>
                        <HarV2TableCell>{row.usable_for_training ? "Yes" : "No"}</HarV2TableCell>
                      </HarV2TableRow>
                    ))}
                  </tbody>
                </HarV2Table>
              </HarV2Panel>
            ) : null}

            <HarV2Panel
              title={`Event log${report.events_truncated ? ` (showing ${report.events_shown} of ${metrics.n_events})` : ` (${metrics.n_events})`}`}
            >
              {report.events.length === 0 ? (
                <HarV2Empty>No inference events linked to this person yet.</HarV2Empty>
              ) : (
                <div className="max-h-[60vh] overflow-auto">
                  <HarV2Table>
                    <HarV2TableHead>
                      <HarV2Th>Crop</HarV2Th>
                      <HarV2Th>Frame</HarV2Th>
                      <HarV2Th>Time</HarV2Th>
                      <HarV2Th>Session</HarV2Th>
                      <HarV2Th>Track</HarV2Th>
                      <HarV2Th>Action</HarV2Th>
                      <HarV2Th>Conf.</HarV2Th>
                      <HarV2Th>Re-ID</HarV2Th>
                      <HarV2Th>Flags</HarV2Th>
                    </HarV2TableHead>
                    <tbody>
                      {report.events.map((ev) => (
                        <HarV2TableRow key={ev.event_id}>
                          <HarV2TableCell>
                            <PersonCropThumb url={ev.crop_url} size="sm" />
                          </HarV2TableCell>
                          <HarV2TableCell>{ev.frame_idx ?? "—"}</HarV2TableCell>
                          <HarV2TableCell className="whitespace-nowrap text-[11px] text-outline">
                            {ev.occurred_at?.slice(0, 19).replace("T", " ") ?? "—"}
                          </HarV2TableCell>
                          <HarV2TableCell className="font-mono text-[11px]">
                            {String(ev.session_id).slice(0, 14)}…
                          </HarV2TableCell>
                          <HarV2TableCell>#{String(ev.track_id)}</HarV2TableCell>
                          <HarV2TableCell>{String(ev.action_label ?? ev.raw_label ?? "—")}</HarV2TableCell>
                          <HarV2TableCell className="text-[11px] text-outline">{pct(ev.confidence)}</HarV2TableCell>
                          <HarV2TableCell className="text-[11px] text-outline">
                            {pct(ev.reid_match_score)}
                          </HarV2TableCell>
                          <HarV2TableCell className="text-[11px] text-outline">
                            {ev.uncertain ? "uncertain " : ""}
                            {ev.label_changed ? "changed" : ""}
                          </HarV2TableCell>
                        </HarV2TableRow>
                      ))}
                    </tbody>
                  </HarV2Table>
                </div>
              )}
            </HarV2Panel>
          </>
        ) : (
          <HarV2Panel title="Person report">
            <HarV2Empty>Select a person from the list.</HarV2Empty>
          </HarV2Panel>
        )}
        {msg ? <HarV2Message tone={msgError ? "error" : "info"}>{msg}</HarV2Message> : null}
      </div>
    </div>
  );
}
