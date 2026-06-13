"use client";

import { useState } from "react";
import { Button } from "@/components/ui/Button";
import { Icon } from "@/components/ui/Icon";
import type { CameraCreateInput } from "@/lib/types";
import { cn } from "@/lib/cn";

const SOURCE_TYPES = [
  { id: "rtsp", label: "RTSP", icon: "settings_input_antenna" },
  { id: "onvif", label: "ONVIF", icon: "hub" },
  { id: "webcam", label: "Webcam", icon: "videocam" },
] as const;

const MODELS = [
  { id: "dinov3", label: "DINOv3", task: "patch_similarity" },
  { id: "vjepa2", label: "V-JEPA 2", task: "anomaly" },
  { id: "yolov8", label: "YOLOv8 + DeepSORT", task: "tracking" },
  { id: "none", label: "None", task: "" },
];

interface AddCameraModalProps {
  open: boolean;
  onClose: () => void;
  onSubmit: (input: CameraCreateInput) => Promise<boolean>;
}

export function AddCameraModal({ open, onClose, onSubmit }: AddCameraModalProps) {
  const [name, setName] = useState("");
  const [location, setLocation] = useState("");
  const [zone, setZone] = useState("");
  const [sourceType, setSourceType] = useState<CameraCreateInput["sourceType"]>("rtsp");
  const [streamUrl, setStreamUrl] = useState("");
  const [coords, setCoords] = useState("");
  const [model, setModel] = useState("dinov3");
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);

  if (!open) return null;

  const reset = () => {
    setName("");
    setLocation("");
    setZone("");
    setSourceType("rtsp");
    setStreamUrl("");
    setCoords("");
    setModel("dinov3");
    setError(null);
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!name.trim() || !location.trim()) {
      setError("Name and location are required.");
      return;
    }
    setSaving(true);
    setError(null);
    const selected = MODELS.find((m) => m.id === model);
    const ok = await onSubmit({
      name: name.trim(),
      location: location.trim(),
      zone: zone.trim() || undefined,
      sourceType,
      streamUrl: streamUrl.trim() || undefined,
      coords: coords.trim() || undefined,
      inferenceModel: model === "none" ? undefined : model,
      inferenceTask: selected?.task || undefined,
      backendCameraId: sourceType === "webcam" ? "webcam-0" : undefined,
    });
    setSaving(false);
    if (ok) {
      reset();
      onClose();
    } else {
      setError("Failed to save camera. Check the backend is running on :8000.");
    }
  };

  return (
    <div className="fixed inset-0 z-[100] flex items-center justify-center p-4">
      <button
        type="button"
        className="absolute inset-0 bg-on-surface/40 backdrop-blur-sm"
        onClick={onClose}
        aria-label="Close"
      />
      <div className="relative z-10 w-full max-w-lg rounded-card border border-outline-variant bg-surface-container-lowest shadow-overlay">
        <div className="flex items-center justify-between border-b border-outline-variant px-6 py-4">
          <div>
            <h2 className="font-headline text-headline-md text-on-surface">Add Camera Feed</h2>
            <p className="text-body-sm text-outline">RTSP · ONVIF · webcam</p>
          </div>
          <button
            type="button"
            onClick={onClose}
            className="rounded-lg p-2 text-outline hover:bg-surface-container-low"
          >
            <Icon name="close" size={20} />
          </button>
        </div>

        <form onSubmit={(e) => void handleSubmit(e)} className="space-y-5 px-6 py-5">
          <div className="grid gap-4 sm:grid-cols-2">
            <label className="block sm:col-span-2">
              <span className="mb-1.5 block font-label text-label-sm text-outline">Camera name</span>
              <input
                value={name}
                onChange={(e) => setName(e.target.value)}
                placeholder="Camera 04 — Packaging"
                className="w-full rounded-lg border border-outline-variant/70 bg-surface-container-low px-3 py-2 text-body-sm focus:border-primary focus:outline-none focus:ring-1 focus:ring-primary/30"
              />
            </label>
            <label className="block sm:col-span-2">
              <span className="mb-1.5 block font-label text-label-sm text-outline">Location</span>
              <input
                value={location}
                onChange={(e) => setLocation(e.target.value)}
                placeholder="Main Hall / Line 5"
                className="w-full rounded-lg border border-outline-variant/70 bg-surface-container-low px-3 py-2 text-body-sm focus:border-primary focus:outline-none focus:ring-1 focus:ring-primary/30"
              />
            </label>
            <label className="block">
              <span className="mb-1.5 block font-label text-label-sm text-outline">Zone</span>
              <input
                value={zone}
                onChange={(e) => setZone(e.target.value)}
                placeholder="ZONE C"
                className="w-full rounded-lg border border-outline-variant/70 bg-surface-container-low px-3 py-2 text-body-sm focus:border-primary focus:outline-none focus:ring-1 focus:ring-primary/30"
              />
            </label>
            <label className="block">
              <span className="mb-1.5 block font-label text-label-sm text-outline">Coordinates</span>
              <input
                value={coords}
                onChange={(e) => setCoords(e.target.value)}
                placeholder="42.3601°N · 71.0589°W"
                className="w-full rounded-lg border border-outline-variant/70 bg-surface-container-low px-3 py-2 text-body-sm focus:border-primary focus:outline-none focus:ring-1 focus:ring-primary/30"
              />
            </label>
          </div>

          <div>
            <span className="mb-2 block font-label text-label-sm text-outline">Source type</span>
            <div className="flex gap-2">
              {SOURCE_TYPES.map((src) => (
                <button
                  key={src.id}
                  type="button"
                  onClick={() => setSourceType(src.id)}
                  className={cn(
                    "flex flex-1 items-center justify-center gap-2 rounded-lg border px-3 py-2 text-body-sm transition-colors",
                    sourceType === src.id
                      ? "border-primary bg-primary-fixed/40 text-primary"
                      : "border-outline-variant/70 text-on-surface-variant hover:bg-surface-container-low",
                  )}
                >
                  <Icon name={src.icon} size={16} />
                  {src.label}
                </button>
              ))}
            </div>
          </div>

          {sourceType !== "webcam" && (
            <label className="block">
              <span className="mb-1.5 block font-label text-label-sm text-outline">Stream URL</span>
              <input
                value={streamUrl}
                onChange={(e) => setStreamUrl(e.target.value)}
                placeholder="rtsp://192.168.1.14:554/stream1"
                className="w-full rounded-lg border border-outline-variant/70 bg-surface-container-low px-3 py-2 font-label text-label-sm focus:border-primary focus:outline-none focus:ring-1 focus:ring-primary/30"
              />
            </label>
          )}

          <div>
            <span className="mb-2 block font-label text-label-sm text-outline">Inference model</span>
            <select
              value={model}
              onChange={(e) => setModel(e.target.value)}
              className="w-full rounded-lg border border-outline-variant/70 bg-surface-container-low px-3 py-2 text-body-sm focus:border-primary focus:outline-none"
            >
              {MODELS.map((m) => (
                <option key={m.id} value={m.id}>
                  {m.label}
                </option>
              ))}
            </select>
          </div>

          {error && <p className="text-body-sm text-error">{error}</p>}

          <div className="flex justify-end gap-3 border-t border-outline-variant/60 pt-4">
            <Button type="button" variant="ghost" onClick={onClose}>
              Cancel
            </Button>
            <Button type="submit" icon="add_a_photo" disabled={saving}>
              {saving ? "Saving…" : "Add Camera"}
            </Button>
          </div>
        </form>
      </div>
    </div>
  );
}
