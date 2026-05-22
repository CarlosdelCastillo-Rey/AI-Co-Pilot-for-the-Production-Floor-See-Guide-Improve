"use client";

import { useCallback, useEffect, useState } from "react";
import Link from "next/link";
import { CameraStream } from "@/components/live/CameraStream";
import { Button } from "@/components/ui/Button";
import { Card } from "@/components/ui/Card";
import {
  deleteFaceEnrollment,
  enrollFace,
  fetchFaceStatus,
  fetchFaceStorage,
  type FaceStatus,
  type FaceStorageInfo,
} from "@/lib/api";

export function IdentityEnrollmentPanel() {
  const [name, setName] = useState("You");
  const [status, setStatus] = useState<FaceStatus | null>(null);
  const [storage, setStorage] = useState<FaceStorageInfo | null>(null);
  const [message, setMessage] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  const refresh = useCallback(async () => {
    const [s, st] = await Promise.all([fetchFaceStatus(), fetchFaceStorage()]);
    setStatus(s);
    setStorage(st);
    if (s?.enrolled && s.name) {
      setName(s.name);
    }
  }, []);

  useEffect(() => {
    refresh();
    const id = setInterval(refresh, 5000);
    return () => clearInterval(id);
  }, [refresh]);

  async function handleEnroll() {
    if (!name.trim()) {
      setMessage("Enter a display name.");
      return;
    }
    setLoading(true);
    setMessage(null);
    const result = await enrollFace(name);
    setMessage(result.message);
    setLoading(false);
    await refresh();
  }

  async function handleDelete() {
    setLoading(true);
    setMessage(null);
    const result = await deleteFaceEnrollment();
    setMessage(result.message);
    setLoading(false);
    await refresh();
  }

  const streamUrl = "/vision-api/api/cameras/webcam-0/stream";
  const previewUrl = status?.has_preview ? "/vision-api/api/faces/preview" : null;

  return (
    <div className="mx-auto grid max-w-[1200px] gap-gutter lg:grid-cols-2">
      <div className="flex flex-col gap-gutter">
        <Card className="p-lg">
          <h2 className="mb-md font-headline text-headline-sm text-on-surface">
            Register your identity
          </h2>
          <p className="mb-lg text-body-sm text-outline">
            Enter the name shown on the green box in{" "}
            <Link href="/live" className="text-primary underline">
              Live Streams
            </Link>
            . Look at the camera while enrolling (~3 seconds).
          </p>

          <label className="mb-sm block text-label-md text-on-surface" htmlFor="display-name">
            Display name
          </label>
          <input
            id="display-name"
            type="text"
            value={name}
            onChange={(e) => setName(e.target.value)}
            placeholder="e.g. Carlos Pano"
            className="mb-lg w-full rounded-lg border border-outline-variant bg-surface-container-lowest px-4 py-3 text-body-md text-on-surface outline-none focus:border-primary"
            maxLength={64}
          />

          <div className="flex flex-wrap gap-2">
            <Button icon="face" onClick={handleEnroll} disabled={loading || status?.ready === false}>
              {loading ? "Enrolling…" : "Enroll my face"}
            </Button>
            <Button
              variant="outline"
              icon="delete"
              onClick={handleDelete}
              disabled={loading || !status?.enrolled}
            >
              Remove enrollment
            </Button>
          </div>

          {message && (
            <p className="mt-md rounded-lg bg-surface-container-high px-3 py-2 text-body-sm text-on-surface">
              {message}
            </p>
          )}

          <div className="mt-lg rounded-lg border border-outline-variant bg-surface-container-low p-md">
            <p className="text-label-md text-on-surface">Status</p>
            <ul className="mt-2 space-y-1 text-body-sm text-outline">
              <li>
                Engine:{" "}
                {status?.ready ? (
                  <span className="text-[#4CAF50]">ready</span>
                ) : (
                  <span className="text-error">not ready — start backend + models</span>
                )}
              </li>
              <li>
                Enrolled:{" "}
                {status?.enrolled ? (
                  <span className="text-on-surface">
                    yes — <strong>{status.name}</strong>
                  </span>
                ) : (
                  "no"
                )}
              </li>
              {status?.error && <li className="text-error">{status.error}</li>}
            </ul>
          </div>
        </Card>

        <Card className="p-lg">
          <h2 className="mb-md font-headline text-headline-sm text-on-surface">
            Where data is stored
          </h2>
          <p className="mb-md text-body-sm text-outline">{storage?.summary}</p>
          {storage?.stored_items.map((item) => (
            <div
              key={item.id}
              className="mb-sm rounded-lg border border-outline-variant bg-surface-container-lowest p-md"
            >
              <p className="text-label-md text-on-surface">
                {item.id}
                {item.used_for_recognition && (
                  <span className="ml-2 text-label-sm text-primary">used for matching</span>
                )}
              </p>
              <p className="mt-1 font-mono text-body-sm text-outline">{item.file ?? "—"}</p>
              <p className="mt-1 text-body-sm text-outline">{item.contents}</p>
            </div>
          ))}
          <p className="mt-md text-label-sm text-outline">
            Folder is in <code className="text-on-surface">.gitignore</code> — not pushed to GitHub.
          </p>
        </Card>
      </div>

      <div className="flex flex-col gap-gutter">
        <Card className="overflow-hidden p-0">
          <div className="border-b border-outline-variant px-md py-sm">
            <p className="text-label-md text-on-surface">Live preview (enrollment)</p>
          </div>
          <div className="relative aspect-video bg-on-surface-variant/10">
            {status?.ready ? (
              <CameraStream streamUrl={streamUrl} alt="Webcam preview" />
            ) : (
              <div className="absolute inset-0 flex items-center justify-center p-md text-center text-body-sm text-outline">
                Start <code>./run-local.sh</code> and allow camera access.
              </div>
            )}
          </div>
        </Card>

        {previewUrl && (
          <Card className="overflow-hidden p-0">
            <div className="border-b border-outline-variant px-md py-sm">
              <p className="text-label-md text-on-surface">Saved enrollment snapshot</p>
              <p className="text-body-sm text-outline">
                Reference only — recognition uses the <code>.npz</code> embedding, not this image.
              </p>
            </div>
            <div className="relative aspect-video bg-on-surface-variant/10">
              {/* eslint-disable-next-line @next/next/no-img-element */}
              <img src={previewUrl} alt="Enrollment preview" className="h-full w-full object-cover" />
            </div>
            <p className="px-md py-sm font-mono text-label-sm text-outline">
              vision-ops-backend/data/faces/enrollment_preview.jpg
            </p>
          </Card>
        )}
      </div>
    </div>
  );
}
