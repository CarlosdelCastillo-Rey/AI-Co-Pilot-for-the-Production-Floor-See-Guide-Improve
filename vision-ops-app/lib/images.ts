/** True when URL points at an image asset (not an app route like /live). */
export function isTimelineThumbnail(src: string | undefined | null): boolean {
  if (!src?.trim()) return false;
  const lower = src.toLowerCase().split("?")[0];
  if (/\.(jpg|jpeg|png|gif|webp|avif)$/.test(lower)) return true;
  if (lower.includes("googleusercontent.com")) return true;
  if (lower.includes("/artifacts/") || lower.includes("/api/vision/") || lower.includes("/vision-api/")) return true;
  return false;
}
