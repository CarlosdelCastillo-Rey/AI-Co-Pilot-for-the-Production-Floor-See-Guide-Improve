/** e.g. SHIFT A-4 — letter from time of day, number from calendar date */
export function currentShiftLabel(now = new Date()): string {
  const h = now.getHours();
  const letter = h >= 6 && h < 14 ? "A" : h >= 14 && h < 22 ? "B" : "C";
  return `SHIFT ${letter}-${now.getDate()}`;
}

export function initialsFromName(name: string): string {
  const parts = name.trim().split(/\s+/).filter(Boolean);
  if (parts.length === 0) return "?";
  if (parts.length === 1) return parts[0].slice(0, 2).toUpperCase();
  return (parts[0][0] + parts[parts.length - 1][0]).toUpperCase();
}
