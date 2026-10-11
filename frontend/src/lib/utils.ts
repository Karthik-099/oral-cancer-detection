import { clsx, type ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";
import type { Tier } from "./api";

export const cn = (...inputs: ClassValue[]) => twMerge(clsx(inputs));

export const pct = (v: number | null | undefined, digits = 0) =>
  v === null || v === undefined ? "–" : `${(v * 100).toFixed(digits)}%`;

export const TIER_META: Record<Tier, { label: string; short: string; text: string; bg: string; ring: string; dot: string }> = {
  low: { label: "Low risk", short: "Low", text: "text-low", bg: "bg-low/10", ring: "ring-low/25", dot: "bg-low" },
  moderate: { label: "Moderate risk", short: "Moderate", text: "text-moderate", bg: "bg-moderate/10", ring: "ring-moderate/25", dot: "bg-moderate" },
  high: { label: "High risk", short: "High", text: "text-high", bg: "bg-high/10", ring: "ring-high/25", dot: "bg-high" },
};

export function timeAgo(iso: string) {
  // API timestamps are naive UTC.
  const d = new Date(iso.endsWith("Z") ? iso : `${iso}Z`);
  const s = Math.round((Date.now() - d.getTime()) / 1000);
  if (s < 60) return "just now";
  if (s < 3600) return `${Math.floor(s / 60)}m ago`;
  if (s < 86400) return `${Math.floor(s / 3600)}h ago`;
  if (s < 86400 * 7) return `${Math.floor(s / 86400)}d ago`;
  return d.toLocaleDateString(undefined, { month: "short", day: "numeric", year: "numeric" });
}

export function formatDate(iso: string) {
  const d = new Date(iso.endsWith("Z") ? iso : `${iso}Z`);
  return d.toLocaleString(undefined, { dateStyle: "medium", timeStyle: "short" });
}

export function downloadBlob(blob: Blob, filename: string) {
  const url = URL.createObjectURL(blob);
  const a = Object.assign(document.createElement("a"), { href: url, download: filename });
  a.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}
