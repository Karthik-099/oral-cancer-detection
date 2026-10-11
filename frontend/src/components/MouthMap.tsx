import { cn } from "@/lib/utils";

/**
 * Schematic front view of the open mouth. Each region maps to an anatomical
 * site the API accepts. Used to pick a site when capturing, and as a heat map
 * of where findings cluster on the dashboard.
 */
export const SITES = [
  "Lip", "Hard palate", "Soft palate", "Gingiva", "Buccal mucosa", "Tongue (dorsal)",
  "Tongue (lateral)", "Tongue (ventral)", "Floor of mouth", "Retromolar trigone", "Other",
] as const;
export type Site = (typeof SITES)[number];

const REGIONS: { site: Site; d: string; label: [number, number] }[] = [
  { site: "Lip", d: "M40 70 Q160 18 280 70 Q160 44 40 70 Z M40 250 Q160 310 280 250 Q160 286 40 250 Z", label: [160, 36] },
  { site: "Gingiva", d: "M62 78 Q160 46 258 78 L250 94 Q160 66 70 94 Z M70 232 Q160 262 250 232 L258 246 Q160 280 62 246 Z", label: [160, 72] },
  { site: "Hard palate", d: "M78 98 Q160 72 242 98 Q228 128 160 132 Q92 128 78 98 Z", label: [160, 108] },
  { site: "Soft palate", d: "M100 132 Q160 140 220 132 Q206 152 160 156 Q114 152 100 132 Z", label: [160, 146] },
  { site: "Buccal mucosa", d: "M40 90 Q30 160 48 236 L72 226 Q58 160 70 100 Z M280 90 Q290 160 272 236 L248 226 Q262 160 250 100 Z", label: [44, 164] },
  { site: "Retromolar trigone", d: "M72 100 L96 132 L88 150 L66 120 Z M248 100 L224 132 L232 150 L254 120 Z", label: [70, 124] },
  { site: "Tongue (dorsal)", d: "M118 164 Q160 154 202 164 Q206 196 160 204 Q114 196 118 164 Z", label: [160, 182] },
  { site: "Tongue (lateral)", d: "M96 166 Q104 160 118 164 Q114 196 128 206 Q102 202 94 186 Z M224 166 Q216 160 202 164 Q206 196 192 206 Q218 202 226 186 Z", label: [104, 186] },
  { site: "Tongue (ventral)", d: "M128 206 Q160 214 192 206 Q186 218 160 220 Q134 218 128 206 Z", label: [160, 213] },
  { site: "Floor of mouth", d: "M80 214 Q160 236 240 214 L246 228 Q160 254 74 228 Z", label: [160, 232] },
];

export function MouthMap({ value, onChange, counts, className }: {
  value?: string | null;
  onChange?: (site: Site) => void;
  counts?: Record<string, number>;
  className?: string;
}) {
  const max = counts ? Math.max(1, ...Object.values(counts)) : 1;
  const interactive = !!onChange;
  return (
    <svg viewBox="0 0 320 300" className={cn("w-full", className)} role={interactive ? "radiogroup" : "img"} aria-label="Oral cavity site map">
      <ellipse cx="160" cy="160" rx="138" ry="128" className="fill-surface-2 stroke-line" strokeWidth="1.5" />
      <path d="M48 160 Q160 120 272 160" className="stroke-line" strokeDasharray="3 4" fill="none" />
      {REGIONS.map((r) => {
        const active = value === r.site;
        const heat = counts ? (counts[r.site] ?? 0) / max : 0;
        return (
          <path
            key={r.site}
            d={r.d}
            role={interactive ? "radio" : undefined}
            aria-checked={interactive ? active : undefined}
            aria-label={r.site}
            tabIndex={interactive ? 0 : -1}
            onClick={() => onChange?.(r.site)}
            onKeyDown={(e) => (e.key === "Enter" || e.key === " ") && onChange?.(r.site)}
            className={cn(
              "stroke-surface transition-all duration-200",
              interactive && "cursor-pointer hover:fill-brand/40 focus:outline-none focus-visible:fill-brand/50",
              active ? "fill-brand" : counts ? "fill-high" : "fill-ink/10",
            )}
            style={counts && !active ? { fillOpacity: 0.08 + heat * 0.72 } : undefined}
            strokeWidth={2}
          >
            <title>{counts ? `${r.site}: ${counts[r.site] ?? 0}` : r.site}</title>
          </path>
        );
      })}
    </svg>
  );
}
