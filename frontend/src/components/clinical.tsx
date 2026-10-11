import { useCallback, useEffect, useRef, useState } from "react";
import { motion } from "framer-motion";
import { MoveHorizontal } from "lucide-react";
import type { Tier } from "@/lib/api";
import { cn, TIER_META } from "@/lib/utils";

export function RiskBadge({ tier, className, size = "md" }: { tier: Tier; className?: string; size?: "sm" | "md" }) {
  const m = TIER_META[tier];
  return (
    <span className={cn("inline-flex items-center gap-1.5 rounded-full font-medium ring-1 ring-inset", m.bg, m.text, m.ring,
      size === "sm" ? "px-2 py-0.5 text-[11px]" : "px-2.5 py-1 text-xs", className)}>
      <span className={cn("h-1.5 w-1.5 rounded-full", m.dot)} />
      {size === "sm" ? m.short : m.label}
    </span>
  );
}

/**
 * Semicircular meter showing P(suspicious) with the two operating thresholds
 * marked, so clinicians can see how close a call was.
 */
export function ProbabilityMeter({ value, threshold, high, size = 220 }: { value: number; threshold: number; high: number; size?: number }) {
  const r = size / 2 - 14;
  const cx = size / 2;
  const cy = size / 2;
  const arc = (from: number, to: number) => {
    const a0 = Math.PI * (1 - from);
    const a1 = Math.PI * (1 - to);
    const p = (a: number) => `${cx + r * Math.cos(a)} ${cy - r * Math.sin(a)}`;
    // The gauge spans at most 180 degrees, so the SVG large-arc flag is always 0.
    return `M ${p(a0)} A ${r} ${r} 0 0 1 ${p(a1)}`;
  };
  const tier: Tier = value >= high ? "high" : value >= threshold ? "moderate" : "low";
  const angle = Math.PI * (1 - value);
  const tick = (t: number) => {
    const a = Math.PI * (1 - t);
    return { x1: cx + (r - 12) * Math.cos(a), y1: cy - (r - 12) * Math.sin(a), x2: cx + (r + 8) * Math.cos(a), y2: cy - (r + 8) * Math.sin(a) };
  };
  return (
    <div className="relative" style={{ width: size, height: size / 2 + 28 }}>
      <svg width={size} height={size / 2 + 12} className="overflow-visible">
        <path d={arc(0, threshold)} className="stroke-low/25" strokeWidth={12} fill="none" strokeLinecap="round" />
        <path d={arc(threshold, high)} className="stroke-moderate/25" strokeWidth={12} fill="none" />
        <path d={arc(high, 1)} className="stroke-high/25" strokeWidth={12} fill="none" strokeLinecap="round" />
        <motion.path
          d={arc(0, Math.max(value, 0.001))}
          className={cn(tier === "high" ? "stroke-high" : tier === "moderate" ? "stroke-moderate" : "stroke-low")}
          strokeWidth={12} fill="none" strokeLinecap="round"
          initial={{ pathLength: 0 }} animate={{ pathLength: 1 }} transition={{ duration: 1, ease: [0.2, 0.7, 0.2, 1] }}
        />
        {[threshold, high].map((t) => <line key={t} {...tick(t)} className="stroke-ink/40" strokeWidth={1.5} />)}
        <motion.circle
          r={7} className="fill-surface stroke-ink" strokeWidth={2.5}
          initial={{ cx: cx - r, cy }} animate={{ cx: cx + r * Math.cos(angle), cy: cy - r * Math.sin(angle) }}
          transition={{ duration: 1, ease: [0.2, 0.7, 0.2, 1] }}
        />
      </svg>
      <div className="absolute inset-x-0 bottom-0 text-center">
        <div className="font-display text-5xl leading-none tabular-nums">{Math.round(value * 100)}<span className="text-2xl text-muted">%</span></div>
        <div className="mt-1 text-xs text-muted">suspicion score</div>
      </div>
    </div>
  );
}

/** Drag to reveal the overlay over the original image. Keyboard accessible. */
export function CompareSlider({ before, after, labelBefore = "Original", labelAfter = "Overlay", opacity = 1 }: {
  before: string; after: string; labelBefore?: string; labelAfter?: string; opacity?: number;
}) {
  const [pos, setPos] = useState(50);
  const ref = useRef<HTMLDivElement>(null);
  const dragging = useRef(false);

  const move = useCallback((clientX: number) => {
    const box = ref.current?.getBoundingClientRect();
    if (!box) return;
    setPos(Math.min(100, Math.max(0, ((clientX - box.left) / box.width) * 100)));
  }, []);

  useEffect(() => {
    const up = () => (dragging.current = false);
    const mv = (e: PointerEvent) => dragging.current && move(e.clientX);
    window.addEventListener("pointerup", up);
    window.addEventListener("pointermove", mv);
    return () => { window.removeEventListener("pointerup", up); window.removeEventListener("pointermove", mv); };
  }, [move]);

  return (
    <div
      ref={ref}
      className="relative aspect-square w-full select-none overflow-hidden rounded-2xl bg-surface-2 touch-none"
      onPointerDown={(e) => { dragging.current = true; move(e.clientX); }}
    >
      <img src={before} alt={labelBefore} className="absolute inset-0 h-full w-full object-cover" draggable={false} />
      <div className="absolute inset-0" style={{ clipPath: `inset(0 0 0 ${pos}%)` }}>
        <img src={after} alt={labelAfter} className="h-full w-full object-cover" style={{ opacity }} draggable={false} />
      </div>
      <div className="absolute inset-y-0" style={{ left: `${pos}%` }}>
        <div className="absolute inset-y-0 -ml-px w-0.5 bg-white/90 shadow-[0_0_0_1px_rgb(0_0_0/.15)]" />
        <button
          role="slider" aria-label="Comparison position" aria-valuenow={Math.round(pos)} aria-valuemin={0} aria-valuemax={100}
          onKeyDown={(e) => {
            if (e.key === "ArrowLeft") setPos((p) => Math.max(0, p - 5));
            if (e.key === "ArrowRight") setPos((p) => Math.min(100, p + 5));
          }}
          className="absolute top-1/2 -ml-5 -mt-5 grid h-10 w-10 place-items-center rounded-full bg-white text-ink shadow-lift"
        >
          <MoveHorizontal className="h-4 w-4" />
        </button>
      </div>
      <span className="absolute left-3 top-3 rounded-md bg-black/55 px-2 py-1 text-[11px] font-medium text-white backdrop-blur">{labelBefore}</span>
      <span className="absolute right-3 top-3 rounded-md bg-black/55 px-2 py-1 text-[11px] font-medium text-white backdrop-blur">{labelAfter}</span>
    </div>
  );
}

export function QualityBar({ score }: { score: number }) {
  const tone = score >= 80 ? "bg-low" : score >= 55 ? "bg-moderate" : "bg-high";
  return (
    <div className="flex items-center gap-2">
      <div className="h-1.5 flex-1 overflow-hidden rounded-full bg-surface-2">
        <motion.div className={cn("h-full rounded-full", tone)} initial={{ width: 0 }} animate={{ width: `${score}%` }} transition={{ duration: 0.8 }} />
      </div>
      <span className="w-8 text-right text-xs tabular-nums text-muted">{score}</span>
    </div>
  );
}
