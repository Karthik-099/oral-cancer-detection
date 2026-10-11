import { useId, useState, type ReactNode } from "react";
import { motion } from "framer-motion";
import { AlertTriangle, CheckCircle2, CircleDashed, HelpCircle, ImageOff, Star, XCircle } from "lucide-react";
import { ApiError, type Feedback, type PatientRef, type Tier } from "@/lib/api";
import { cn } from "@/lib/utils";
import { Modal } from "@/components/Modal";
import { Button } from "@/components/ui";

/**
 * Operating points of OralNet-B4. Defaults match the backend; syncThresholds()
 * replaces them with the deployment's configured values from /api/models.
 * (`export let` bindings are live, so importers see the update.)
 */
export let THRESHOLD = 0.3;
export let HIGH_THRESHOLD = 0.55;

export function syncThresholds(flag: number, high: number) {
  if (flag > 0 && high > flag && high < 1) {
    THRESHOLD = flag;
    HIGH_THRESHOLD = high;
  }
}

export const RISK_FACTORS = [
  "Tobacco smoking", "Smokeless tobacco", "Areca / betel nut", "Alcohol", "Previous oral lesion", "HPV-positive history",
] as const;

export const SPECIALTIES = [
  "General dentist", "Oral medicine", "Maxillofacial surgeon", "ENT / Head & neck", "Oral pathologist",
  "Community health worker", "Researcher", "Other",
] as const;

export const IMPRESSIONS = [
  { value: "normal", label: "Normal" },
  { value: "benign", label: "Benign" },
  { value: "opmd", label: "OPMD" },
  { value: "malignant", label: "Malignant" },
] as const;

export const SEX_LABEL: Record<string, string> = { female: "Female", male: "Male", other: "Other" };

export function patientName(p: PatientRef | null | undefined) {
  return p?.name || p?.reference || "Unnamed patient";
}

export function patientMeta(p: PatientRef | null | undefined) {
  if (!p) return "";
  const parts: string[] = [];
  if (p.name && p.reference) parts.push(p.reference);
  if (p.age !== null && p.age !== undefined) parts.push(`${p.age} y`);
  if (p.sex) parts.push(SEX_LABEL[p.sex] ?? p.sex);
  return parts.join(" · ");
}

export function tierOf(p: number): Tier {
  return p >= HIGH_THRESHOLD ? "high" : p >= THRESHOLD ? "moderate" : "low";
}

export const TIER_RGB: Record<Tier, string> = { low: "rgb(var(--low))", moderate: "rgb(var(--moderate))", high: "rgb(var(--high))" };

export function needsReview(c: { tier: Tier; feedback: Feedback | null }) {
  return c.tier !== "low" && !c.feedback;
}

/** Turns API failures into a message a clinician can act on. */
export function errorMessage(e: unknown, fallback = "Something went wrong. Please try again.") {
  if (e instanceof ApiError) {
    if (e.status === 429) return "You're going a little fast. Wait a moment and try again.";
    if (e.status === 503) return "The AI model is warming up. Try again in a few seconds.";
    if (e.status >= 500) return "The OralAID service hit a problem. Please try again shortly.";
    return e.message || fallback;
  }
  return fallback;
}

export function Segmented<T extends string>({ value, onChange, options, label, size = "md", className }: {
  value: T | null | undefined;
  onChange: (v: T) => void;
  options: readonly { value: T; label: ReactNode; icon?: ReactNode }[];
  label: string;
  size?: "sm" | "md";
  className?: string;
}) {
  const id = useId();
  return (
    <div role="radiogroup" aria-label={label} className={cn("inline-flex rounded-xl border border-line bg-surface-2/70 p-1", className)}>
      {options.map((o) => {
        const active = o.value === value;
        return (
          <button
            key={o.value} type="button" role="radio" aria-checked={active} onClick={() => onChange(o.value)}
            className={cn(
              "relative flex flex-1 items-center justify-center gap-1.5 whitespace-nowrap rounded-lg font-medium transition-colors",
              size === "sm" ? "h-7 px-2.5 text-xs" : "h-9 px-3.5 text-sm",
              active ? "text-ink" : "text-muted hover:text-ink",
            )}
          >
            {active && (
              <motion.span layoutId={`seg-${id}`} className="absolute inset-0 rounded-lg border border-line bg-surface shadow-sm"
                transition={{ type: "spring", damping: 30, stiffness: 400 }} />
            )}
            <span className="relative flex items-center gap-1.5">{o.icon}{o.label}</span>
          </button>
        );
      })}
    </div>
  );
}

export function Stars({ value, onChange, size = "md" }: { value: number | null | undefined; onChange?: (v: number) => void; size?: "sm" | "md" }) {
  const [hover, setHover] = useState<number | null>(null);
  const shown = hover ?? value ?? 0;
  const s = size === "sm" ? "h-3.5 w-3.5" : "h-6 w-6";
  return (
    <div className="flex items-center gap-0.5" role={onChange ? "radiogroup" : "img"} aria-label={`Explanation usefulness ${value ?? 0} of 5`}
      onMouseLeave={() => setHover(null)}>
      {[1, 2, 3, 4, 5].map((n) => {
        const icon = <Star className={cn(s, "transition", n <= shown ? "fill-moderate text-moderate" : "text-line")} />;
        return onChange ? (
          <button key={n} type="button" role="radio" aria-checked={value === n} aria-label={`${n} star${n > 1 ? "s" : ""}`}
            onMouseEnter={() => setHover(n)} onClick={() => onChange(n)} className="rounded p-0.5 hover:scale-110 transition">
            {icon}
          </button>
        ) : <span key={n}>{icon}</span>;
      })}
    </div>
  );
}

export function ConfirmDialog({ open, onClose, onConfirm, title, body, confirmLabel = "Delete", loading }: {
  open: boolean; onClose: () => void; onConfirm: () => void; title: string; body: ReactNode; confirmLabel?: string; loading?: boolean;
}) {
  return (
    <Modal open={open} onClose={onClose}>
      <div className="flex gap-4">
        <div className="grid h-10 w-10 shrink-0 place-items-center rounded-xl bg-high/10 text-high"><AlertTriangle className="h-5 w-5" /></div>
        <div className="min-w-0">
          <h2 className="text-base font-semibold">{title}</h2>
          <div className="mt-1 text-sm text-muted">{body}</div>
        </div>
      </div>
      <div className="mt-6 flex justify-end gap-2">
        <Button variant="ghost" onClick={onClose} disabled={loading}>Cancel</Button>
        <Button variant="danger" onClick={onConfirm} loading={loading}>{confirmLabel}</Button>
      </div>
    </Modal>
  );
}

export function Sparkline({ data, className, color = "rgb(var(--brand))" }: { data: number[]; className?: string; color?: string }) {
  const gid = useId().replace(/:/g, "");
  if (data.length < 2) return null;
  const w = 100, h = 28;
  const max = Math.max(1, ...data);
  const pts = data.map((v, i) => [(i / (data.length - 1)) * w, h - 2 - (v / max) * (h - 4)] as const);
  const line = pts.map(([x, y], i) => `${i ? "L" : "M"}${x.toFixed(2)} ${y.toFixed(2)}`).join(" ");
  return (
    <svg viewBox={`0 0 ${w} ${h}`} preserveAspectRatio="none" className={cn("h-7 w-full overflow-visible", className)} aria-hidden>
      <defs>
        <linearGradient id={gid} x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor={color} stopOpacity={0.25} />
          <stop offset="100%" stopColor={color} stopOpacity={0} />
        </linearGradient>
      </defs>
      <path d={`${line} L${w} ${h} L0 ${h} Z`} fill={`url(#${gid})`} />
      <path d={line} fill="none" stroke={color} strokeWidth={1.5} vectorEffect="non-scaling-stroke" strokeLinejoin="round" strokeLinecap="round" />
    </svg>
  );
}

export function StatTile({ label, value, sub, icon, spark, sparkColor, tone, loading }: {
  label: string; value: ReactNode; sub?: ReactNode; icon?: ReactNode; spark?: number[]; sparkColor?: string;
  tone?: "high" | "moderate" | "low" | "brand"; loading?: boolean;
}) {
  const toneText = { high: "text-high", moderate: "text-moderate", low: "text-low", brand: "text-brand" };
  return (
    <div className="card flex min-w-0 flex-col p-4 sm:p-5">
      <div className="flex items-center justify-between gap-2">
        <span className="truncate text-[13px] text-muted">{label}</span>
        {icon && <span className={cn("text-muted", tone && toneText[tone])}>{icon}</span>}
      </div>
      {loading ? (
        <div className="skeleton mt-3 h-8 w-20" />
      ) : (
        <div className={cn("mt-2 text-[28px] font-semibold leading-none tracking-tight tabular-nums", tone && toneText[tone])}>{value}</div>
      )}
      <div className="mt-auto pt-3">
        {spark && !loading ? <Sparkline data={spark} color={sparkColor} /> : sub ? <p className="truncate text-xs text-muted">{sub}</p> : <div className="h-4" />}
      </div>
      {spark && sub && !loading && <p className="mt-1 truncate text-xs text-muted">{sub}</p>}
    </div>
  );
}

export function Thumb({ src, alt = "", className }: { src?: string | null; alt?: string; className?: string }) {
  const [failed, setFailed] = useState(false);
  return (
    <div className={cn("relative shrink-0 overflow-hidden rounded-lg bg-surface-2", className)}>
      {src && !failed ? (
        <img src={src} alt={alt} loading="lazy" onError={() => setFailed(true)} className="h-full w-full object-cover" />
      ) : (
        <div className="grid h-full w-full place-items-center text-muted"><ImageOff className="h-4 w-4" /></div>
      )}
    </div>
  );
}

export function FeedbackStatus({ feedback, tier, compact }: { feedback: Feedback | null; tier: Tier; compact?: boolean }) {
  if (feedback) {
    const map = {
      agree: { icon: CheckCircle2, label: "Agreed", cls: "text-low" },
      disagree: { icon: XCircle, label: "Disagreed", cls: "text-high" },
      unsure: { icon: HelpCircle, label: "Unsure", cls: "text-moderate" },
    }[feedback.verdict];
    return <span className={cn("inline-flex items-center gap-1.5 text-xs font-medium", map.cls)}><map.icon className="h-3.5 w-3.5" />{map.label}</span>;
  }
  if (tier !== "low") {
    return <span className="inline-flex items-center gap-1.5 text-xs font-medium text-accent"><CircleDashed className="h-3.5 w-3.5" />{compact ? "Review" : "Awaiting review"}</span>;
  }
  return <span className="text-xs text-muted">Not reviewed</span>;
}

export function ProbabilityBar({ value }: { value: number }) {
  const t = tierOf(value);
  return (
    <div className="flex items-center gap-2">
      <span className="w-9 text-right text-sm font-medium tabular-nums">{Math.round(value * 100)}%</span>
      <div className="relative h-1.5 w-16 overflow-hidden rounded-full bg-surface-2">
        <div className="absolute inset-y-0 left-0 rounded-full" style={{ width: `${Math.max(3, value * 100)}%`, background: TIER_RGB[t] }} />
      </div>
    </div>
  );
}

export function Chip({ active, onClick, children, className }: { active: boolean; onClick: () => void; children: ReactNode; className?: string }) {
  return (
    <button type="button" aria-pressed={active} onClick={onClick}
      className={cn(
        "inline-flex items-center gap-1.5 rounded-full border px-3 py-1.5 text-[13px] font-medium transition",
        active ? "border-brand/40 bg-brand/10 text-brand" : "border-line bg-surface text-muted hover:border-ink/20 hover:text-ink",
        className,
      )}>
      {children}
    </button>
  );
}

export function SectionTitle({ children, action, className }: { children: ReactNode; action?: ReactNode; className?: string }) {
  return (
    <div className={cn("mb-4 flex items-center justify-between gap-3", className)}>
      <h2 className="text-[15px] font-semibold tracking-tight">{children}</h2>
      {action}
    </div>
  );
}

export function Toggle({ checked, onChange, label }: { checked: boolean; onChange: (v: boolean) => void; label: string }) {
  return (
    <button type="button" role="switch" aria-checked={checked} aria-label={label} onClick={() => onChange(!checked)}
      className={cn("relative mt-0.5 h-6 w-11 shrink-0 rounded-full transition-colors", checked ? "bg-brand" : "bg-line")}>
      <motion.span layout transition={{ type: "spring", damping: 30, stiffness: 500 }}
        className={cn("absolute top-0.5 h-5 w-5 rounded-full bg-white shadow", checked ? "right-0.5" : "left-0.5")} />
    </button>
  );
}
