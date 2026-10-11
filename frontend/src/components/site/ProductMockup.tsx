import type { CSSProperties, ReactNode } from "react";
import { motion } from "framer-motion";
import { Activity, FileText, Layers, ScanLine, ShieldCheck } from "lucide-react";
import { ProbabilityMeter, QualityBar, RiskBadge } from "@/components/clinical";
import { cn } from "@/lib/utils";

/*
 * Illustrations of the product. Everything here is drawn with CSS/SVG and is
 * clearly illustrative: no patient images are used anywhere on the site.
 * Hex colours are allowed here because these are image-like artwork.
 */

const TISSUE: CSSProperties = {
  backgroundImage: [
    "radial-gradient(ellipse 55% 40% at 50% 62%, #b8475a 0%, transparent 70%)",
    "radial-gradient(ellipse 70% 30% at 50% 22%, #e9a3a6 0%, transparent 70%)",
    "radial-gradient(ellipse 30% 24% at 30% 48%, #d77f86 0%, transparent 70%)",
    "radial-gradient(ellipse 34% 26% at 72% 54%, #cf6f7a 0%, transparent 70%)",
    "radial-gradient(circle at 50% 50%, #e7a2a1 0%, #c86a73 55%, #6e2a36 100%)",
  ].join(","),
};

const HEAT: CSSProperties = {
  backgroundImage: [
    "radial-gradient(ellipse 26% 22% at 63% 57%, rgba(255,40,20,.85) 0%, rgba(255,170,0,.55) 45%, rgba(255,240,0,.2) 70%, transparent 100%)",
    "radial-gradient(ellipse 16% 12% at 40% 40%, rgba(255,200,0,.35) 0%, transparent 100%)",
    "linear-gradient(rgba(10,40,160,.28), rgba(10,40,160,.28))",
  ].join(","),
  mixBlendMode: "hard-light",
};

/** Deterministic superpixel tiles for the LIME illustration. */
const LIME_CELLS = Array.from({ length: 36 }, (_, i) => {
  const x = i % 6;
  const y = Math.floor(i / 6);
  const dx = x - 3.6;
  const dy = y - 3.3;
  const d = Math.sqrt(dx * dx + dy * dy);
  const w = d < 1.4 ? 0.75 : d < 2.2 ? 0.4 : (i * 7) % 11 === 0 ? -0.45 : (i * 5) % 7 === 0 ? -0.25 : 0;
  return { x, y, w, jitter: ((i * 37) % 9) - 4 };
});

export type TileVariant = "original" | "gradcam" | "region" | "lime";

export function TissueTile({ variant = "original", scan = false, className, children }: {
  variant?: TileVariant; scan?: boolean; className?: string; children?: ReactNode;
}) {
  return (
    <div className={cn("relative aspect-square w-full overflow-hidden rounded-xl bg-surface-2", className)} role="img"
      aria-label={`Illustration of an intraoral photo${variant === "original" ? "" : ` with ${variant === "gradcam" ? "Grad-CAM heatmap" : variant === "region" ? "region outline" : "LIME superpixels"}`}`}>
      <div className="absolute inset-0" style={TISSUE} />
      {/* soft specular highlights */}
      <div className="absolute inset-0 opacity-60" style={{ backgroundImage: "radial-gradient(circle at 34% 30%, rgba(255,255,255,.35), transparent 18%), radial-gradient(circle at 70% 36%, rgba(255,255,255,.18), transparent 12%)" }} />
      {variant === "gradcam" && <div className="absolute inset-0" style={HEAT} />}
      {variant === "region" && (
        <svg viewBox="0 0 100 100" className="absolute inset-0 h-full w-full" aria-hidden>
          <path d="M50 46 C58 41 72 44 75 53 C78 62 72 70 63 71 C54 72 46 66 47 58 C47.5 52 46 48 50 46 Z"
            fill="rgba(52,211,186,.18)" stroke="#34d3ba" strokeWidth="1.2" strokeDasharray="2.5 1.8" />
          <circle cx="63" cy="57" r="1.4" fill="#34d3ba" />
        </svg>
      )}
      {variant === "lime" && (
        <svg viewBox="0 0 60 60" className="absolute inset-0 h-full w-full" aria-hidden>
          {LIME_CELLS.map((c) => (
            <rect key={`${c.x}-${c.y}`} x={c.x * 10 + 0.4} y={c.y * 10 + 0.4} width={9.2} height={9.2} rx={2.5 + c.jitter * 0.3}
              fill={c.w > 0 ? `rgba(239,68,68,${c.w * 0.75})` : c.w < 0 ? `rgba(34,197,94,${-c.w})` : "transparent"}
              stroke="rgba(255,255,255,.35)" strokeWidth=".3" />
          ))}
        </svg>
      )}
      {scan && (
        <div aria-hidden className="pointer-events-none absolute inset-0 overflow-hidden">
          <div className="absolute inset-x-0 h-1/3 animate-scan bg-gradient-to-b from-transparent via-white/25 to-transparent">
            <div className="absolute inset-x-0 bottom-1/2 h-px bg-white/80 shadow-[0_0_12px_2px_rgba(255,255,255,.6)]" />
          </div>
        </div>
      )}
      {children}
    </div>
  );
}

function WindowChrome({ title, children, className }: { title: string; children: ReactNode; className?: string }) {
  return (
    <div className={cn("overflow-hidden rounded-2xl border border-line bg-surface shadow-lift", className)}>
      <div className="flex items-center gap-2 border-b border-line bg-surface-2/60 px-4 py-2.5">
        <span className="h-2.5 w-2.5 rounded-full bg-line" />
        <span className="h-2.5 w-2.5 rounded-full bg-line" />
        <span className="h-2.5 w-2.5 rounded-full bg-line" />
        <span className="ml-3 truncate font-mono text-[11px] text-muted">{title}</span>
      </div>
      {children}
    </div>
  );
}

/** Hero visual: an analysis view with meter, risk tier and heatmap. */
export function AnalysisMockup({ className }: { className?: string }) {
  return (
    <div className={cn("relative", className)}>
      <WindowChrome title="app.oralaid.ai / cases / C-0142">
        <div className="grid gap-5 p-4 sm:grid-cols-[1.05fr_1fr] sm:p-5">
          <div className="space-y-3">
            <div className="relative">
              <TissueTile variant="gradcam" scan />
              <span className="absolute left-2.5 top-2.5 rounded-md bg-black/55 px-2 py-1 text-[10px] font-medium text-white backdrop-blur">Grad-CAM</span>
              <span className="absolute bottom-2.5 right-2.5 rounded-md bg-black/55 px-2 py-1 font-mono text-[10px] text-white backdrop-blur">Illustrative</span>
            </div>
            <div className="grid grid-cols-3 gap-2">
              {(["original", "region", "lime"] as const).map((v) => <TissueTile key={v} variant={v} className="rounded-lg" />)}
            </div>
          </div>
          <div className="flex flex-col">
            <div className="flex items-center justify-between gap-2">
              <span className="text-xs text-muted">Lateral tongue</span>
              <RiskBadge tier="high" size="sm" />
            </div>
            <div className="mt-3 flex justify-center">
              <ProbabilityMeter value={0.74} threshold={0.3} high={0.55} size={180} />
            </div>
            <dl className="mt-4 space-y-3 text-xs">
              <div>
                <dt className="mb-1.5 text-muted">Image quality</dt>
                <dd><QualityBar score={86} /></dd>
              </div>
              <div className="flex justify-between border-t border-line pt-3">
                <dt className="text-muted">Uncertainty</dt><dd className="tabular-nums">± 0.06</dd>
              </div>
            </dl>
            <div className="mt-auto rounded-xl border border-high/20 bg-high/5 p-3 text-xs leading-relaxed text-ink/90">
              <span className="font-medium text-high">Recommendation.</span> Refer for specialist evaluation and biopsy.
            </div>
          </div>
        </div>
      </WindowChrome>
      <motion.div
        initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.9, duration: 0.6 }}
        className="absolute -bottom-5 -left-3 hidden items-center gap-2.5 rounded-xl border border-line bg-surface px-3.5 py-2.5 shadow-lift sm:flex"
      >
        <ShieldCheck className="h-4 w-4 text-brand" />
        <span className="text-xs"><span className="font-medium">Clinician review</span> <span className="text-muted">required before action</span></span>
      </motion.div>
    </div>
  );
}

/** Compact mockup of the case list, used on the Platform page. */
export function CaseListMockup({ className }: { className?: string }) {
  const rows: { id: string; site: string; tier: "low" | "moderate" | "high"; p: number; status: string }[] = [
    { id: "C-0142", site: "Lateral tongue", tier: "high", p: 0.74, status: "Referred" },
    { id: "C-0141", site: "Buccal mucosa", tier: "moderate", p: 0.41, status: "Open" },
    { id: "C-0139", site: "Gingiva", tier: "low", p: 0.12, status: "Closed" },
    { id: "C-0137", site: "Floor of mouth", tier: "moderate", p: 0.36, status: "Open" },
  ];
  return (
    <WindowChrome title="Cases" className={className}>
      <ul className="divide-y divide-line">
        {rows.map((r) => (
          <li key={r.id} className="flex items-center gap-3 px-4 py-3">
            <TissueTile variant={r.tier === "low" ? "original" : "gradcam"} className="h-10 w-10 shrink-0 rounded-lg" />
            <div className="min-w-0 flex-1">
              <p className="truncate text-sm font-medium">{r.site}</p>
              <p className="font-mono text-[11px] text-muted">{r.id} · {r.status}</p>
            </div>
            <span className="hidden font-mono text-xs tabular-nums text-muted sm:block">{Math.round(r.p * 100)}%</span>
            <RiskBadge tier={r.tier} size="sm" />
          </li>
        ))}
      </ul>
    </WindowChrome>
  );
}

/** Patient timeline mockup. */
export function TimelineMockup({ className }: { className?: string }) {
  const points = [0.18, 0.22, 0.34, 0.47, 0.61];
  const w = 280;
  const h = 90;
  const xy = points.map((p, i) => [12 + (i * (w - 24)) / (points.length - 1), h - 10 - p * (h - 20)] as const);
  return (
    <WindowChrome title="Patient P-2231 · timeline" className={className}>
      <div className="p-4">
        <svg viewBox={`0 0 ${w} ${h}`} className="h-auto w-full" aria-label="Illustrative suspicion score trend rising across five visits" role="img">
          <rect x="0" y={h - 10 - 0.55 * (h - 20)} width={w} height={0.45 * (h - 20)} className="fill-high/5" />
          <rect x="0" y={h - 10 - 0.3 * (h - 20)} width={w} height={0.25 * (h - 20)} className="fill-moderate/5" />
          <polyline points={xy.map((p) => p.join(",")).join(" ")} fill="none" className="stroke-brand" strokeWidth="2" strokeLinejoin="round" />
          {xy.map(([x, y], i) => <circle key={i} cx={x} cy={y} r="3.5" className="fill-surface stroke-brand" strokeWidth="2" />)}
        </svg>
        <div className="mt-3 flex items-center justify-between text-xs text-muted">
          <span>5 visits</span>
          <span className="inline-flex items-center gap-1 text-high"><Activity className="h-3.5 w-3.5" /> Rising since last visit</span>
        </div>
      </div>
    </WindowChrome>
  );
}

/** Feedback form mockup for the clinician loop. */
export function FeedbackMockup({ className }: { className?: string }) {
  return (
    <WindowChrome title="Clinician feedback" className={className}>
      <div className="space-y-4 p-4 text-sm">
        <div>
          <p className="mb-2 text-xs text-muted">Do you agree with the model?</p>
          <div className="grid grid-cols-3 gap-2 text-xs">
            {["Agree", "Disagree", "Unsure"].map((l, i) => (
              <span key={l} className={cn("rounded-lg border px-2 py-1.5 text-center", i === 0 ? "border-brand bg-brand/10 text-brand" : "border-line text-muted")}>{l}</span>
            ))}
          </div>
        </div>
        <div className="grid grid-cols-2 gap-2 text-xs">
          <div className="rounded-lg border border-line px-2.5 py-2"><p className="text-muted">Impression</p><p className="mt-0.5 font-medium">OPMD</p></div>
          <div className="rounded-lg border border-line px-2.5 py-2"><p className="text-muted">Histopathology</p><p className="mt-0.5 font-medium">Pending</p></div>
        </div>
        <div>
          <p className="mb-2 text-xs text-muted">Was the explanation useful?</p>
          <div className="flex gap-1.5">
            {[1, 2, 3, 4, 5].map((n) => <span key={n} className={cn("h-1.5 flex-1 rounded-full", n <= 4 ? "bg-brand" : "bg-surface-2")} />)}
          </div>
        </div>
      </div>
    </WindowChrome>
  );
}

/** Report mockup. */
export function ReportMockup({ className }: { className?: string }) {
  return (
    <WindowChrome title="Report · printable" className={className}>
      <div className="space-y-3 p-4">
        <div className="flex items-center gap-2 text-xs"><FileText className="h-4 w-4 text-brand" /><span className="font-medium">Screening summary</span></div>
        <div className="grid grid-cols-[72px_1fr] gap-3">
          <TissueTile variant="region" className="rounded-lg" />
          <div className="space-y-2 pt-1">
            <div className="h-2 w-3/4 rounded bg-surface-2" />
            <div className="h-2 w-full rounded bg-surface-2" />
            <div className="h-2 w-2/3 rounded bg-surface-2" />
            <RiskBadge tier="moderate" size="sm" />
          </div>
        </div>
        <div className="flex gap-3 border-t border-line pt-3 text-[11px] text-muted">
          <span className="inline-flex items-center gap-1"><Layers className="h-3.5 w-3.5" /> 3 images</span>
          <span className="inline-flex items-center gap-1"><ScanLine className="h-3.5 w-3.5" /> Quality 82</span>
        </div>
      </div>
    </WindowChrome>
  );
}
