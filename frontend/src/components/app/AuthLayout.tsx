import type { ReactNode } from "react";
import { Link } from "react-router-dom";
import { motion } from "framer-motion";
import { ShieldCheck } from "lucide-react";
import { Logo } from "@/components/Logo";
import { ThemeButton } from "@/components/SiteLayout";

/** Split auth screen: form on the left, calm branded panel on large screens. */
export function AuthLayout({ title, subtitle, children, footer }: { title: string; subtitle: ReactNode; children: ReactNode; footer?: ReactNode }) {
  return (
    <div className="min-h-screen bg-bg lg:grid lg:grid-cols-[minmax(0,1fr)_minmax(0,1.05fr)]">
      <div className="flex min-h-screen flex-col px-5 py-6 sm:px-10">
        <div className="flex items-center justify-between">
          <Link to="/" aria-label="OralAID home"><Logo /></Link>
          <ThemeButton />
        </div>
        <div className="flex flex-1 items-center justify-center py-10">
          <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.4, ease: [0.2, 0.7, 0.2, 1] }}
            className="w-full max-w-[400px]">
            <h1 className="display text-[40px] sm:text-[44px]">{title}</h1>
            <p className="mt-2 text-sm text-muted">{subtitle}</p>
            <div className="mt-8">{children}</div>
            {footer && <div className="mt-8 text-center text-sm text-muted">{footer}</div>}
          </motion.div>
        </div>
        <p className="text-center text-xs text-muted lg:text-left">Clinical decision support. Not a diagnostic device.</p>
      </div>
      <BrandPanel />
    </div>
  );
}

function BrandPanel() {
  return (
    <div className="relative hidden overflow-hidden border-l border-line bg-surface-2/60 lg:flex lg:flex-col lg:justify-between lg:p-12">
      <div className="grid-bg pointer-events-none absolute inset-0 opacity-50 [mask-image:radial-gradient(ellipse_at_center,black_30%,transparent_75%)]" />
      <div className="pointer-events-none absolute -right-32 -top-32 h-96 w-96 rounded-full bg-brand/15 blur-3xl" />
      <div className="relative">
        <p className="eyebrow">Explainable screening support</p>
        <h2 className="display mt-4 max-w-md text-[52px] xl:text-[60px]">
          See what the model sees, <em className="text-brand">before</em> you decide.
        </h2>
        <p className="mt-5 max-w-md text-[15px] text-muted">
          Every result arrives with Grad-CAM attention, a localised region and an image quality check, so you can weigh the score
          against your own examination.
        </p>
      </div>
      <motion.div initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.15, duration: 0.6 }} className="relative my-10">
        <MiniPreview />
      </motion.div>
      <div className="relative flex items-center gap-2 text-xs text-muted">
        <ShieldCheck className="h-4 w-4 text-brand" />Clinical decision support. Not a diagnostic device.
      </div>
    </div>
  );
}

/** Stylised product preview drawn with CSS and SVG only. */
function MiniPreview() {
  return (
    <div className="mx-auto w-full max-w-md rounded-2xl border border-line bg-surface p-4 shadow-lift">
      <div className="mb-3 flex items-center justify-between">
        <div className="flex items-center gap-2">
          <span className="h-2 w-2 rounded-full bg-line" /><span className="h-2 w-2 rounded-full bg-line" /><span className="h-2 w-2 rounded-full bg-line" />
        </div>
        <span className="font-mono text-[10px] text-muted">case 7f3a91c2</span>
      </div>
      <div className="grid grid-cols-[1.2fr_1fr] gap-3">
        <div className="relative aspect-square overflow-hidden rounded-xl bg-[radial-gradient(circle_at_45%_55%,#f3b2a5,#d9857a_45%,#a8574f_80%)]">
          <svg viewBox="0 0 100 100" className="absolute inset-0 h-full w-full">
            <defs>
              <radialGradient id="mp-cam" cx="58%" cy="46%" r="30%">
                <stop offset="0%" stopColor="#ff3b2f" stopOpacity="0.85" />
                <stop offset="45%" stopColor="#ffb020" stopOpacity="0.55" />
                <stop offset="100%" stopColor="#2563eb" stopOpacity="0" />
              </radialGradient>
            </defs>
            <rect width="100" height="100" fill="url(#mp-cam)" />
            <path d="M44 34 Q62 28 70 42 Q74 56 60 60 Q46 62 42 50 Q40 40 44 34 Z" fill="none" stroke="white" strokeWidth="1.2" strokeDasharray="2.5 2" />
          </svg>
          <div className="absolute inset-x-0 h-10 animate-scan bg-gradient-to-b from-transparent via-white/30 to-transparent" />
          <span className="absolute left-2 top-2 rounded bg-black/50 px-1.5 py-0.5 text-[9px] font-medium text-white">Grad-CAM</span>
        </div>
        <div className="flex flex-col gap-2.5">
          <div className="rounded-xl border border-line p-2.5">
            <p className="text-[10px] text-muted">Suspicion score</p>
            <p className="font-display text-3xl leading-none tabular-nums">62<span className="text-base text-muted">%</span></p>
            <div className="mt-2 flex h-1.5 overflow-hidden rounded-full">
              <span className="w-[30%] bg-low/40" /><span className="w-[25%] bg-moderate/40" /><span className="flex-1 bg-high/40" />
            </div>
          </div>
          <span className="inline-flex w-fit items-center gap-1.5 rounded-full bg-high/10 px-2 py-0.5 text-[10px] font-medium text-high ring-1 ring-inset ring-high/25">
            <span className="h-1.5 w-1.5 rounded-full bg-high" />High risk
          </span>
          <div className="space-y-1.5 rounded-xl border border-line p-2.5">
            <div className="flex justify-between text-[10px] text-muted"><span>Image quality</span><span className="tabular-nums">86</span></div>
            <div className="h-1 overflow-hidden rounded-full bg-surface-2"><div className="h-full w-[86%] rounded-full bg-low" /></div>
            <div className="flex justify-between text-[10px] text-muted"><span>Attention area</span><span className="tabular-nums">12%</span></div>
          </div>
        </div>
      </div>
    </div>
  );
}
