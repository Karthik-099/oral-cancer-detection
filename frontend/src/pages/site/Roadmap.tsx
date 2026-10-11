import { useState } from "react";
import { AnimatePresence, motion } from "framer-motion";
import { Cable, CheckCircle2, Cpu, FolderInput, MonitorCheck, Radio, WifiOff, Zap } from "lucide-react";
import { useSeo } from "@/lib/seo";
import { cn } from "@/lib/utils";
import { Reveal } from "@/components/Reveal";
import { Badge } from "@/components/ui";
import { RiskBadge } from "@/components/clinical";
import { CheckList, PageHero, Section, SectionHeading } from "@/components/site/marketing";
import { LeadForm } from "@/components/site/LeadForm";
import { TissueTile } from "@/components/site/ProductMockup";

type Status = "Shipped" | "In progress" | "Planned" | "Exploring";

const STATUS_STYLE: Record<Status, { tone: "low" | "brand" | "neutral" | "accent"; dot: string }> = {
  Shipped: { tone: "low", dot: "bg-low" },
  "In progress": { tone: "brand", dot: "bg-brand" },
  Planned: { tone: "neutral", dot: "bg-muted" },
  Exploring: { tone: "accent", dot: "bg-accent" },
};

const ITEMS: { status: Status; title: string; body: string }[] = [
  { status: "Shipped", title: "OralNet-B4 with test-time augmentation", body: "EfficientNet-B4 screening model with multi-scale and flip TTA, uncertainty estimates and three risk tiers." },
  { status: "Shipped", title: "Grad-CAM, Region Localizer and LIME", body: "Three complementary explanations, with LIME superpixels available on demand." },
  { status: "Shipped", title: "Clinician feedback loop", body: "Verdict, clinical impression, histopathology and explanation rating captured per case." },
  { status: "Shipped", title: "Cases, patients and reports", body: "Case queues, longitudinal timelines, printable reports, CSV export and dashboard analytics." },
  { status: "Shipped", title: "API keys and audit log", body: "Integration keys via X-API-Key, a full audit trail, signed URLs, lockout and rate limiting." },
  { status: "Shipped", title: "On-premise Docker edition", body: "CPU-only, air-gap capable deployment for clinics with strict data residency." },
  { status: "In progress", title: "OralAID Desktop for Windows", body: "Offline app for clinic PCs with folder batch import and a bridge from intraoral camera software." },
  { status: "In progress", title: "Annotated lesion mask dataset", body: "Clinician-drawn lesion outlines to train a true pixel-level segmentation model." },
  { status: "Planned", title: "External validation study", body: "Independent, multi-site evaluation across devices and populations, published on the Models page." },
  { status: "Planned", title: "HL7 FHIR and DICOM", body: "Structured results into EHRs and DICOM visible-light studies in and out." },
  { status: "Planned", title: "Multi-class lesion typing", body: "Beyond suspicious vs not: distinguishing common oral potentially malignant disorders." },
  { status: "Exploring", title: "On-device model for dental hardware", body: "A quantized model running inside intraoral cameras and chairside units, in real time." },
  { status: "Exploring", title: "Multilingual patient summaries", body: "Plain-language explanations for patients, in their own language." },
];

const FILTERS: ("All" | Status)[] = ["All", "Shipped", "In progress", "Planned", "Exploring"];

function Timeline() {
  const [filter, setFilter] = useState<(typeof FILTERS)[number]>("All");
  const shown = ITEMS.filter((i) => filter === "All" || i.status === filter);
  return (
    <Section>
      <div className="flex flex-col justify-between gap-6 lg:flex-row lg:items-end">
        <SectionHeading eyebrow="Timeline" title="What's shipped, and what's next" />
        <div className="flex flex-wrap gap-2" role="group" aria-label="Filter by status">
          {FILTERS.map((f) => (
            <button key={f} onClick={() => setFilter(f)} aria-pressed={filter === f}
              className={cn("rounded-full border px-3 py-1 text-xs font-medium transition",
                filter === f ? "border-ink bg-ink text-bg" : "border-line text-muted hover:border-ink/30 hover:text-ink")}>
              {f}
            </button>
          ))}
        </div>
      </div>
      <ol className="relative mt-12 border-l border-line pl-6 sm:ml-2 sm:pl-10">
        <AnimatePresence initial={false}>
          {shown.map((it) => (
            <motion.li key={it.title} layout initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0 }}
              transition={{ duration: 0.3 }} className="relative pb-9 last:pb-0">
              <span className={cn("absolute -left-[31px] top-1.5 h-3 w-3 rounded-full ring-4 ring-bg sm:-left-[47px]", STATUS_STYLE[it.status].dot)} />
              <div className="flex flex-wrap items-center gap-2.5">
                <h3 className="text-base font-semibold">{it.title}</h3>
                <Badge tone={STATUS_STYLE[it.status].tone}>{it.status}</Badge>
              </div>
              <p className="mt-1.5 max-w-2xl text-sm leading-relaxed text-muted">{it.body}</p>
            </motion.li>
          ))}
        </AnimatePresence>
      </ol>
    </Section>
  );
}

function DesktopMockup() {
  const files = [
    { n: "IMG_2041.jpg", tier: "low" as const },
    { n: "IMG_2042.jpg", tier: "moderate" as const },
    { n: "IMG_2043.jpg", tier: "high" as const },
    { n: "IMG_2044.jpg", tier: null },
  ];
  return (
    <div className="overflow-hidden rounded-xl border border-line bg-surface shadow-lift">
      <div className="flex items-center justify-between border-b border-line bg-surface-2/70 px-3 py-2">
        <span className="text-[11px] font-medium">OralAID Desktop</span>
        <span className="flex gap-3 text-muted" aria-hidden><span className="h-px w-2.5 self-center bg-muted" /><span className="h-2.5 w-2.5 border border-muted" /><span className="text-[11px] leading-none">×</span></span>
      </div>
      <div className="grid sm:grid-cols-[150px_1fr]">
        <div className="hidden space-y-1 border-r border-line p-3 text-[11px] text-muted sm:block">
          <p className="eyebrow mb-2 text-[10px]">Watched folders</p>
          {["Camera exports", "Today", "Referrals"].map((f, i) => (
            <div key={f} className={cn("flex items-center gap-1.5 rounded-md px-2 py-1.5", i === 0 && "bg-brand/10 text-brand")}><FolderInput className="h-3.5 w-3.5" />{f}</div>
          ))}
          <div className="mt-4 flex items-center gap-1.5 px-2 text-low"><WifiOff className="h-3.5 w-3.5" />Offline</div>
        </div>
        <ul className="divide-y divide-line">
          {files.map((f) => (
            <li key={f.n} className="flex items-center gap-3 px-3 py-2.5">
              <TissueTile variant={f.tier === "high" ? "gradcam" : "original"} className="h-9 w-9 shrink-0 rounded-md" />
              <span className="min-w-0 flex-1 truncate font-mono text-[11px]">{f.n}</span>
              {f.tier ? <RiskBadge tier={f.tier} size="sm" /> : <span className="text-[11px] text-muted">Analysing...</span>}
            </li>
          ))}
        </ul>
      </div>
    </div>
  );
}

function HardwareDiagram() {
  return (
    <svg viewBox="0 0 420 220" className="h-auto w-full" role="img" aria-label="Intraoral camera captures an image, an on-device model analyses it, and the result appears on the chairside screen">
      {/* camera wand */}
      <g transform="translate(18 120) rotate(-18)">
        <rect width="130" height="22" rx="11" className="fill-surface stroke-line" />
        <rect x="108" y="5" width="14" height="12" rx="3" className="fill-brand/20 stroke-brand/60" />
        <circle cx="115" cy="11" r="3" className="fill-brand" />
      </g>
      <text x="40" y="190" className="fill-muted text-[11px]">Intraoral camera</text>
      {/* chip */}
      <path d="M150 92 L196 108" className="stroke-line" strokeWidth="1.5" strokeDasharray="4 4" />
      <g transform="translate(196 84)">
        <rect width="56" height="56" rx="10" className="fill-ink" />
        {[10, 22, 34, 46].map((v) => (
          <g key={v}>
            <rect x={v - 2} y={-6} width="4" height="6" className="fill-muted" />
            <rect x={v - 2} y={56} width="4" height="6" className="fill-muted" />
          </g>
        ))}
        <text x="28" y="33" textAnchor="middle" className="fill-bg text-[10px] font-semibold">AI</text>
      </g>
      <text x="224" y="170" textAnchor="middle" className="fill-muted text-[11px]">On-device model</text>
      <path d="M252 112 L290 112" className="stroke-brand" strokeWidth="1.5">
        <animate attributeName="stroke-dasharray" values="0 40;40 0" dur="1.6s" repeatCount="indefinite" />
      </path>
      {/* chairside screen */}
      <g transform="translate(290 54)">
        <rect width="118" height="86" rx="10" className="fill-surface stroke-line" />
        <rect x="10" y="10" width="44" height="44" rx="6" className="fill-high/20" />
        <circle cx="34" cy="34" r="12" className="fill-high/50" />
        <rect x="62" y="14" width="44" height="6" rx="3" className="fill-line" />
        <rect x="62" y="26" width="30" height="6" rx="3" className="fill-line" />
        <rect x="62" y="40" width="40" height="12" rx="6" className="fill-high/15" />
        <text x="82" y="49" textAnchor="middle" className="fill-high text-[8px] font-semibold">High</text>
        <rect x="10" y="64" width="98" height="10" rx="5" className="fill-surface-2" />
        <rect x="52" y="140" width="14" height="18" className="fill-line" />
        <rect x="34" y="158" width="50" height="5" rx="2.5" className="fill-line" />
      </g>
      <text x="349" y="190" textAnchor="middle" className="fill-muted text-[11px]">Chairside result</text>
    </svg>
  );
}

export default function Roadmap() {
  useSeo("Roadmap", "What's shipped and what's next for OralAID, including OralAID Desktop for Windows and on-device AI inside dental hardware.");
  return (
    <>
      <PageHero eyebrow="Roadmap" title={<>Where we're <em className="italic text-brand">headed.</em></>}
        body="Built in the open. Here is what we've shipped, what we're building now, and the bigger bets we're exploring." />

      <Timeline />

      <Section id="desktop" className="border-y border-line bg-surface/50">
        <div className="grid items-center gap-12 lg:grid-cols-2 lg:gap-20">
          <div>
            <Badge tone="brand">Coming soon</Badge>
            <h2 className="display mt-5 text-4xl sm:text-6xl">OralAID Desktop <em className="italic text-brand">for Windows.</em></h2>
            <p className="mt-5 text-base leading-relaxed text-muted sm:text-lg">
              The full screening workflow on the clinic PC you already have. No internet required, no data leaving the room.
            </p>
            <CheckList className="mt-7" items={[
              <><span className="font-medium">Works offline</span> on standard clinic PCs, CPU only</>,
              <><span className="font-medium">Batch import</span> from watched folders and camera exports</>,
              <><span className="font-medium">Bridges</span> to intraoral camera software and DICOM workflows</>,
              <><span className="font-medium">Syncs</span> to your cloud or on-prem workspace when you choose</>,
            ]} />
            <div className="mt-10 rounded-2xl border border-line bg-surface p-6 shadow-soft">
              <p className="mb-4 flex items-center gap-2 text-sm font-semibold"><MonitorCheck className="h-4 w-4 text-brand" />Join the desktop waitlist</p>
              <LeadForm kind="waitlist-desktop" withName={false} withMessage={false} submitLabel="Join waitlist"
                successTitle="You're on the waitlist." successBody="We'll email you when the beta opens. Early clinics help shape it." />
            </div>
          </div>
          <Reveal delay={0.08} className="relative">
            <div aria-hidden className="absolute -inset-8 rounded-[40px] bg-gradient-to-br from-brand/15 via-transparent to-transparent blur-2xl" />
            <div className="relative"><DesktopMockup /></div>
          </Reveal>
        </div>
      </Section>

      <Section id="hardware">
        <div className="grid items-center gap-12 lg:grid-cols-2 lg:gap-20">
          <Reveal className="relative order-2 lg:order-1">
            <div aria-hidden className="absolute -inset-8 rounded-[40px] bg-gradient-to-br from-accent/10 via-transparent to-transparent blur-2xl" />
            <div className="card relative p-6 text-brand sm:p-8"><HardwareDiagram /></div>
          </Reveal>
          <div className="order-1 lg:order-2">
            <Badge tone="accent">Exploring partnerships</Badge>
            <h2 className="display mt-5 text-4xl sm:text-6xl">OralAID inside <em className="italic text-brand">dental hardware.</em></h2>
            <p className="mt-5 text-base leading-relaxed text-muted sm:text-lg">
              We want screening to happen without a single extra step. We're looking to partner with intraoral camera and dental chair manufacturers:
              the camera captures, our model predicts on the device in real time, and results appear chairside.
            </p>
            <div className="mt-7 grid gap-3 sm:grid-cols-3">
              {[[Radio, "Real time", "Predictions as you capture"], [Cpu, "On device", "Quantized model, no cloud"], [Cable, "Integrated", "Results on the chair display"]].map(([Icon, t, d]) => {
                const I = Icon as typeof Radio;
                return (
                  <div key={t as string} className="rounded-xl border border-line p-4">
                    <I className="h-4 w-4 text-brand" />
                    <p className="mt-3 text-sm font-semibold">{t as string}</p>
                    <p className="mt-0.5 text-xs text-muted">{d as string}</p>
                  </div>
                );
              })}
            </div>
            <div className="mt-8 rounded-2xl border border-line bg-surface p-6 shadow-soft">
              <p className="mb-4 flex items-center gap-2 text-sm font-semibold"><Zap className="h-4 w-4 text-accent" />Hardware partner or early adopter?</p>
              <LeadForm kind="waitlist-hardware" withMessage messageLabel="What do you build or use? (optional)" submitLabel="Get in touch"
                successTitle="Thanks for reaching out." successBody="Our team will follow up about hardware integration." />
            </div>
          </div>
        </div>
      </Section>

      <Section className="border-t border-line">
        <div className="grid gap-10 lg:grid-cols-2 lg:gap-20">
          <Reveal>
            <p className="eyebrow text-brand">Mission</p>
            <p className="display mt-5 text-4xl sm:text-5xl">Make expert-level oral cancer screening available in <em className="italic text-brand">every chair.</em></p>
          </Reveal>
          <Reveal delay={0.08}>
            <p className="eyebrow text-brand">Vision</p>
            <p className="mt-5 text-lg leading-relaxed text-muted">
              A world where no oral cancer is found late because nobody looked. Where every dental visit includes a quick, explainable screen,
              where clinicians trust the tool because it shows its reasoning, and where it works as well in a rural clinic without internet as in a city hospital.
            </p>
            <p className="mt-6 flex items-center gap-2 text-sm text-muted"><CheckCircle2 className="h-4 w-4 text-low" />Roadmap items are plans, not commitments, and may change.</p>
          </Reveal>
        </div>
      </Section>
    </>
  );
}
