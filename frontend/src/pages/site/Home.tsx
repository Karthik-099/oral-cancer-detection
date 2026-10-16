import { useCallback, useEffect, useRef, useState, type DragEvent, type ReactNode } from "react";
import { Link } from "react-router-dom";
import { AnimatePresence, motion, useScroll, useTransform } from "framer-motion";
import {
  AlertTriangle, ArrowRight, ArrowUpRight, Building2, Camera, Cloud, Command, FileBarChart2, Fingerprint, GitBranch,
  HardDrive, ImageUp, KeyRound, LineChart, Microscope, RefreshCw, ScanSearch, Server, ShieldCheck, Sparkles,
  Stethoscope, Timer, Trash2, Upload,
} from "lucide-react";
import { Coffee, Github, Star } from "lucide-react";
import { REPO_URL } from "@/lib/links";
import { BuiltWith, CaseStudy, FutureVision, Site3DShowcase } from "@/components/site/HomeSections";
import { api, ApiError, fileUrl, type ImageResult } from "@/lib/api";
import { useAuth } from "@/lib/auth";
import { useSeo } from "@/lib/seo";
import { cn } from "@/lib/utils";
import { Reveal, Tilt } from "@/components/Reveal";
import { Badge, Button, ButtonLink, Kbd, buttonClass } from "@/components/ui";
import { CompareSlider, ProbabilityMeter, QualityBar, RiskBadge } from "@/components/clinical";
import { CTABand, ClinicalNote, FAQ, IconTile, Section, SectionHeading } from "@/components/site/marketing";
import { AnalysisMockup, TissueTile, type TileVariant } from "@/components/site/ProductMockup";

/* ------------------------------------------------------------------ Hero */

function Hero() {
  const user = useAuth((s) => s.user);
  return (
    <section className="relative overflow-hidden">
      <div aria-hidden className="pointer-events-none absolute inset-0 grid-bg [mask-image:radial-gradient(ellipse_80%_70%_at_50%_0%,black,transparent)]" />
      <div aria-hidden className="pointer-events-none absolute -top-48 right-[-10%] h-[520px] w-[520px] rounded-full bg-brand/15 blur-3xl" />
      <div aria-hidden className="pointer-events-none absolute top-40 -left-40 h-80 w-80 rounded-full bg-accent/10 blur-3xl" />
      <div className="container relative grid items-center gap-14 pb-20 pt-14 sm:pt-20 lg:grid-cols-[1.05fr_1fr] lg:gap-16 lg:pb-28 lg:pt-24">
        <motion.div initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.8, ease: [0.2, 0.7, 0.2, 1] }}>
          <Link to="/models" className="group inline-flex items-center gap-2 rounded-full border border-line bg-surface/70 py-1 pl-1 pr-3 text-xs text-muted backdrop-blur transition hover:border-brand/40 hover:text-ink">
            <span className="rounded-full bg-brand/10 px-2 py-0.5 font-medium text-brand">New</span>
            Transparent model cards, including retired models
            <ArrowRight className="h-3 w-3 transition group-hover:translate-x-0.5" />
          </Link>
          <h1 className="display mt-7 text-[3rem] sm:text-7xl xl:text-[5.25rem]" aria-label="Catch oral cancer earlier. See why.">
            <AnimatedWords text="Catch oral cancer earlier." />{" "}
            <AnimatedWords text="See why." delay={0.45} className="text-gradient" />
          </h1>
          <p className="mt-6 max-w-xl text-base leading-relaxed text-muted sm:text-lg">
            OralAID is explainable AI screening support for the dental chair. Upload intraoral photos and get a suspicion score,
            a risk tier and a heatmap of exactly what the model looked at, so you can decide who needs a specialist, faster.
          </p>
          <div className="mt-9 flex flex-wrap gap-3">
            <ButtonLink to={user ? "/app" : "/register"} size="lg">{user ? "Open the app" : "Start free"} <ArrowRight className="h-4 w-4" /></ButtonLink>
            <ButtonLink to="/contact" size="lg" variant="secondary">Book a demo</ButtonLink>
          </div>
          <ul className="mt-10 grid max-w-lg gap-3 text-sm text-muted sm:grid-cols-2">
            {[
              [ScanSearch, "Heatmap on every analysis"],
              [ShieldCheck, "Clinician stays in control"],
              [Server, "Cloud, private VPC or on-prem"],
              [Trash2, "Demo images deleted in 1 hour"],
            ].map(([Icon, label]) => {
              const I = Icon as typeof ScanSearch;
              return <li key={label as string} className="flex items-center gap-2"><I className="h-4 w-4 text-brand" aria-hidden />{label as string}</li>;
            })}
          </ul>
        </motion.div>
        <HeroVisual />
      </div>
    </section>
  );
}

/** Headline words rise into place one after another. */
function AnimatedWords({ text, delay = 0, className }: { text: string; delay?: number; className?: string }) {
  return (
    <span aria-hidden>
      {text.split(" ").map((w, i) => (
        <span key={i} className="inline-block overflow-hidden pb-[0.08em] align-bottom">
          <motion.span className={cn("inline-block", className)} initial={{ y: "110%", rotateX: -60, opacity: 0 }}
            animate={{ y: "0%", rotateX: 0, opacity: 1 }} style={{ transformPerspective: 600 }}
            transition={{ duration: 0.9, delay: delay + i * 0.08, ease: [0.16, 1, 0.3, 1] }}>
            {w}{"\u00a0"}
          </motion.span>
        </span>
      ))}
    </span>
  );
}

/** Product mockup in 3D: orbit rings behind, mouse tilt, and it leans back as you scroll. */
function HeroVisual() {
  const ref = useRef<HTMLDivElement>(null);
  const { scrollYProgress } = useScroll({ target: ref, offset: ["start start", "end start"] });
  const rotateX = useTransform(scrollYProgress, [0, 1], [0, 28]);
  const scale = useTransform(scrollYProgress, [0, 1], [1, 0.86]);
  const y = useTransform(scrollYProgress, [0, 1], [0, 120]);
  const opacity = useTransform(scrollYProgress, [0, 0.9], [1, 0.2]);
  return (
    <motion.div ref={ref} initial={{ opacity: 0, y: 40, rotateX: 20 }} animate={{ opacity: 1, y: 0, rotateX: 0 }}
      transition={{ duration: 1.1, delay: 0.2, ease: [0.16, 1, 0.3, 1] }} style={{ transformPerspective: 1200 }} className="relative">
      <div aria-hidden className="pointer-events-none absolute inset-0 -z-10 grid place-items-center [perspective:900px]">
        {[0, 1, 2].map((i) => (
          <div key={i} className="absolute rounded-full border border-brand/25 shadow-[0_0_40px_rgb(var(--glow-1)/.15)]"
            style={{ width: `${110 + i * 22}%`, aspectRatio: "1", transform: `rotateX(72deg) rotateZ(${i * 30}deg)`,
              animation: `spin-slow ${36 + i * 14}s linear infinite ${i % 2 ? "reverse" : ""}` }} />
        ))}
        <div className="absolute h-2/3 w-2/3 rounded-full bg-[radial-gradient(circle,rgb(var(--glow-3)/.35),transparent_65%)] blur-2xl" />
      </div>
      <motion.div style={{ rotateX, scale, y, opacity, transformPerspective: 1200 }}>
        <Tilt max={7}>
          <AnalysisMockup className="mx-auto max-w-xl" />
        </Tilt>
      </motion.div>
    </motion.div>
  );
}

/* --------------------------------------------------------- Metrics strip */

const METRICS = [
  { value: "0.83", label: "AUC", note: "internal evaluation" },
  { value: "70.5%", label: "Sensitivity", note: "at threshold 0.30" },
  { value: "80.2%", label: "Specificity", note: "at threshold 0.30" },
  { value: "90%", label: "High-tier PPV", note: "score at or above 0.55" },
];

function MetricsStrip() {
  return (
    <section className="border-y border-line bg-surface/60">
      <div className="container py-10">
        <div className="grid grid-cols-2 gap-y-8 lg:grid-cols-[1.2fr_repeat(4,1fr)] lg:items-center">
          <div className="col-span-2 lg:col-span-1 lg:pr-8">
            <p className="eyebrow">Real numbers, honestly framed</p>
            <p className="mt-2 text-sm text-muted">
              OralNet-B4 on 400 specialist-labelled images. <Link to="/models" className="font-medium text-brand hover:underline">How we evaluate</Link>
            </p>
          </div>
          {METRICS.map((m, i) => (
            <Reveal key={m.label} delay={i * 0.06} className="border-l border-line pl-5">
              <p className="display text-4xl tabular-nums sm:text-5xl">{m.value}</p>
              <p className="mt-1 text-sm font-medium">{m.label}</p>
              <p className="text-xs text-muted">{m.note}</p>
            </Reveal>
          ))}
        </div>
        <p className="mt-8 text-xs text-muted">
          Internal evaluation set may overlap training data. Held-out validation accuracy 76.8%. Independent external validation is pending.
        </p>
      </div>
    </section>
  );
}

/* ------------------------------------------------------------- How it works */

const STEPS = [
  { icon: <Camera />, title: "Capture", body: "Upload one to six intraoral photos per case, or capture them with the device camera. Tag the anatomical site on a mouth map." },
  { icon: <ImageUp />, title: "Quality check", body: "Every image is scored for sharpness, exposure and resolution before analysis, so blurry or dark shots are flagged, not silently scored." },
  { icon: <Sparkles />, title: "Analyse and explain", body: "OralNet-B4 returns a suspicion score with uncertainty and a risk tier, plus a Grad-CAM heatmap and an outlined region of interest." },
  { icon: <Stethoscope />, title: "Review and refer", body: "You make the call. Record agreement, clinical impression and histopathology. Suspicious findings come with a referral recommendation." },
];

function HowItWorks() {
  return (
    <Section>
      <SectionHeading eyebrow="How it works" title={<>From photo to <em className="italic text-brand">explained</em> risk tier in seconds.</>}
        body="Designed around the way screening actually happens: a quick look, a decision about referral, and a record you can stand behind." />
      <ol className="relative mt-16 grid gap-10 sm:grid-cols-2 lg:grid-cols-4 lg:gap-8">
        <div aria-hidden className="absolute left-7 right-7 top-7 hidden h-px bg-[linear-gradient(90deg,rgb(var(--glow-1)/.7),rgb(var(--glow-3)/.5),rgb(var(--glow-2)/.7))] lg:block" />
        {STEPS.map((s, i) => (
          <Reveal key={s.title} delay={i * 0.1} className="relative">
            <li className="group">
              <div className="relative grid h-14 w-14 place-items-center rounded-full border border-brand/40 bg-bg text-brand shadow-[0_0_0_6px_rgb(var(--bg)),0_0_30px_rgb(var(--glow-1)/.25)] transition group-hover:shadow-[0_0_0_6px_rgb(var(--bg)),0_0_45px_rgb(var(--glow-1)/.5)]">
                {s.icon}
                <span className="absolute -right-1 -top-1 grid h-5 w-5 place-items-center rounded-full bg-[linear-gradient(135deg,rgb(var(--glow-1)),rgb(var(--glow-2)))] font-mono text-[10px] font-semibold text-white">{i + 1}</span>
              </div>
              <h3 className="mt-6 font-display text-xl font-semibold">{s.title}</h3>
              <p className="mt-2 max-w-xs text-sm leading-relaxed text-muted">{s.body}</p>
            </li>
          </Reveal>
        ))}
      </ol>
    </Section>
  );
}

/* --------------------------------------------------------------- Live demo */

type DemoResult = ImageResult & { id: string };
type DemoState =
  | { phase: "idle" }
  | { phase: "scanning"; preview: string }
  | { phase: "done"; preview: string; result: DemoResult }
  | { phase: "error"; preview?: string; title: string; body: string };

const MAX_BYTES = 15 * 1024 * 1024;

function explainError(err: unknown): { title: string; body: string } {
  if (err instanceof ApiError) {
    switch (err.status) {
      case 422: return { title: "That doesn't look like an intraoral photo", body: `${err.message} Try a well-lit, in-focus photo of the inside of the mouth.` };
      case 429: return { title: "Demo limit reached", body: "The public demo allows a few analyses per hour per visitor. Create a free account to keep going." };
      case 503: return { title: "The AI service is offline", body: "The analysis server is starting up or under maintenance. Please try again in a minute." };
      case 413: return { title: "Image too large", body: "Please upload an image under 15 MB." };
      case 415: return { title: "Unsupported file", body: "Use a JPEG, PNG, WebP or HEIC image." };
      case 0: return { title: "Can't reach the service", body: err.message };
      default: return { title: "Analysis failed", body: err.message || "Something went wrong. Please try again." };
    }
  }
  return { title: "Analysis failed", body: (err as Error)?.message ?? "Something went wrong." };
}

function LiveDemo() {
  const [state, setState] = useState<DemoState>({ phase: "idle" });
  const [overlay, setOverlay] = useState<"gradcam" | "segmentation">("gradcam");
  const [drag, setDrag] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);
  const previewRef = useRef<string | null>(null);

  useEffect(() => () => { if (previewRef.current) URL.revokeObjectURL(previewRef.current); }, []);

  const run = useCallback(async (file: File) => {
    if (previewRef.current) URL.revokeObjectURL(previewRef.current);
    const preview = URL.createObjectURL(file);
    previewRef.current = preview;
    if (file.type && !file.type.startsWith("image/")) {
      setState({ phase: "error", title: "Unsupported file", body: "Please choose an image file (JPEG, PNG, WebP or HEIC)." });
      return;
    }
    if (file.size > MAX_BYTES) {
      setState({ phase: "error", title: "Image too large", body: "Please upload an image under 15 MB." });
      return;
    }
    setOverlay("gradcam");
    setState({ phase: "scanning", preview });
    try {
      const result = await api.demo(file);
      setState({ phase: "done", preview, result });
    } catch (err) {
      setState({ phase: "error", preview, ...explainError(err) });
    }
  }, []);

  const onDrop = (e: DragEvent) => {
    e.preventDefault();
    setDrag(false);
    const f = e.dataTransfer.files?.[0];
    if (f) run(f);
  };

  const reset = () => {
    setState({ phase: "idle" });
    if (inputRef.current) inputRef.current.value = "";
  };

  return (
    <Section id="demo" className="relative">
      <div aria-hidden className="pointer-events-none absolute inset-x-0 top-0 h-full bg-gradient-to-b from-brand-soft/40 to-transparent dark:from-brand-soft/20" />
      <div className="relative grid gap-12 lg:grid-cols-[0.85fr_1.15fr] lg:gap-16">
        <div>
          <SectionHeading eyebrow="Live demo" title={<>Try it on a photo. <em className="italic text-brand">Right here.</em></>}
            body="Upload an intraoral photo and run the same production pipeline clinicians use: quality check, OralNet-B4 with test-time augmentation, Grad-CAM heatmap and region outline." />
          <Reveal delay={0.1}>
            <ul className="mt-8 space-y-3 text-sm text-muted">
              <li className="flex gap-2.5"><Timer className="mt-0.5 h-4 w-4 shrink-0 text-brand" />Demo images and results are deleted automatically after one hour.</li>
              <li className="flex gap-2.5"><ShieldCheck className="mt-0.5 h-4 w-4 shrink-0 text-brand" />Never used for training. No account needed.</li>
              <li className="flex gap-2.5"><AlertTriangle className="mt-0.5 h-4 w-4 shrink-0 text-moderate" />Please do not upload images that identify a person, such as full faces.</li>
            </ul>
            <ClinicalNote className="mt-8 max-w-md" />
          </Reveal>
        </div>

        <Reveal delay={0.05}>
          <div className="card overflow-hidden p-2 shadow-lift">
            <input ref={inputRef} type="file" accept="image/jpeg,image/png,image/webp,image/heic,image/*" className="sr-only" id="demo-file"
              onChange={(e) => { const f = e.target.files?.[0]; if (f) run(f); }} />
            <AnimatePresence mode="wait">
              {state.phase === "idle" && (
                <motion.label key="idle" htmlFor="demo-file" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
                  onDragOver={(e) => { e.preventDefault(); setDrag(true); }} onDragLeave={() => setDrag(false)} onDrop={onDrop}
                  className={cn("flex min-h-[420px] cursor-pointer flex-col items-center justify-center gap-4 rounded-xl border-2 border-dashed p-8 text-center transition",
                    drag ? "border-brand bg-brand/5" : "border-line hover:border-brand/50 hover:bg-surface-2/50")}>
                  <div className="grid h-14 w-14 place-items-center rounded-2xl bg-brand/10 text-brand"><Upload className="h-6 w-6" /></div>
                  <div>
                    <p className="text-base font-semibold">Drop an intraoral photo, or click to browse</p>
                    <p className="mt-1 text-sm text-muted">JPEG, PNG, WebP or HEIC up to 15 MB. On mobile you can use the camera.</p>
                  </div>
                  <span className={cn("mt-2", "inline-flex h-10 items-center gap-2 rounded-xl bg-ink px-4 text-sm font-medium text-bg")}>
                    <ImageUp className="h-4 w-4" /> Choose image
                  </span>
                </motion.label>
              )}

              {state.phase === "scanning" && (
                <motion.div key="scan" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} className="grid gap-4 p-2 sm:grid-cols-[1fr_0.9fr]" aria-live="polite">
                  <div className="relative aspect-square overflow-hidden rounded-xl bg-surface-2">
                    <img src={state.preview} alt="Uploaded photo being analysed" className="h-full w-full object-cover" />
                    <div className="absolute inset-0 bg-brand/10" />
                    <div aria-hidden className="absolute inset-0 grid-bg opacity-40" />
                    <div aria-hidden className="absolute inset-x-0 h-1/3 animate-scan bg-gradient-to-b from-transparent via-brand/30 to-transparent">
                      <div className="absolute inset-x-0 bottom-1/2 h-0.5 bg-brand shadow-[0_0_16px_4px_rgb(var(--brand)/.6)]" />
                    </div>
                  </div>
                  <div className="flex flex-col justify-center gap-4 p-2">
                    <p className="eyebrow text-brand">Analysing</p>
                    {["Checking image quality", "Running OralNet-B4 with test-time augmentation", "Computing Grad-CAM and region outline"].map((t, i) => (
                      <motion.div key={t} initial={{ opacity: 0, x: -6 }} animate={{ opacity: 1, x: 0 }} transition={{ delay: i * 0.6 }} className="flex items-center gap-3 text-sm">
                        <span className="relative flex h-2.5 w-2.5">
                          <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-brand/50" />
                          <span className="relative inline-flex h-2.5 w-2.5 rounded-full bg-brand" />
                        </span>
                        {t}
                      </motion.div>
                    ))}
                    <div className="mt-2 space-y-2"><div className="skeleton h-3 w-3/4" /><div className="skeleton h-3 w-1/2" /><div className="skeleton h-16 w-full" /></div>
                  </div>
                </motion.div>
              )}

              {state.phase === "error" && (
                <motion.div key="err" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} role="alert"
                  className="flex min-h-[420px] flex-col items-center justify-center gap-4 p-8 text-center">
                  {state.preview && <img src={state.preview} alt="" className="h-24 w-24 rounded-xl object-cover opacity-60 grayscale" />}
                  <div className="grid h-11 w-11 place-items-center rounded-full bg-high/10 text-high"><AlertTriangle className="h-5 w-5" /></div>
                  <div className="max-w-sm">
                    <p className="text-base font-semibold">{state.title}</p>
                    <p className="mt-1.5 text-sm text-muted">{state.body}</p>
                  </div>
                  <div className="flex flex-wrap justify-center gap-2">
                    <Button variant="ink" onClick={reset}><RefreshCw className="h-4 w-4" /> Try another image</Button>
                    <ButtonLink to="/register" variant="secondary">Create free account</ButtonLink>
                  </div>
                </motion.div>
              )}

              {state.phase === "done" && <DemoResultView key="done" result={state.result} overlay={overlay} setOverlay={setOverlay} onReset={reset} />}
            </AnimatePresence>
          </div>
        </Reveal>
      </div>
    </Section>
  );
}

function DemoResultView({ result, overlay, setOverlay, onReset }: {
  result: DemoResult; overlay: "gradcam" | "segmentation"; setOverlay: (o: "gradcam" | "segmentation") => void; onReset: () => void;
}) {
  return (
    <motion.div initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0 }} className="grid gap-5 p-2 md:grid-cols-[1fr_0.95fr]" aria-live="polite">
      <div className="space-y-3">
        <CompareSlider before={fileUrl(result.files.original)} after={fileUrl(result.files[overlay])}
          labelBefore="Original" labelAfter={overlay === "gradcam" ? "Grad-CAM" : "Region"} />
        <div className="flex rounded-xl bg-surface-2 p-1 text-xs font-medium" role="group" aria-label="Overlay">
          {(["gradcam", "segmentation"] as const).map((o) => (
            <button key={o} type="button" onClick={() => setOverlay(o)} aria-pressed={overlay === o}
              className={cn("flex-1 rounded-lg px-3 py-1.5 transition", overlay === o ? "bg-surface text-ink shadow-soft" : "text-muted hover:text-ink")}>
              {o === "gradcam" ? "Grad-CAM heatmap" : "Region outline"}
            </button>
          ))}
        </div>
      </div>
      <div className="flex flex-col p-1">
        <div className="flex items-center justify-between gap-2">
          <RiskBadge tier={result.tier} />
          <span className="font-mono text-[11px] text-muted">{Math.round(result.inference_ms)} ms</span>
        </div>
        <div className="mt-3 flex justify-center">
          <ProbabilityMeter value={result.probability} threshold={0.3} high={0.55} size={200} />
        </div>
        <p className="mt-2 text-center text-xs text-muted">Uncertainty ± {result.uncertainty.toFixed(2)}</p>
        <div className="mt-5 space-y-4 text-sm">
          <div>
            <p className="mb-1.5 text-xs text-muted">Image quality</p>
            <QualityBar score={Math.round(result.quality.score)} />
            {result.quality.issues.length > 0 && <p className="mt-1.5 text-xs text-moderate">{result.quality.issues.join(" · ")}</p>}
          </div>
          <div className="border-t border-line pt-4">
            <p className="text-xs text-muted">Finding</p>
            <p className="mt-1 font-medium">{result.finding}</p>
          </div>
          <div className="rounded-xl border border-line bg-surface-2/60 p-3 text-[13px] leading-relaxed">
            <span className="font-medium">Recommendation. </span>{result.recommendation}
          </div>
        </div>
        <div className="mt-5 flex flex-wrap gap-2">
          <Button variant="secondary" size="sm" onClick={onReset}><RefreshCw className="h-3.5 w-3.5" /> Analyse another</Button>
          <ButtonLink to="/register" size="sm" variant="ink">Get LIME, reports and timelines <ArrowRight className="h-3.5 w-3.5" /></ButtonLink>
        </div>
      </div>
    </motion.div>
  );
}

/* --------------------------------------------------------- Explainability */

const EXPLAIN: { id: TileVariant; name: string; tag: string; body: string; detail: string }[] = [
  { id: "gradcam", name: "Grad-CAM", tag: "Where it looked", body: "A heatmap of the image areas that most influenced the score, computed from the final EfficientNet stage.",
    detail: "If the heat lands on a finger, a retractor or the background, you know to distrust the call and crop or retake." },
  { id: "region", name: "Region Localizer", tag: "What it outlined", body: "Grad-CAM seeds refined with GrabCut into a weakly supervised outline of the tissue that drove the prediction.",
    detail: "It is an attention outline, not a pixel-trained lesion boundary. We say so in the product and on the model card." },
  { id: "lime", name: "LIME superpixels", tag: "Why it decided", body: "On demand, the image is split into superpixels and each is scored by how much it pushes the prediction.",
    detail: "Red regions push toward suspicious; green regions push toward non-suspicious. A second, independent view on the same decision." },
];

function Explainability() {
  const [active, setActive] = useState(0);
  const item = EXPLAIN[active];
  return (
    <Section className="border-y border-line bg-surface/50">
      <div className="grid items-center gap-12 lg:grid-cols-2 lg:gap-20">
        <div>
          <SectionHeading eyebrow="Explainability" title={<>A score is not enough. <em className="italic text-brand">Show the evidence.</em></>}
            body="Every analysis ships with visual explanations, so clinicians can check whether the model is looking at tissue or at a shortcut." />
          <div className="mt-10 space-y-2" role="tablist" aria-label="Explanation methods">
            {EXPLAIN.map((e, i) => (
              <button key={e.id} role="tab" aria-selected={i === active} onClick={() => setActive(i)}
                className={cn("w-full rounded-2xl border p-5 text-left transition",
                  i === active ? "border-brand/30 bg-surface shadow-soft" : "border-transparent hover:bg-surface/70")}>
                <div className="flex items-center justify-between gap-3">
                  <span className="text-base font-semibold">{e.name}</span>
                  <span className={cn("text-xs", i === active ? "text-brand" : "text-muted")}>{e.tag}</span>
                </div>
                <AnimatePresence initial={false}>
                  {i === active && (
                    <motion.div initial={{ height: 0, opacity: 0 }} animate={{ height: "auto", opacity: 1 }} exit={{ height: 0, opacity: 0 }} className="overflow-hidden">
                      <p className="pt-2 text-sm leading-relaxed text-muted">{e.body}</p>
                      <p className="pt-2 text-sm leading-relaxed text-ink/80">{e.detail}</p>
                    </motion.div>
                  )}
                </AnimatePresence>
              </button>
            ))}
          </div>
        </div>
        <Reveal>
          <div className="relative mx-auto max-w-md">
            <div aria-hidden className="absolute -inset-6 rounded-[36px] bg-gradient-to-br from-brand/15 via-transparent to-accent/10 blur-2xl" />
            <div className="card relative p-3 shadow-lift">
              <AnimatePresence mode="wait">
                <motion.div key={item.id} initial={{ opacity: 0, scale: 0.98 }} animate={{ opacity: 1, scale: 1 }} exit={{ opacity: 0 }} transition={{ duration: 0.35 }}>
                  <TissueTile variant={item.id} className="rounded-2xl" />
                </motion.div>
              </AnimatePresence>
              <div className="flex items-center justify-between px-2 pb-1 pt-3 text-xs text-muted">
                <span className="font-medium text-ink">{item.name}</span>
                {item.id === "lime" ? (
                  <span className="flex items-center gap-3">
                    <span className="flex items-center gap-1"><span className="h-2 w-2 rounded-sm bg-high" />Suspicious</span>
                    <span className="flex items-center gap-1"><span className="h-2 w-2 rounded-sm bg-low" />Non-suspicious</span>
                  </span>
                ) : <span className="font-mono">Illustrative</span>}
              </div>
            </div>
          </div>
        </Reveal>
      </div>
    </Section>
  );
}

/* ------------------------------------------------------------ Feature bento */

function BentoCard({ className, icon, title, body, children, delay = 0 }: {
  className?: string; icon: ReactNode; title: string; body: string; children?: ReactNode; delay?: number;
}) {
  return (
    <Reveal delay={delay} className={cn("card spotlight group relative flex flex-col overflow-hidden p-6 transition hover:shadow-lift", className)}>
      <IconTile>{icon}</IconTile>
      <h3 className="mt-5 text-base font-semibold">{title}</h3>
      <p className="mt-2 text-sm leading-relaxed text-muted">{body}</p>
      {children && <div className="mt-6 flex-1">{children}</div>}
    </Reveal>
  );
}

function Features() {
  return (
    <Section>
      <SectionHeading eyebrow="Platform" title={<>Everything a screening workflow needs. <em className="italic text-brand">Nothing it doesn't.</em></>} />
      <div className="mt-14 grid gap-4 sm:grid-cols-2 lg:grid-cols-6">
        <BentoCard className="lg:col-span-4" icon={<ImageUp />} title="Multi-image cases with a quality gate"
          body="Up to six photos per case. Each is scored for sharpness, exposure and resolution, and the case-level result combines them. A crop tool lets you focus the model on the lesion.">
          <div className="grid grid-cols-3 gap-3">
            {[{ v: "original" as const, q: 91 }, { v: "gradcam" as const, q: 78 }, { v: "region" as const, q: 52 }].map((t) => (
              <div key={t.v} className="space-y-2">
                <TissueTile variant={t.v} className="rounded-lg" />
                <QualityBar score={t.q} />
              </div>
            ))}
          </div>
        </BentoCard>
        <BentoCard className="lg:col-span-2" icon={<ScanSearch />} title="Explainable by default" delay={0.05}
          body="Grad-CAM and a region outline on every image. LIME superpixels on demand.">
          <div className="grid grid-cols-2 gap-2"><TissueTile variant="gradcam" className="rounded-lg" /><TissueTile variant="lime" className="rounded-lg" /></div>
        </BentoCard>
        <BentoCard className="lg:col-span-2" icon={<LineChart />} title="Patient timelines" delay={0.05}
          body="Follow a lesion across visits and see when the score starts moving.">
          <svg viewBox="0 0 200 60" className="h-16 w-full" aria-hidden>
            <polyline points="4,50 50,46 96,38 142,26 196,12" fill="none" className="stroke-brand" strokeWidth="2" />
            {[[4, 50], [50, 46], [96, 38], [142, 26], [196, 12]].map(([x, y]) => <circle key={x} cx={x} cy={y} r="3" className="fill-surface stroke-brand" strokeWidth="2" />)}
          </svg>
        </BentoCard>
        <BentoCard className="lg:col-span-2" icon={<FileBarChart2 />} title="Cases, reports and analytics" delay={0.1}
          body="Open, referred and closed queues, printable reports, CSV export and a dashboard of risk mix and agreement rate." />
        <BentoCard className="lg:col-span-2" icon={<Command />} title="Fast by keyboard" delay={0.15}
          body="A command palette for jumping to any case, patient or action.">
          <div className="flex items-center gap-2 rounded-xl border border-line bg-surface-2/60 px-3 py-2 text-xs text-muted">
            <Kbd>⌘</Kbd><Kbd>K</Kbd><span className="truncate">Search cases, patients, actions</span>
          </div>
        </BentoCard>
        <BentoCard className="lg:col-span-3" icon={<KeyRound />} title="API and integrations"
          body="A documented REST API with per-integration API keys, so your EHR, imaging or research pipeline can submit cases and read results.">
          <code className="block overflow-x-auto rounded-xl border border-line bg-surface-2/60 px-3 py-2.5 font-mono text-[11.5px] text-muted">
            POST /api/cases &nbsp;X-API-Key: oak_...
          </code>
        </BentoCard>
        <BentoCard className="lg:col-span-3" icon={<Fingerprint />} title="Security you can audit" delay={0.05}
          body="Signed, expiring image URLs, an audit log of every access, account lockout, rate limiting and sign-out-everywhere.">
          <div className="flex flex-wrap gap-2">
            {["Audit log", "Signed URLs", "bcrypt", "Lockout", "Rate limits"].map((t) => <Badge key={t}>{t}</Badge>)}
          </div>
        </BentoCard>
      </div>
      <Reveal className="mt-8 text-center">
        <Link to="/platform" className="inline-flex items-center gap-1.5 text-sm font-medium text-brand hover:underline">Explore the platform <ArrowRight className="h-4 w-4" /></Link>
      </Reveal>
    </Section>
  );
}

/* --------------------------------------------------------- Deployment teaser */

function DeploymentTeaser() {
  const opts = [
    { id: "cloud", icon: <Cloud />, title: "Cloud SaaS", body: "Sign up and start screening in minutes. Managed updates, encrypted storage, nothing to install." },
    { id: "vpc", icon: <Building2 />, title: "Private VPC", body: "Single-tenant deployment in your own AWS, GCP or Azure account and region. Your network, your keys." },
    { id: "onprem", icon: <HardDrive />, title: "On-premise", body: "Docker images that run on a clinic server, CPU only, fully air-gapped. No image ever leaves the building." },
  ];
  return (
    <Section>
      <div className="flex flex-col justify-between gap-6 lg:flex-row lg:items-end">
        <SectionHeading eyebrow="Deployment" title={<>Runs where your <em className="italic text-brand">data</em> has to stay.</>} />
        <Reveal><ButtonLink to="/deployment" variant="secondary">Compare deployment options <ArrowRight className="h-4 w-4" /></ButtonLink></Reveal>
      </div>
      <div className="mt-12 grid gap-4 md:grid-cols-3">
        {opts.map((o, i) => (
          <Reveal key={o.id} delay={i * 0.06}>
            <Link to={`/deployment#${o.id}`} className="card group flex h-full flex-col p-6 transition hover:-translate-y-0.5 hover:border-brand/30 hover:shadow-lift">
              <div className="flex items-center justify-between">
                <IconTile>{o.icon}</IconTile>
                <ArrowUpRight className="h-4 w-4 text-muted transition group-hover:-translate-y-0.5 group-hover:translate-x-0.5 group-hover:text-brand" />
              </div>
              <h3 className="mt-5 text-lg font-semibold">{o.title}</h3>
              <p className="mt-2 text-sm leading-relaxed text-muted">{o.body}</p>
            </Link>
          </Reveal>
        ))}
      </div>
    </Section>
  );
}

/* ---------------------------------------------------------------- Mission */

function Mission() {
  return (
    <Section className="border-y border-line bg-surface/50">
      <Reveal className="mx-auto max-w-4xl text-center">
        <p className="eyebrow text-brand">Our mission</p>
        <p className="display mt-6 text-4xl sm:text-6xl">
          Make expert-level oral cancer screening available in <em className="italic text-brand">every chair</em>, and make every call explainable.
        </p>
        <div className="mt-10 flex flex-wrap justify-center gap-3">
          {[[Microscope, "Transparent"], [Stethoscope, "Clinician-first"], [ShieldCheck, "Private by design"], [GitBranch, "Built for low-resource settings"]].map(([Icon, l]) => {
            const I = Icon as typeof Microscope;
            return <span key={l as string} className="inline-flex items-center gap-2 rounded-full border border-line bg-surface px-3.5 py-1.5 text-sm"><I className="h-4 w-4 text-brand" />{l as string}</span>;
          })}
        </div>
        <Link to="/about" className="mt-8 inline-flex items-center gap-1.5 text-sm font-medium text-brand hover:underline">Read about us <ArrowRight className="h-4 w-4" /></Link>
      </Reveal>
    </Section>
  );
}

/* -------------------------------------------------------------------- FAQ */

const FAQS = [
  { q: "Is OralAID a diagnostic device?", a: "No. OralAID is clinical decision support for screening. It estimates how suspicious an image looks and shows why, but it does not diagnose and is not cleared as a medical device in any jurisdiction. Diagnosis requires clinical examination and, for suspicious lesions, biopsy." },
  { q: "How accurate is it?", a: <>On an internal evaluation of 400 specialist-labelled images, OralNet-B4 reached an AUC of 0.83, sensitivity of 70.5% and specificity of 80.2% at the 0.30 threshold, and 90% positive predictive value in the high-risk tier. That set may overlap training data, and independent external validation is pending. Full details are on the <Link to="/models" className="text-brand underline">models page</Link>.</> },
  { q: "What do the risk tiers mean?", a: "The model outputs a probability that the image is suspicious. Below 0.30 is Low, 0.30 to 0.55 is Moderate, and 0.55 or above is High. Each tier comes with a recommendation, and suspicious findings are always paired with a referral suggestion." },
  { q: "What images work best?", a: "Well-lit, in-focus intraoral photos where the area of concern fills a good part of the frame. Each image is quality-checked for sharpness, exposure and resolution. Fingers, retractors and background can draw the model's attention, which is why we show heatmaps and offer a crop tool." },
  { q: "Do you train on my images?", a: "Not without your explicit opt-in. Images from the public demo are deleted after one hour and are never used for training. Customer data is isolated per tenant and accessed through signed, expiring URLs." },
  { q: "Can it run without an internet connection?", a: <>Yes. The on-premise edition runs in Docker on a clinic server or workstation, CPU only, fully air-gapped. A Windows desktop app is on the <Link to="/roadmap#desktop" className="text-brand underline">roadmap</Link>.</> },
  { q: "Will it work for my patient population?", a: "Performance can shift with population, camera and lighting. We recommend local validation before clinical use, and our Enterprise plan includes a custom validation study on your own labelled data." },
];

/* ------------------------------------------------------------ Open source */

function OpenSource() {
  return (
    <Section id="open-source">
      <Reveal>
        <div className="card glow-border relative overflow-hidden p-8 sm:p-12">
          <div aria-hidden className="pointer-events-none absolute -right-24 -top-24 h-72 w-72 rounded-full bg-[radial-gradient(circle,rgb(var(--glow-2)/.25),transparent_65%)] blur-2xl" />
          <div className="relative grid items-center gap-10 lg:grid-cols-[auto_1fr_auto]">
            <div className="grid h-24 w-24 place-items-center rounded-3xl border border-line bg-surface-2 shadow-[0_0_60px_rgb(var(--glow-1)/.2)]">
              <Github className="h-12 w-12" />
            </div>
            <div>
              <p className="eyebrow mb-2 text-brand">Open source, MIT licensed</p>
              <h2 className="display text-3xl sm:text-5xl">Built in the open. <em>Support it.</em></h2>
              <p className="mt-3 max-w-xl text-muted">
                Every line of OralAID, from the model pipeline to this website, is public. Star the project, report issues,
                contribute, or buy us a coffee to keep it free.
              </p>
              <a href={REPO_URL} target="_blank" rel="noreferrer" className="mt-4 inline-block break-all font-mono text-sm text-brand hover:underline">
                github.com/Karthik-099/oral-cancer-detection
              </a>
            </div>
            <div className="flex flex-col gap-3 sm:flex-row lg:flex-col">
              <a href={REPO_URL} target="_blank" rel="noreferrer" className={buttonClass("ink", "lg")}><Star className="h-4 w-4" />Star on GitHub</a>
              <ButtonLink to="/support" size="lg"><Coffee className="h-4 w-4" />Buy us a coffee</ButtonLink>
            </div>
          </div>
        </div>
      </Reveal>
    </Section>
  );
}

export default function Home() {
  useSeo("OralAID: explainable AI for oral cancer screening",
    "Upload intraoral photos and get a suspicion score, risk tier and heatmap of what the model saw. Clinical decision support for earlier referral.");
  return (
    <>
      <Hero />
      <BuiltWith />
      <MetricsStrip />
      <HowItWorks />
      <LiveDemo />
      <Explainability />
      <Site3DShowcase />
      <Features />
      <CaseStudy />
      <DeploymentTeaser />
      <FutureVision />
      <Mission />
      <Section>
        <div className="grid gap-12 lg:grid-cols-[0.8fr_1.2fr]">
          <SectionHeading eyebrow="FAQ" title={<>Questions, <em className="italic text-brand">answered plainly.</em></>}
            body={<>Something else? <Link to="/contact" className="text-brand underline">Talk to us</Link>.</>} />
          <Reveal><FAQ items={FAQS} /></Reveal>
        </div>
      </Section>
      <OpenSource />
      <CTABand />
    </>
  );
}
