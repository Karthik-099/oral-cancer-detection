import { useEffect, useState, type ReactNode } from "react";
import { Link } from "react-router-dom";
import { motion } from "framer-motion";
import {
  Archive, ArrowRight, Boxes, Camera, CircleDot, Cpu, FlaskConical, Hand, Layers, Lightbulb, Microscope, ScanSearch, Sparkles, SunMedium, Users, WifiOff,
} from "lucide-react";
import { api, type ModelCard } from "@/lib/api";
import { useSeo } from "@/lib/seo";
import { cn, pct } from "@/lib/utils";
import { Reveal } from "@/components/Reveal";
import { Badge } from "@/components/ui";
import { CTABand, ClinicalNote, IconTile, PageHero, Section, SectionHeading } from "@/components/site/marketing";

/** Mirrors the backend registry so the page still works if the API is unreachable. */
const STATIC_MODELS: ModelCard[] = [
  {
    id: "oralnet-b4", name: "OralNet-B4", task: "classification", status: "production",
    architecture: "EfficientNet-B4 backbone, 3-layer MLP head (BatchNorm + dropout)",
    input: "300×300 and 380×380 RGB (multi-scale TTA with flips)", classes: ["Non-suspicious", "Suspicious"], params_m: 18.4,
    metrics: {
      val_accuracy: 0.768, auc: 0.826, sensitivity: 0.705, specificity: 0.802, ppv: 0.768, npv: 0.744, high_risk_ppv: 0.9,
      eval_set: "val_accuracy: held-out split at training time. Other metrics: internal evaluation on 400 specialist-labelled images at threshold 0.30, which may overlap training data. Independent external validation is pending.",
    },
    notes: "Binary screening model (suspicious vs non-suspicious). Not a diagnostic device.",
  },
  {
    id: "cam-grabcut", name: "Region Localizer", task: "segmentation", status: "production",
    architecture: "Grad-CAM seeds → GrabCut graph-cut refinement → peak-filtered components",
    input: "512×512 RGB + OralNet-B4 activation map", classes: ["Background", "Region of interest"], metrics: {},
    notes: "Weakly supervised: outlines the tissue that drove the prediction. Not a pixel-trained lesion mask.",
  },
  {
    id: "explain-gradcam", name: "Grad-CAM", task: "explainability", status: "production",
    architecture: "Gradient-weighted class activation on the final EfficientNet stage", input: "Shares OralNet-B4", classes: [], metrics: {},
    notes: "Shows where the classifier looked when making its call.",
  },
  {
    id: "explain-lime", name: "LIME superpixels", task: "explainability", status: "production",
    architecture: "SLIC superpixels + kernel-weighted ridge surrogate", input: "Shares OralNet-B4", classes: [], metrics: {},
    notes: "Ranks image regions by how much they push the prediction toward 'Suspicious'.",
  },
  {
    id: "lesionseg-r34", name: "U-Net R34 (legacy)", task: "segmentation", status: "retired",
    architecture: "U-Net decoder, ResNet-34 encoder", input: "512×512 RGB", classes: ["Background", "Lesion"], params_m: 24.4, metrics: {},
    notes: "Trained on whole-image pseudo-masks, so it marks the entire photo. Needs pixel-level annotations to retrain.",
  },
  {
    id: "mobilenet-v2-legacy", name: "MobileNetV2 (legacy)", task: "classification", status: "retired",
    architecture: "MobileNetV2, linear head", input: "224×224 RGB", classes: ["Non-suspicious", "Suspicious"], params_m: 2.2,
    metrics: { auc: 0.41, eval_set: "same held-out set" },
    notes: "Failed validation (below chance). Kept for audit only; never served.",
  },
];

const METRIC_LABELS: Record<string, string> = {
  val_accuracy: "Validation accuracy",
  auc: "AUC",
  sensitivity: "Sensitivity",
  specificity: "Specificity",
  ppv: "PPV",
  npv: "NPV",
  high_risk_ppv: "High-tier PPV",
  accuracy: "Accuracy",
  dice: "Dice",
  iou: "IoU",
};

const TASK_ICON: Record<string, ReactNode> = {
  classification: <Sparkles />,
  segmentation: <Layers />,
  explainability: <ScanSearch />,
};

function fmtMetric(key: string, v: number) {
  return key === "auc" ? v.toFixed(2) : pct(v, 1);
}

function MetricBar({ label, value, k, tone = "brand" }: { label: string; value: number; k: string; tone?: "brand" | "high" }) {
  return (
    <div>
      <div className="flex items-baseline justify-between gap-3 text-xs">
        <span className="text-muted">{label}</span>
        <span className="font-mono tabular-nums text-ink">{fmtMetric(k, value)}</span>
      </div>
      <div className="mt-1.5 h-1.5 overflow-hidden rounded-full bg-surface-2">
        <motion.div className={cn("h-full rounded-full", tone === "high" ? "bg-high/70" : "bg-brand")}
          initial={{ width: 0 }} whileInView={{ width: `${Math.max(0, Math.min(1, value)) * 100}%` }} viewport={{ once: true }}
          transition={{ duration: 0.9, ease: [0.2, 0.7, 0.2, 1] }} />
      </div>
    </div>
  );
}

function StatusBadge({ status }: { status: ModelCard["status"] }) {
  if (status === "production") return <Badge tone="low"><CircleDot className="h-3 w-3" />Production</Badge>;
  if (status === "beta") return <Badge tone="moderate">Beta</Badge>;
  return <Badge tone="neutral"><Archive className="h-3 w-3" />Retired</Badge>;
}

function ModelCardView({ m, index }: { m: ModelCard; index: number }) {
  const numeric = Object.entries(m.metrics).filter((e): e is [string, number] => typeof e[1] === "number");
  const evalNote = typeof m.metrics.eval_set === "string" ? m.metrics.eval_set : null;
  const retired = m.status === "retired";
  const featured = m.id === "oralnet-b4";
  return (
    <Reveal delay={index * 0.05} className={cn("card flex flex-col p-6 sm:p-7", featured && "lg:col-span-2", retired && "bg-surface-2/40 shadow-none")}>
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div className="flex items-center gap-3">
          <IconTile tone={retired ? "neutral" : "brand"}>{TASK_ICON[m.task] ?? <Boxes />}</IconTile>
          <div>
            <h3 className={cn("text-lg font-semibold", retired && "text-muted")}>{m.name}</h3>
            <p className="text-xs capitalize text-muted">{m.task}{m.params_m ? ` · ${m.params_m}M params` : ""}</p>
          </div>
        </div>
        <StatusBadge status={m.status} />
      </div>

      <div className={cn("mt-6 grid gap-6", featured && numeric.length > 0 && "lg:grid-cols-[1fr_1.1fr] lg:gap-10")}>
        <dl className="space-y-3 text-sm">
          <div><dt className="text-xs text-muted">Architecture</dt><dd className="mt-0.5">{m.architecture}</dd></div>
          <div><dt className="text-xs text-muted">Input</dt><dd className="mt-0.5">{m.input}</dd></div>
          {m.classes.length > 0 && <div><dt className="text-xs text-muted">Output</dt><dd className="mt-1 flex flex-wrap gap-1.5">{m.classes.map((c) => <Badge key={c}>{c}</Badge>)}</dd></div>}
          {m.runtime && <div><dt className="text-xs text-muted">Runtime</dt><dd className="mt-0.5 font-mono text-xs">{m.runtime}</dd></div>}
        </dl>
        {numeric.length > 0 && (
          <div className="space-y-3.5">
            {numeric.map(([k, v]) => <MetricBar key={k} k={k} label={METRIC_LABELS[k] ?? k} value={v} tone={retired ? "high" : "brand"} />)}
          </div>
        )}
      </div>

      <div className={cn("mt-6 rounded-xl border p-3.5 text-sm leading-relaxed", retired ? "border-high/20 bg-high/5" : "border-line bg-surface-2/50")}>
        {retired && <span className="font-medium text-high">Why it was retired. </span>}
        {m.notes}
      </div>
      {evalNote && <p className="mt-3 text-xs leading-relaxed text-muted">{evalNote}</p>}
    </Reveal>
  );
}

function TierScale({ threshold, high }: { threshold: number; high: number }) {
  const tiers = [
    { label: "Low", range: `< ${threshold.toFixed(2)}`, w: threshold, cls: "bg-low", text: "text-low", body: "Routine recall. Re-screen at the next scheduled visit, or sooner if symptoms change." },
    { label: "Moderate", range: `${threshold.toFixed(2)} – ${high.toFixed(2)}`, w: high - threshold, cls: "bg-moderate", text: "text-moderate", body: "Possible suspicious features. Clinical re-examination within 2 to 3 weeks; consider specialist review if the area persists." },
    { label: "High", range: `≥ ${high.toFixed(2)}`, w: 1 - high, cls: "bg-high", text: "text-high", body: "Suspicious lesion features. Refer to an oral medicine or maxillofacial specialist for evaluation and possible biopsy." },
  ];
  return (
    <div>
      <div className="flex h-4 overflow-hidden rounded-full">
        {tiers.map((t, i) => (
          <motion.div key={t.label} className={cn(t.cls, "h-full origin-left")} style={{ width: `${t.w * 100}%` }}
            initial={{ opacity: 0, scaleX: 0 }} whileInView={{ opacity: 1, scaleX: 1 }} viewport={{ once: true }}
            transition={{ delay: i * 0.15, duration: 0.6 }} />
        ))}
      </div>
      <div className="relative mt-2 h-5 font-mono text-[11px] text-muted">
        <span className="absolute left-0">0</span>
        <span className="absolute -translate-x-1/2" style={{ left: `${threshold * 100}%` }}>{threshold.toFixed(2)}</span>
        <span className="absolute -translate-x-1/2" style={{ left: `${high * 100}%` }}>{high.toFixed(2)}</span>
        <span className="absolute right-0">1</span>
      </div>
      <div className="mt-8 grid gap-4 md:grid-cols-3">
        {tiers.map((t) => (
          <div key={t.label} className="card p-5">
            <div className="flex items-center justify-between">
              <span className={cn("flex items-center gap-2 text-sm font-semibold", t.text)}><span className={cn("h-2 w-2 rounded-full", t.cls)} />{t.label} risk</span>
              <span className="font-mono text-xs text-muted">{t.range}</span>
            </div>
            <p className="mt-3 text-sm text-muted">{t.body}</p>
          </div>
        ))}
      </div>
    </div>
  );
}

const LIMITATIONS = [
  { icon: <Hand />, t: "Shortcut learning", d: "Fingers, retractors, lips or background can draw the model's attention. That is exactly why every result shows a heatmap and the app offers a crop-to-region tool." },
  { icon: <SunMedium />, t: "Lighting and focus", d: "Dim, overexposed or blurry photos degrade accuracy. The quality gate flags these, but cannot fix them. Retake when in doubt." },
  { icon: <Camera />, t: "Device variation", d: "Phone cameras, intraoral cameras and DSLRs differ in colour and optics. Performance on an unseen device may differ from our evaluation." },
  { icon: <Users />, t: "Population shift", d: "Lesion types, prevalence and mucosal appearance vary between populations. We recommend local validation before clinical use." },
  { icon: <Microscope />, t: "Binary output", d: "The model separates suspicious from non-suspicious. It does not type lesions or stage disease. Histopathology remains the reference standard." },
  { icon: <FlaskConical />, t: "Evaluation caveats", d: "Our internal evaluation set may overlap training data, so it likely overestimates real-world performance. External validation is pending." },
];

const ROADMAP = [
  { icon: <Layers />, t: "Pixel-level lesion segmentation", d: "A segmentation model trained on clinician-annotated lesion masks, replacing the weakly supervised outline." },
  { icon: <Boxes />, t: "Multi-class lesion typing", d: "Distinguishing leukoplakia, erythroplakia, lichen planus and other presentations, not just suspicious vs not." },
  { icon: <FlaskConical />, t: "External validation study", d: "Prospective evaluation on independent sites and devices, with results published on this page." },
  { icon: <Cpu />, t: "On-device quantized model", d: "A compact model for the desktop app and dental hardware, running in real time without a network." },
];

export default function Models() {
  useSeo("Models", "Transparent model cards for OralAID: OralNet-B4 metrics, evaluation method, risk tier thresholds, limitations and retired models.");
  const [data, setData] = useState<{ threshold: number; high: number; models: ModelCard[]; live: boolean } | null>(null);

  useEffect(() => {
    let alive = true;
    api.models()
      .then((r) => alive && setData({ threshold: r.threshold, high: r.high_risk_threshold, models: r.models, live: true }))
      .catch(() => alive && setData({ threshold: 0.3, high: 0.55, models: STATIC_MODELS, live: false }));
    return () => { alive = false; };
  }, []);

  const production = data?.models.filter((m) => m.status !== "retired") ?? [];
  const retired = data?.models.filter((m) => m.status === "retired") ?? [];

  return (
    <>
      <PageHero eyebrow="Models" title={<>Model cards, <em className="italic text-brand">including the failures.</em></>}
        body="Every model we serve, how we evaluated it, where it falls short, and the models we retired and why. Trust in clinical AI starts with showing your working." />

      <Section>
        <div className="flex flex-col justify-between gap-4 sm:flex-row sm:items-end">
          <SectionHeading eyebrow="In production" title="What runs on every analysis" />
          {data && (
            <Reveal>
              <span className="inline-flex items-center gap-2 text-xs text-muted">
                {data.live ? <><span className="h-2 w-2 rounded-full bg-low" />Live from the API</> : <><WifiOff className="h-3.5 w-3.5" />API unreachable, showing published cards</>}
              </span>
            </Reveal>
          )}
        </div>
        <div className="mt-10 grid gap-4 lg:grid-cols-2">
          {!data && [0, 1, 2, 3].map((i) => <div key={i} className={cn("skeleton h-64 rounded-2xl", i === 0 && "lg:col-span-2")} />)}
          {production.map((m, i) => <ModelCardView key={m.id} m={m} index={i} />)}
        </div>
      </Section>

      {retired.length > 0 && (
        <Section className="border-t border-line">
          <SectionHeading eyebrow="Retired" title={<>Models we <em className="italic text-brand">stopped serving.</em></>}
            body="We keep these for audit, never serve them, and show them here because knowing what didn't work is part of judging what does." />
          <div className="mt-10 grid gap-4 lg:grid-cols-2">
            {retired.map((m, i) => <ModelCardView key={m.id} m={m} index={i} />)}
          </div>
        </Section>
      )}

      <Section className="border-y border-line bg-surface/50">
        <div className="grid gap-12 lg:grid-cols-[0.9fr_1.1fr] lg:gap-20">
          <SectionHeading eyebrow="How we evaluate" title={<>Numbers with their <em className="italic text-brand">context attached.</em></>} />
          <Reveal className="space-y-6 text-[15px] leading-relaxed text-muted">
            <p><span className="font-medium text-ink">Held-out validation.</span> During training, a split of images is held out. OralNet-B4 reached 76.8% accuracy on it.</p>
            <p><span className="font-medium text-ink">Internal evaluation.</span> We then run the full production pipeline, including test-time augmentation, on 400 images with a specialist diagnosis. At the 0.30 threshold: AUC 0.83, sensitivity 70.5%, specificity 80.2%. In the high-risk tier (score at or above 0.55), 90% of images were truly suspicious.</p>
            <p><span className="font-medium text-ink">What that does not tell you.</span> The internal set may overlap training data, so these figures are likely optimistic. They are not a substitute for independent, prospective validation, which is pending. We will publish external results here when available.</p>
            <p><span className="font-medium text-ink">Why the threshold is low.</span> In screening, a missed lesion costs more than an extra referral. We set the operating threshold at 0.30 to favour sensitivity, and use the high tier to flag the cases the model is most confident about.</p>
          </Reveal>
        </div>
      </Section>

      <Section>
        <SectionHeading eyebrow="Thresholds" title={<>One probability. <em className="italic text-brand">Three tiers.</em></>}
          body="OralNet-B4 outputs the probability that an image is suspicious. We map it to tiers with fixed thresholds, so the same score always means the same thing." />
        <Reveal className="mt-12">{data ? <TierScale threshold={data.threshold} high={data.high} /> : <TierScale threshold={0.3} high={0.55} />}</Reveal>
      </Section>

      <Section className="border-t border-line">
        <SectionHeading eyebrow="Limitations" title={<>Where it can <em className="italic text-brand">go wrong.</em></>}
          body="Known failure modes, stated up front. Each one is a reason the clinician makes the final call." />
        <div className="mt-12 grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
          {LIMITATIONS.map((l, i) => (
            <Reveal key={l.t} delay={i * 0.04} className="card p-6">
              <IconTile tone="accent">{l.icon}</IconTile>
              <h3 className="mt-5 font-semibold">{l.t}</h3>
              <p className="mt-2 text-sm leading-relaxed text-muted">{l.d}</p>
            </Reveal>
          ))}
        </div>
        <ClinicalNote className="mt-10 max-w-3xl" />
      </Section>

      <Section className="border-t border-line">
        <div className="flex flex-col justify-between gap-6 lg:flex-row lg:items-end">
          <SectionHeading eyebrow="Model roadmap" title="What we're building next" />
          <Reveal><Link to="/roadmap" className="inline-flex items-center gap-1.5 text-sm font-medium text-brand hover:underline">Full product roadmap <ArrowRight className="h-4 w-4" /></Link></Reveal>
        </div>
        <ol className="mt-12 grid gap-4 md:grid-cols-2">
          {ROADMAP.map((r, i) => (
            <li key={r.t}>
              <Reveal delay={i * 0.05} className="flex h-full gap-5 rounded-2xl border border-dashed border-line p-6">
                <IconTile tone="neutral">{r.icon}</IconTile>
                <div>
                  <div className="flex flex-wrap items-center gap-2"><h3 className="font-semibold">{r.t}</h3><Badge>Planned</Badge></div>
                  <p className="mt-1.5 text-sm text-muted">{r.d}</p>
                </div>
              </Reveal>
            </li>
          ))}
        </ol>
        <p className="mt-6 flex items-center gap-2 text-xs text-muted"><Lightbulb className="h-3.5 w-3.5" />Have labelled data or want to run a validation study with us? <Link to="/contact" className="text-brand hover:underline">Get in touch</Link>.</p>
      </Section>

      <CTABand title={<>See the model <em className="italic">on your images.</em></>} />
    </>
  );
}
