import { useCallback, useEffect, useMemo, useState, type ReactNode } from "react";
import { Link, useNavigate, useParams } from "react-router-dom";
import { AnimatePresence, motion } from "framer-motion";
import {
  AlertTriangle, ArrowLeft, Brain, Calendar, Check, ChevronRight, Clock, Cpu, Crosshair, FileText, Gauge, HelpCircle, Layers, Lightbulb,
  MapPin, Pencil, Printer, Sparkles, Stethoscope, ThumbsDown, ThumbsUp, Trash2, UserRound, ZoomIn,
} from "lucide-react";
import { api, ApiError, fileUrl, type CaseDetail as CaseData, type Feedback, type ImageResult } from "@/lib/api";
import { useSeo } from "@/lib/seo";
import { cn, formatDate, pct, timeAgo } from "@/lib/utils";
import { Badge, Button, ButtonLink, Card, EmptyState, ErrorNote, buttonClass } from "@/components/ui";
import { CompareSlider, ProbabilityMeter, QualityBar, RiskBadge } from "@/components/clinical";
import { Modal } from "@/components/Modal";
import { toast } from "@/components/Toast";
import {
  ConfirmDialog, errorMessage, HIGH_THRESHOLD, IMPRESSIONS, patientMeta, patientName, Segmented, Stars, THRESHOLD, TIER_RGB,
} from "@/components/app/kit";

type Tab = "compare" | "gradcam" | "region" | "heatmap" | "lime";
const TABS: { value: Tab; label: string }[] = [
  { value: "compare", label: "Compare" },
  { value: "gradcam", label: "Grad-CAM" },
  { value: "region", label: "Region" },
  { value: "heatmap", label: "Heatmap" },
  { value: "lime", label: "LIME" },
];
const TAB_HELP: Record<Tab, string> = {
  compare: "Drag the handle to compare the original photo with the Grad-CAM attention overlay.",
  gradcam: "Warm colours show where the classifier looked most when scoring the image.",
  region: "The outlined area is the region localised from the attention map. It is approximate, not a lesion boundary.",
  heatmap: "Raw attention intensity, from blue (little influence) to red (strong influence).",
  lime: "Superpixels that pushed the score toward suspicious (red) or non-suspicious (green).",
};
const STATUS_OPTIONS = [
  { value: "open", label: "Open" },
  { value: "referred", label: "Referred" },
  { value: "closed", label: "Closed" },
] as const;

export default function CaseDetail() {
  const { id = "" } = useParams();
  useSeo(`Case ${id.slice(0, 8)}`, undefined, { noindex: true });
  const navigate = useNavigate();
  const [data, setData] = useState<CaseData | null>(null);
  const [error, setError] = useState<{ status: number; message: string } | null>(null);
  const [idx, setIdx] = useState(0);
  const [deleting, setDeleting] = useState(false);
  const [confirmDelete, setConfirmDelete] = useState(false);

  useEffect(() => {
    let cancelled = false;
    setData(null);
    setError(null);
    api.case(id).then((c) => { if (!cancelled) setData(c); }).catch((e) => {
      if (!cancelled) setError({ status: e instanceof ApiError ? e.status : 0, message: errorMessage(e, "Couldn't load this case.") });
    });
    return () => { cancelled = true; };
  }, [id]);

  const img = data?.images[Math.min(idx, (data?.images.length ?? 1) - 1)];

  const setStatus = async (status: CaseData["status"]) => {
    if (!data || status === data.status) return;
    const prev = data.status;
    setData({ ...data, status });
    try {
      const res = await api.updateCase(data.id, { status });
      setData(res);
      toast.success(`Marked as ${status}`);
    } catch (e) {
      setData((d) => (d ? { ...d, status: prev } : d));
      toast.error(errorMessage(e, "Couldn't update status."));
    }
  };

  const remove = async () => {
    if (!data) return;
    setDeleting(true);
    try {
      await api.deleteCase(data.id);
      toast.success("Case deleted");
      navigate("/app/cases", { replace: true });
    } catch (e) {
      toast.error(errorMessage(e, "Couldn't delete the case."));
      setDeleting(false);
    }
  };

  if (error) {
    return (
      <div>
        <BackLink />
        {error.status === 404 ? (
          <EmptyState icon={<FileText className="h-5 w-5" />} title="Case not found"
            body="It may have been deleted, or it belongs to another account."
            action={<ButtonLink to="/app/cases" variant="secondary">Back to cases</ButtonLink>} />
        ) : <ErrorNote>{error.message}</ErrorNote>}
      </div>
    );
  }

  if (!data || !img) return <DetailSkeleton />;

  return (
    <div>
      <BackLink />
      {/* Header */}
      <div className="mb-6 flex flex-col gap-4 lg:flex-row lg:items-start lg:justify-between">
        <div className="min-w-0">
          <div className="flex flex-wrap items-center gap-2">
            <h1 className="text-2xl font-semibold tracking-tight sm:text-[28px]">{patientName(data.patient)}</h1>
            <RiskBadge tier={data.tier} />
          </div>
          <div className="mt-2 flex flex-wrap items-center gap-x-4 gap-y-1.5 text-[13px] text-muted">
            <span className="font-mono text-xs text-ink/80">#{data.id}</span>
            <span className="flex items-center gap-1.5"><Calendar className="h-3.5 w-3.5" />{formatDate(data.created_at)}</span>
            {data.site && <span className="flex items-center gap-1.5"><MapPin className="h-3.5 w-3.5" />{data.site}</span>}
            {patientMeta(data.patient) && <span className="flex items-center gap-1.5"><UserRound className="h-3.5 w-3.5" />{patientMeta(data.patient)}</span>}
          </div>
        </div>
        <div className="flex flex-wrap items-center gap-2">
          <Segmented label="Case status" size="sm" value={data.status} onChange={setStatus} options={STATUS_OPTIONS} />
          <a href={`/app/cases/${data.id}/report`} target="_blank" rel="noopener" className={buttonClass("secondary", "sm")}>
            <Printer className="h-4 w-4" />Print report
          </a>
          <Button variant="ghost" size="sm" onClick={() => setConfirmDelete(true)} className="text-high hover:bg-high/10" aria-label="Delete case">
            <Trash2 className="h-4 w-4" /><span className="sm:inline">Delete</span>
          </Button>
        </div>
      </div>

      <div className="grid gap-6 lg:grid-cols-[minmax(0,1.25fr)_minmax(0,1fr)]">
        <div className="min-w-0 space-y-4 lg:sticky lg:top-6 lg:self-start">
          {data.images.length > 1 && (
            <div className="flex gap-2 overflow-x-auto pb-1" role="tablist" aria-label="Images">
              {data.images.map((im, i) => (
                <button key={im.index} role="tab" aria-selected={i === idx} onClick={() => setIdx(i)}
                  className={cn("relative shrink-0 overflow-hidden rounded-xl border-2 transition", i === idx ? "border-brand" : "border-transparent opacity-70 hover:opacity-100")}>
                  <img src={fileUrl(im.files.original)} alt={`Image ${i + 1}`} className="h-16 w-16 object-cover" />
                  <span className="absolute inset-x-0 bottom-0 flex items-center justify-center gap-1 bg-black/60 py-0.5 text-[10px] font-medium tabular-nums text-white">
                    <span className="h-1.5 w-1.5 rounded-full" style={{ background: TIER_RGB[im.tier] }} />{Math.round(im.probability * 100)}%
                  </span>
                </button>
              ))}
            </div>
          )}
          <Viewer key={img.index} caseId={data.id} img={img} onUpdated={setData} />
        </div>

        <div className="min-w-0 space-y-4">
          <ResultCard img={img} multi={data.images.length > 1} caseProbability={data.probability} />
          <Explain img={img} site={data.site} />
          <QualityCard img={img} />
          <FeedbackCard data={data} onSaved={setData} />
          <NotesCard data={data} onSaved={setData} />
          {data.patient && (
            <Link to={`/app/patients/${data.patient.id}`} className="card group flex items-center gap-4 p-4 transition hover:shadow-lift">
              <div className="grid h-10 w-10 place-items-center rounded-xl bg-brand/10 text-brand"><UserRound className="h-5 w-5" /></div>
              <div className="min-w-0 flex-1">
                <p className="text-sm font-medium">Patient timeline</p>
                <p className="truncate text-xs text-muted">See how {patientName(data.patient)}'s score has changed across visits</p>
                {!!data.patient.risk_factors?.length && (
                  <div className="mt-2 flex flex-wrap gap-1">{data.patient.risk_factors.map((f) => <Badge key={f}>{f}</Badge>)}</div>
                )}
              </div>
              <ChevronRight className="h-4 w-4 text-muted transition group-hover:translate-x-0.5" />
            </Link>
          )}
        </div>
      </div>

      <ConfirmDialog open={confirmDelete} onClose={() => setConfirmDelete(false)} onConfirm={remove} loading={deleting}
        title="Delete this case?" confirmLabel="Delete case"
        body={<>This permanently removes the images, explanations and feedback for case <span className="font-mono">{data.id.slice(0, 10)}</span>. This can't be undone.</>} />
    </div>
  );
}

function BackLink() {
  return (
    <Link to="/app/cases" className="mb-4 inline-flex items-center gap-1.5 text-[13px] text-muted transition hover:text-ink">
      <ArrowLeft className="h-3.5 w-3.5" />Cases
    </Link>
  );
}

function Viewer({ caseId, img, onUpdated }: { caseId: string; img: ImageResult; onUpdated: (c: CaseData) => void }) {
  const [tab, setTab] = useState<Tab>("compare");
  const [opacity, setOpacity] = useState(0.9);
  const [zoom, setZoom] = useState(false);
  const [limeBusy, setLimeBusy] = useState(false);
  const [limeError, setLimeError] = useState<string | null>(null);
  const [progress, setProgress] = useState(0);

  const original = fileUrl(img.files.original);
  const overlay = {
    compare: fileUrl(img.files.gradcam),
    gradcam: fileUrl(img.files.gradcam),
    region: fileUrl(img.files.segmentation),
    heatmap: fileUrl(img.files.heatmap),
    lime: img.files.lime ? fileUrl(img.files.lime) : "",
  }[tab];

  useEffect(() => {
    if (!limeBusy) return;
    setProgress(0);
    const started = Date.now();
    // Asymptotic progress: approaches ~95% over ~30 s, completes when the request returns.
    const t = setInterval(() => setProgress(Math.min(95, 100 * (1 - Math.exp(-(Date.now() - started) / 11000)))), 200);
    return () => clearInterval(t);
  }, [limeBusy]);

  const generateLime = async () => {
    setLimeBusy(true);
    setLimeError(null);
    try {
      const res = await api.lime(caseId);
      setProgress(100);
      onUpdated(res);
      toast.success("LIME explanation ready");
    } catch (e) {
      setLimeError(errorMessage(e, "Couldn't generate the LIME explanation."));
    } finally {
      setLimeBusy(false);
    }
  };

  const showOverlayControls = tab !== "lime" || !!img.files.lime;

  return (
    <Card className="overflow-hidden p-3 sm:p-4">
      <div className="mb-3 flex items-center gap-2 overflow-x-auto">
        <div role="tablist" aria-label="Explanation view" className="flex rounded-xl bg-surface-2/70 p-1">
          {TABS.map((t) => (
            <button key={t.value} role="tab" aria-selected={tab === t.value} onClick={() => setTab(t.value)}
              className={cn("relative whitespace-nowrap rounded-lg px-3 py-1.5 text-[13px] font-medium transition-colors", tab === t.value ? "text-ink" : "text-muted hover:text-ink")}>
              {tab === t.value && <motion.span layoutId={`viewer-tab-${caseId}`} className="absolute inset-0 rounded-lg border border-line bg-surface shadow-sm" transition={{ type: "spring", damping: 30, stiffness: 400 }} />}
              <span className="relative flex items-center gap-1">
                {t.label}{t.value === "lime" && !img.files.lime && <span className="h-1.5 w-1.5 rounded-full bg-accent" />}
              </span>
            </button>
          ))}
        </div>
      </div>

      <AnimatePresence mode="wait">
        <motion.div key={tab} initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} transition={{ duration: 0.15 }}>
          {tab === "compare" ? (
            <div className="relative">
              <CompareSlider before={original} after={overlay} labelBefore="Original" labelAfter="Grad-CAM" opacity={opacity} />
              <button onClick={() => setZoom(true)} aria-label="Enlarge"
                className="absolute bottom-3 right-3 grid h-9 w-9 place-items-center rounded-lg bg-black/55 text-white backdrop-blur hover:bg-black/70"><ZoomIn className="h-4 w-4" /></button>
            </div>
          ) : tab === "lime" && !img.files.lime ? (
            <div className="relative grid aspect-square place-items-center overflow-hidden rounded-2xl bg-surface-2">
              <img src={original} alt="" className="absolute inset-0 h-full w-full object-cover opacity-25 blur-sm" />
              <div className="relative max-w-xs p-6 text-center">
                <div className="mx-auto grid h-12 w-12 place-items-center rounded-2xl bg-surface text-brand shadow-soft"><Sparkles className="h-5 w-5" /></div>
                <h3 className="mt-4 text-base font-semibold">Second-opinion explanation</h3>
                <p className="mt-1 text-sm text-muted">LIME perturbs regions of the image to see which ones change the score. It takes about 15–30 seconds.</p>
                {limeBusy ? (
                  <div className="mt-5">
                    <div className="h-1.5 overflow-hidden rounded-full bg-surface">
                      <motion.div className="h-full rounded-full bg-brand" animate={{ width: `${progress}%` }} transition={{ ease: "linear" }} />
                    </div>
                    <p className="mt-2 text-xs tabular-nums text-muted">Perturbing superpixels... {Math.round(progress)}%</p>
                  </div>
                ) : (
                  <Button className="mt-5" onClick={generateLime}><Sparkles className="h-4 w-4" />Generate LIME explanation</Button>
                )}
                {limeError && <div className="mt-3 text-left"><ErrorNote>{limeError}</ErrorNote></div>}
              </div>
            </div>
          ) : (
            <button onClick={() => setZoom(true)} className="group relative block aspect-square w-full cursor-zoom-in overflow-hidden rounded-2xl bg-surface-2" aria-label="Enlarge image">
              <img src={original} alt="Original" className="absolute inset-0 h-full w-full object-cover" />
              <img src={overlay} alt={TABS.find((t) => t.value === tab)?.label} className="absolute inset-0 h-full w-full object-cover transition-opacity" style={{ opacity }} />
              <span className="absolute bottom-3 right-3 grid h-9 w-9 place-items-center rounded-lg bg-black/55 text-white opacity-0 backdrop-blur transition group-hover:opacity-100"><ZoomIn className="h-4 w-4" /></span>
            </button>
          )}
        </motion.div>
      </AnimatePresence>

      <div className="mt-3 space-y-3 px-1">
        {showOverlayControls && (
          <label className="flex items-center gap-3 text-xs text-muted">
            <Layers className="h-3.5 w-3.5 shrink-0" /><span className="w-14 shrink-0">Overlay</span>
            <input type="range" min={0} max={1} step={0.01} value={opacity} onChange={(e) => setOpacity(Number(e.target.value))}
              className="h-1 flex-1 cursor-pointer accent-[rgb(var(--brand))]" aria-label="Overlay opacity" />
            <span className="w-9 text-right tabular-nums">{Math.round(opacity * 100)}%</span>
          </label>
        )}
        {tab === "lime" && img.files.lime && (
          <div className="flex flex-wrap items-center gap-4 text-xs text-muted">
            <span className="flex items-center gap-1.5"><span className="h-2.5 w-2.5 rounded-sm bg-[#e5484d]" />Pushes toward suspicious</span>
            <span className="flex items-center gap-1.5"><span className="h-2.5 w-2.5 rounded-sm bg-[#30a46c]" />Pushes toward non-suspicious</span>
          </div>
        )}
        {(tab === "heatmap" || tab === "gradcam") && (
          <div className="flex items-center gap-2 text-[11px] text-muted">
            <span>Low</span><span className="h-1.5 flex-1 rounded-full bg-[linear-gradient(90deg,#1e3a8a,#06b6d4,#84cc16,#facc15,#ef4444)]" /><span>High attention</span>
          </div>
        )}
        <p className="text-xs text-muted">{TAB_HELP[tab]}</p>
      </div>

      <Modal open={zoom} onClose={() => setZoom(false)} title={TABS.find((t) => t.value === tab)?.label} className="sm:max-w-4xl">
        <div className="relative mx-auto aspect-square max-h-[75vh] overflow-hidden rounded-2xl bg-surface-2">
          <img src={original} alt="Original" className="absolute inset-0 h-full w-full object-contain" />
          {overlay && <img src={overlay} alt="Overlay" className="absolute inset-0 h-full w-full object-contain" style={{ opacity }} />}
        </div>
      </Modal>
    </Card>
  );
}

function ResultCard({ img, multi, caseProbability }: { img: ImageResult; multi: boolean; caseProbability: number }) {
  const high = img.tier === "high";
  return (
    <Card className="p-5 sm:p-6">
      <div className="flex items-center justify-between">
        <p className="eyebrow">AI screening result</p>
        <RiskBadge tier={img.tier} size="sm" />
      </div>
      <div className="mt-4 flex flex-col items-center">
        <ProbabilityMeter value={img.probability} threshold={img.threshold || THRESHOLD} high={HIGH_THRESHOLD} />
        <p className="mt-4 text-center text-base font-medium">{img.finding}</p>
        <div className="mt-3 flex flex-wrap justify-center gap-x-5 gap-y-1 text-xs text-muted">
          <span>Uncertainty <span className="font-medium tabular-nums text-ink">± {(img.uncertainty * 100).toFixed(1)} pts</span></span>
          <span>Confidence <span className="font-medium tabular-nums text-ink">{pct(img.confidence)}</span></span>
        </div>
        {multi && Math.abs(caseProbability - img.probability) > 0.0001 && (
          <p className="mt-2 text-center text-xs text-muted">Case result uses the most concerning image ({Math.round(caseProbability * 100)}%).</p>
        )}
      </div>
      <div className={cn("mt-5 flex gap-3 rounded-xl border p-4",
        high ? "border-high/30 bg-high/[0.07]" : img.tier === "moderate" ? "border-moderate/30 bg-moderate/[0.06]" : "border-line bg-surface-2/50")}>
        {high ? <AlertTriangle className="mt-0.5 h-5 w-5 shrink-0 text-high" /> : <Stethoscope className={cn("mt-0.5 h-5 w-5 shrink-0", img.tier === "moderate" ? "text-moderate" : "text-muted")} />}
        <div>
          <p className={cn("text-xs font-semibold uppercase tracking-wide", high ? "text-high" : img.tier === "moderate" ? "text-moderate" : "text-muted")}>Recommendation</p>
          <p className={cn("mt-1 text-sm", high && "font-medium")}>{img.recommendation}</p>
        </div>
      </div>
      <dl className="mt-5 grid grid-cols-2 gap-px overflow-hidden rounded-xl border border-line bg-line text-sm">
        <Metric icon={<Crosshair className="h-3.5 w-3.5" />} label="Attention area" value={pct(img.attention_area_ratio, 1)} />
        <Metric icon={<Layers className="h-3.5 w-3.5" />} label="Region area" value={pct(img.region_area_ratio, 1)} />
        <Metric icon={<Clock className="h-3.5 w-3.5" />} label="Inference" value={img.inference_ms >= 1000 ? `${(img.inference_ms / 1000).toFixed(1)} s` : `${img.inference_ms} ms`} />
        <Metric icon={<Cpu className="h-3.5 w-3.5" />} label="Model" value="OralNet-B4" />
      </dl>
    </Card>
  );
}

function Metric({ icon, label, value }: { icon: ReactNode; label: string; value: string }) {
  return (
    <div className="bg-surface p-3">
      <dt className="flex items-center gap-1.5 text-xs text-muted">{icon}{label}</dt>
      <dd className="mt-1 font-medium tabular-nums">{value}</dd>
    </div>
  );
}

function explain(img: ImageResult, site?: string | null): { icon: ReactNode; text: string }[] {
  const p = img.probability;
  const th = img.threshold || THRESHOLD;
  const pts = (v: number) => `${Math.round(Math.abs(v) * 100)} percentage point${Math.round(Math.abs(v) * 100) === 1 ? "" : "s"}`;
  const out: { icon: ReactNode; text: string }[] = [];

  let score = `The model's suspicion score is ${Math.round(p * 100)}%. `;
  if (p < th) score += `That is ${pts(th - p)} below the ${Math.round(th * 100)}% flag threshold, so the image is classed as low risk.`;
  else if (p < HIGH_THRESHOLD) score += `That is above the ${Math.round(th * 100)}% flag threshold but ${pts(HIGH_THRESHOLD - p)} under the ${Math.round(HIGH_THRESHOLD * 100)}% high-risk line.`;
  else score += `That is ${pts(p - HIGH_THRESHOLD)} above the ${Math.round(HIGH_THRESHOLD * 100)}% high-risk line.`;
  const near = Math.min(Math.abs(p - th), Math.abs(p - HIGH_THRESHOLD));
  if (near < 0.06) score += " Because it sits close to a threshold, treat this as a borderline result.";
  out.push({ icon: <Gauge className="h-4 w-4" />, text: score });

  const a = img.attention_area_ratio;
  const focus = a < 0.08
    ? `The model focused on a small, concentrated area (about ${pct(a)} of the image).`
    : a < 0.25
      ? `The model's attention covered a moderate area (about ${pct(a)} of the image).`
      : `Attention was spread over a large part of the image (about ${pct(a)}), so no single feature clearly drove the result.`;
  out.push({
    icon: <Crosshair className="h-4 w-4" />,
    text: `${focus} Check that the highlighted area matches the tissue you are concerned about${site ? ` on the ${site.toLowerCase()}` : ""}, not teeth, instruments or background.`,
  });

  const u = img.uncertainty;
  out.push({
    icon: <Brain className="h-4 w-4" />,
    text: u > 0.08
      ? `Predictions varied across the augmented views (± ${(u * 100).toFixed(1)} points), which lowers confidence in the exact score.`
      : `Predictions were consistent across augmented views (± ${(u * 100).toFixed(1)} points).`,
  });

  const q = img.quality;
  out.push({
    icon: <Lightbulb className="h-4 w-4" />,
    text: q.score < 55
      ? `Image quality is low (${q.score}/100)${q.issues.length ? `: ${q.issues.join(", ").toLowerCase()}` : ""}. Consider retaking the photo before relying on this result.`
      : q.issues.length
        ? `Image quality is acceptable (${q.score}/100), with notes: ${q.issues.join(", ").toLowerCase()}.`
        : `Image quality looks good (${q.score}/100), so it is unlikely to have distorted the result.`,
  });
  return out;
}

function Explain({ img, site }: { img: ImageResult; site?: string | null }) {
  const items = useMemo(() => explain(img, site), [img, site]);
  return (
    <Card className="p-5 sm:p-6">
      <div className="mb-4 flex items-center gap-2"><HelpCircle className="h-4 w-4 text-brand" /><h2 className="text-[15px] font-semibold">Explain this result</h2></div>
      <ul className="space-y-3.5">
        {items.map((it, i) => (
          <li key={i} className="flex gap-3 text-sm leading-relaxed text-ink/90">
            <span className="mt-0.5 grid h-7 w-7 shrink-0 place-items-center rounded-lg bg-surface-2 text-muted">{it.icon}</span>
            <span>{it.text}</span>
          </li>
        ))}
      </ul>
      <p className="mt-4 border-t border-line pt-3 text-xs text-muted">Generated from the model outputs. Supports, and does not replace, clinical examination and histopathology.</p>
    </Card>
  );
}

function QualityCard({ img }: { img: ImageResult }) {
  const q = img.quality;
  return (
    <Card className="p-5 sm:p-6">
      <div className="mb-3 flex items-center justify-between">
        <h2 className="text-[15px] font-semibold">Image quality</h2>
        <span className="text-xs text-muted">{q.resolution}</span>
      </div>
      <QualityBar score={q.score} />
      <div className="mt-3 grid grid-cols-2 gap-3 text-xs text-muted">
        <span>Sharpness <span className="font-medium tabular-nums text-ink">{q.sharpness}</span></span>
        <span>Brightness <span className="font-medium tabular-nums text-ink">{q.brightness}</span></span>
      </div>
      {q.issues.length ? (
        <ul className="mt-3 space-y-1.5">
          {q.issues.map((i) => <li key={i} className="flex items-center gap-2 text-[13px] text-moderate"><AlertTriangle className="h-3.5 w-3.5 shrink-0" />{i}</li>)}
        </ul>
      ) : (
        <p className="mt-3 flex items-center gap-2 text-[13px] text-low"><Check className="h-3.5 w-3.5" />No quality issues detected</p>
      )}
    </Card>
  );
}

const VERDICTS = [
  { value: "agree", label: "Agree", icon: <ThumbsUp className="h-3.5 w-3.5" /> },
  { value: "disagree", label: "Disagree", icon: <ThumbsDown className="h-3.5 w-3.5" /> },
  { value: "unsure", label: "Unsure", icon: <HelpCircle className="h-3.5 w-3.5" /> },
] as const;

function FeedbackCard({ data, onSaved }: { data: CaseData; onSaved: (c: CaseData) => void }) {
  const init = useCallback((f: Feedback | null) => ({
    verdict: f?.verdict ?? null,
    clinical_impression: f?.clinical_impression ?? null,
    histopathology: f?.histopathology ?? "",
    explanation_rating: f?.explanation_rating ?? null,
    comment: f?.comment ?? "",
  }), []);
  const [form, setForm] = useState(() => init(data.feedback));
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const saved = init(data.feedback);
  const dirty = JSON.stringify(form) !== JSON.stringify(saved);

  const save = async () => {
    if (!form.verdict) return;
    setSaving(true);
    setError(null);
    try {
      const res = await api.feedback(data.id, {
        verdict: form.verdict,
        clinical_impression: form.clinical_impression,
        histopathology: form.histopathology.trim() || null,
        explanation_rating: form.explanation_rating,
        comment: form.comment.trim() || null,
      });
      onSaved(res);
      setForm(init(res.feedback));
      toast.success("Feedback saved");
    } catch (e) {
      setError(errorMessage(e, "Couldn't save feedback."));
    } finally {
      setSaving(false);
    }
  };

  return (
    <Card className="p-5 sm:p-6">
      <div className="mb-1 flex items-center justify-between gap-2">
        <h2 className="text-[15px] font-semibold">Clinician feedback</h2>
        {data.feedback && !dirty && (
          <span className="flex items-center gap-1.5 text-xs text-low"><Check className="h-3.5 w-3.5" />Saved {data.feedback.created_at ? timeAgo(data.feedback.created_at) : ""}</span>
        )}
      </div>
      <p className="mb-5 text-[13px] text-muted">Your review tracks agreement and helps improve the model.</p>
      <div className="space-y-5">
        <div>
          <span className="label">Do you agree with the AI result?</span>
          <Segmented label="Verdict" value={form.verdict} onChange={(v) => setForm({ ...form, verdict: v })} options={VERDICTS} className="w-full" />
        </div>
        <div>
          <span className="label">Clinical impression</span>
          <div className="grid grid-cols-2 gap-2 sm:grid-cols-4">
            {IMPRESSIONS.map((i) => (
              <button key={i.value} type="button" aria-pressed={form.clinical_impression === i.value}
                onClick={() => setForm({ ...form, clinical_impression: form.clinical_impression === i.value ? null : i.value })}
                className={cn("h-9 rounded-lg border text-[13px] font-medium transition",
                  form.clinical_impression === i.value ? "border-brand/40 bg-brand/10 text-brand" : "border-line text-muted hover:text-ink")}>
                {i.label}
              </button>
            ))}
          </div>
        </div>
        <label className="block">
          <span className="label">Histopathology result</span>
          <input className="input" value={form.histopathology} maxLength={200} onChange={(e) => setForm({ ...form, histopathology: e.target.value })}
            placeholder="e.g. Mild epithelial dysplasia (if biopsied)" />
        </label>
        <div>
          <span className="label">How useful was the explanation?</span>
          <Stars value={form.explanation_rating} onChange={(v) => setForm({ ...form, explanation_rating: v })} />
        </div>
        <label className="block">
          <span className="label">Comment</span>
          <textarea className="input min-h-[80px] resize-y" value={form.comment} maxLength={4000} onChange={(e) => setForm({ ...form, comment: e.target.value })}
            placeholder="Anything the model missed or got right" />
        </label>
        {error && <ErrorNote>{error}</ErrorNote>}
        <div className="flex items-center justify-end gap-2">
          {dirty && data.feedback && <Button variant="ghost" onClick={() => setForm(saved)}>Discard</Button>}
          <Button onClick={save} loading={saving} disabled={!form.verdict || !dirty}>
            {data.feedback ? "Update feedback" : "Save feedback"}
          </Button>
        </div>
      </div>
    </Card>
  );
}

function NotesCard({ data, onSaved }: { data: CaseData; onSaved: (c: CaseData) => void }) {
  const [editing, setEditing] = useState(false);
  const [value, setValue] = useState(data.notes ?? "");
  const [saving, setSaving] = useState(false);

  const save = async () => {
    setSaving(true);
    try {
      const res = await api.updateCase(data.id, { notes: value.trim() });
      onSaved(res);
      setEditing(false);
      toast.success("Notes saved");
    } catch (e) {
      toast.error(errorMessage(e, "Couldn't save notes."));
    } finally {
      setSaving(false);
    }
  };

  return (
    <Card className="p-5 sm:p-6">
      <div className="mb-3 flex items-center justify-between">
        <h2 className="text-[15px] font-semibold">Clinical notes</h2>
        {!editing && (
          <button onClick={() => { setValue(data.notes ?? ""); setEditing(true); }} className="flex items-center gap-1 text-xs font-medium text-muted hover:text-ink">
            <Pencil className="h-3.5 w-3.5" />{data.notes ? "Edit" : "Add"}
          </button>
        )}
      </div>
      {editing ? (
        <div className="space-y-3">
          <textarea className="input min-h-[110px] resize-y" value={value} onChange={(e) => setValue(e.target.value)} maxLength={4000} autoFocus aria-label="Clinical notes" />
          <div className="flex justify-end gap-2">
            <Button variant="ghost" size="sm" onClick={() => setEditing(false)}>Cancel</Button>
            <Button size="sm" onClick={save} loading={saving}>Save notes</Button>
          </div>
        </div>
      ) : data.notes ? (
        <p className="whitespace-pre-wrap text-sm leading-relaxed text-ink/90">{data.notes}</p>
      ) : (
        <p className="text-sm text-muted">No notes recorded for this case.</p>
      )}
    </Card>
  );
}

function DetailSkeleton() {
  return (
    <div>
      <div className="skeleton mb-4 h-4 w-16" />
      <div className="mb-6 space-y-3"><div className="skeleton h-8 w-64" /><div className="skeleton h-4 w-80 max-w-full" /></div>
      <div className="grid gap-6 lg:grid-cols-[minmax(0,1.25fr)_minmax(0,1fr)]">
        <div className="card p-4"><div className="skeleton mb-3 h-9 w-72 max-w-full" /><div className="skeleton aspect-square w-full rounded-2xl" /></div>
        <div className="space-y-4">
          <div className="card flex flex-col items-center p-6"><div className="skeleton h-32 w-56 rounded-t-full" /><div className="skeleton mt-4 h-4 w-48" /><div className="skeleton mt-5 h-16 w-full" /></div>
          <div className="card space-y-3 p-6"><div className="skeleton h-4 w-32" /><div className="skeleton h-3 w-full" /><div className="skeleton h-3 w-5/6" /><div className="skeleton h-3 w-4/6" /></div>
        </div>
      </div>
    </div>
  );
}
