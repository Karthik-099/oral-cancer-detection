import { useEffect, useRef, useState, type DragEvent } from "react";
import { AnimatePresence, motion } from "framer-motion";
import {
  AlertTriangle, Camera, Cpu, Crop, Download, ImagePlus, Lock, Printer, RefreshCw, ScanLine, ShieldCheck, Trash2, WifiOff, X,
} from "lucide-react";
import { useSeo } from "@/lib/seo";
import { cn, pct } from "@/lib/utils";
import {
  analyzeOnDevice, HIGH_THRESHOLD, isModelCached, loadModel, OnDeviceError, THRESHOLD, type LoadProgress, type OnDeviceResult,
} from "@/lib/ondevice";
import { Button, Badge, ErrorNote } from "@/components/ui";
import { CompareSlider, ProbabilityMeter, QualityBar, RiskBadge } from "@/components/clinical";
import { MouthMap, SITES } from "@/components/MouthMap";
import { Modal } from "@/components/Modal";
import { ImageCropper } from "@/components/ImageCropper";
import { CameraCapture } from "@/components/app/CameraCapture";
import { Segmented } from "@/components/app/kit";

const MAX_IMAGES = 6;
const MAX_BYTES = 15 * 1024 * 1024;

interface Item { id: string; file: File; url: string }
interface Analysed { item: Item; result?: OnDeviceResult; error?: string }
type View = "compare" | "gradcam" | "segmentation" | "heatmap";

export default function Screen() {
  useSeo(
    "Free on-device oral lesion screening",
    "Screen intraoral photos for suspicious oral lesions with explainable AI that runs entirely in your browser. Photos are never uploaded.",
  );
  const [items, setItems] = useState<Item[]>([]);
  const [site, setSite] = useState<string | null>(null);
  const [label, setLabel] = useState("");
  const [notes, setNotes] = useState("");
  const [cropping, setCropping] = useState<Item | null>(null);
  const [camera, setCamera] = useState(false);
  const [drag, setDrag] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [running, setRunning] = useState(false);
  const [progress, setProgress] = useState<LoadProgress | null>(null);
  const [results, setResults] = useState<Analysed[] | null>(null);
  const [cached, setCached] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => { isModelCached().then(setCached); }, []);
  useEffect(() => () => items.forEach((i) => URL.revokeObjectURL(i.url)), []); // eslint-disable-line react-hooks/exhaustive-deps

  const add = (files: FileList | File[]) => {
    setError(null);
    const next = [...items];
    for (const f of Array.from(files)) {
      if (next.length >= MAX_IMAGES) { setError(`Up to ${MAX_IMAGES} photos per screening.`); break; }
      if (f.type && !f.type.startsWith("image/")) { setError(`${f.name} isn't an image.`); continue; }
      if (f.size > MAX_BYTES) { setError(`${f.name} is larger than 15 MB.`); continue; }
      next.push({ id: crypto.randomUUID?.() ?? `${Date.now()}-${Math.random()}`, file: f, url: URL.createObjectURL(f) });
    }
    setItems(next);
  };

  const remove = (id: string) => setItems((xs) => {
    const it = xs.find((x) => x.id === id);
    if (it) URL.revokeObjectURL(it.url);
    return xs.filter((x) => x.id !== id);
  });

  const replace = (id: string, file: File) => setItems((xs) => xs.map((x) => {
    if (x.id !== id) return x;
    URL.revokeObjectURL(x.url);
    return { ...x, file, url: URL.createObjectURL(file) };
  }));

  const run = async () => {
    if (!items.length) return;
    setRunning(true);
    setError(null);
    try {
      await loadModel(setProgress);
      setCached(true);
      const out: Analysed[] = [];
      for (const item of items) {
        try {
          out.push({ item, result: await analyzeOnDevice(item.file) });
        } catch (e) {
          out.push({ item, error: e instanceof OnDeviceError ? e.message : "This photo couldn't be analysed." });
        }
      }
      setResults(out);
      window.scrollTo({ top: 0, behavior: "smooth" });
    } catch (e) {
      setError(e instanceof OnDeviceError ? e.message : "The model couldn't be loaded. Check your connection and try again.");
    } finally {
      setRunning(false);
      setProgress(null);
    }
  };

  const startOver = () => {
    items.forEach((i) => URL.revokeObjectURL(i.url));
    setItems([]);
    setResults(null);
    setNotes("");
    setLabel("");
    setSite(null);
  };

  if (results) {
    return <Results results={results} site={site} label={label} notes={notes} onBack={() => setResults(null)} onReset={startOver} />;
  }

  const downloading = progress && !progress.cached && progress.loaded < progress.total;

  return (
    <div className="container py-10 sm:py-14">
      <div className="mx-auto max-w-5xl">
        <div className="mb-10 flex flex-col gap-5 sm:flex-row sm:items-end sm:justify-between">
          <div>
            <Badge tone="brand" className="mb-4"><Lock className="h-3 w-3" />Runs on your device</Badge>
            <h1 className="display text-4xl sm:text-6xl">Screen a patient. <em className="italic text-brand">Privately.</em></h1>
            <p className="mt-4 max-w-xl text-muted">
              Add intraoral photos and OralNet-B4 analyses them right here in your browser. Nothing is uploaded, nothing is stored,
              and closing this tab clears everything.
            </p>
          </div>
          <div className="flex flex-wrap gap-2 text-xs text-muted">
            <span className="inline-flex items-center gap-1.5 rounded-full border border-line px-3 py-1.5"><ShieldCheck className="h-3.5 w-3.5 text-brand" />No upload</span>
            <span className="inline-flex items-center gap-1.5 rounded-full border border-line px-3 py-1.5"><WifiOff className="h-3.5 w-3.5 text-brand" />{cached ? "Model ready, works offline" : "Works offline after first run"}</span>
          </div>
        </div>

        <div className="grid gap-6 lg:grid-cols-[1.4fr_1fr]">
          <section className="card p-5 sm:p-6" aria-labelledby="photos-h">
            <div className="mb-4 flex items-center justify-between">
              <h2 id="photos-h" className="font-semibold">Photos <span className="text-muted">({items.length}/{MAX_IMAGES})</span></h2>
              <div className="flex gap-2">
                <Button size="sm" variant="secondary" onClick={() => setCamera(true)}><Camera className="h-4 w-4" />Camera</Button>
                <Button size="sm" variant="secondary" onClick={() => inputRef.current?.click()}><ImagePlus className="h-4 w-4" />Add</Button>
              </div>
            </div>
            <input ref={inputRef} type="file" multiple accept="image/jpeg,image/png,image/webp,image/*" className="sr-only"
              onChange={(e) => { if (e.target.files) add(e.target.files); e.target.value = ""; }} />

            {items.length === 0 ? (
              <label
                onClick={() => inputRef.current?.click()}
                onDragOver={(e) => { e.preventDefault(); setDrag(true); }}
                onDragLeave={() => setDrag(false)}
                onDrop={(e: DragEvent) => { e.preventDefault(); setDrag(false); add(e.dataTransfer.files); }}
                className={cn("flex min-h-[300px] cursor-pointer flex-col items-center justify-center gap-3 rounded-2xl border-2 border-dashed p-8 text-center transition",
                  drag ? "border-brand bg-brand/5" : "border-line hover:border-brand/50 hover:bg-surface-2/50")}
              >
                <div className="grid h-14 w-14 place-items-center rounded-2xl bg-brand/10 text-brand"><ImagePlus className="h-6 w-6" /></div>
                <p className="font-semibold">Drop photos here, or click to choose</p>
                <p className="text-sm text-muted">Up to {MAX_IMAGES} photos · JPEG, PNG or WebP</p>
              </label>
            ) : (
              <div className="grid grid-cols-2 gap-3 sm:grid-cols-3">
                {items.map((it) => (
                  <div key={it.id} className="group relative aspect-square overflow-hidden rounded-xl bg-surface-2">
                    <img src={it.url} alt={it.file.name} className="h-full w-full object-cover" />
                    <div className="absolute inset-x-0 bottom-0 flex justify-end gap-1 bg-gradient-to-t from-black/60 to-transparent p-2">
                      <button onClick={() => setCropping(it)} className="grid h-8 w-8 place-items-center rounded-lg bg-white/90 text-ink" aria-label="Crop photo"><Crop className="h-4 w-4" /></button>
                      <button onClick={() => remove(it.id)} className="grid h-8 w-8 place-items-center rounded-lg bg-white/90 text-high" aria-label="Remove photo"><Trash2 className="h-4 w-4" /></button>
                    </div>
                  </div>
                ))}
                {items.length < MAX_IMAGES && (
                  <button onClick={() => inputRef.current?.click()}
                    className="flex aspect-square flex-col items-center justify-center gap-2 rounded-xl border-2 border-dashed border-line text-sm text-muted transition hover:border-brand/50 hover:text-ink">
                    <ImagePlus className="h-5 w-5" />Add photo
                  </button>
                )}
              </div>
            )}

            <div className="mt-5 rounded-xl bg-surface-2/60 p-4 text-[13px] text-muted">
              <p className="mb-1.5 font-medium text-ink">For reliable results</p>
              Good light, dry the area, retract the cheek and fill the frame with the tissue. Use <Crop className="inline h-3.5 w-3.5" /> to cut out
              fingers, mirrors and background: the model can be distracted by them.
            </div>
          </section>

          <aside className="space-y-6">
            <section className="card p-5 sm:p-6" aria-labelledby="site-h">
              <h2 id="site-h" className="mb-1 font-semibold">Site <span className="font-normal text-muted">(optional)</span></h2>
              <p className="mb-3 text-sm text-muted">{site ?? "Tap the area being photographed."}</p>
              <MouthMap value={site} onChange={setSite} className="mx-auto max-w-[260px]" />
              <select value={site ?? ""} onChange={(e) => setSite(e.target.value || null)} className="input mt-3" aria-label="Anatomical site">
                <option value="">Select site…</option>
                {SITES.map((s) => <option key={s}>{s}</option>)}
              </select>
            </section>
            <section className="card space-y-4 p-5 sm:p-6">
              <label className="block">
                <span className="label">Reference <span className="font-normal text-muted">(optional, stays on this device)</span></span>
                <input value={label} onChange={(e) => setLabel(e.target.value)} maxLength={60} className="input" placeholder="e.g. Visit 2, left buccal" />
              </label>
              <label className="block">
                <span className="label">Notes</span>
                <textarea value={notes} onChange={(e) => setNotes(e.target.value)} maxLength={1000} rows={3} className="input" placeholder="Clinical observations for the printed report" />
              </label>
            </section>
          </aside>
        </div>

        <div className="sticky bottom-4 z-20 mt-8">
          <div className="card flex flex-col gap-3 p-4 shadow-lift sm:flex-row sm:items-center">
            <div className="flex-1 text-sm">
              {downloading ? (
                <div>
                  <p className="mb-1.5 font-medium">Downloading the model (one time)</p>
                  <div className="h-1.5 overflow-hidden rounded-full bg-surface-2"><div className="h-full rounded-full bg-brand transition-[width]" style={{ width: `${(progress!.loaded / progress!.total) * 100}%` }} /></div>
                  <p className="mt-1 text-xs tabular-nums text-muted">{(progress!.loaded / 1048576).toFixed(0)} / {(progress!.total / 1048576).toFixed(0)} MB</p>
                </div>
              ) : error ? <ErrorNote>{error}</ErrorNote> : (
                <p className="text-muted">{items.length ? `${items.length} photo${items.length > 1 ? "s" : ""} ready.` : "Add at least one photo."} {cached ? "" : "First run downloads about 75 MB."}</p>
              )}
            </div>
            <Button size="lg" onClick={run} loading={running} disabled={!items.length}>
              {!running && <ScanLine className="h-4 w-4" />}{running ? (downloading ? "Downloading…" : "Analysing…") : "Analyse on this device"}
            </Button>
          </div>
        </div>
        <p className="mt-6 text-center text-xs text-muted">
          Clinical decision support only. Not a diagnosis and not a cleared medical device. Refer anything that worries you.
        </p>
      </div>

      <Modal open={!!cropping} onClose={() => setCropping(null)} title="Crop to the tissue" className="sm:max-w-2xl">
        {cropping && <ImageCropper file={cropping.file} onCancel={() => setCropping(null)} onDone={(f) => { replace(cropping.id, f); setCropping(null); }} />}
      </Modal>
      <Modal open={camera} onClose={() => setCamera(false)} title="Capture photo" className="sm:max-w-2xl">
        {camera && <CameraCapture onClose={() => setCamera(false)} onCapture={(f) => { add([f]); setCamera(false); }} />}
      </Modal>
    </div>
  );
}

function Results({ results, site, label, notes, onBack, onReset }: {
  results: Analysed[]; site: string | null; label: string; notes: string; onBack: () => void; onReset: () => void;
}) {
  const ok = results.filter((r) => r.result) as Required<Analysed>[];
  const worst = ok.reduce<Required<Analysed> | null>((w, r) => (!w || r.result.probability > w.result.probability ? r : w), null);
  const [active, setActive] = useState(worst ? results.indexOf(worst) : 0);
  const [view, setView] = useState<View>("compare");
  const current = results[active];
  const r = current?.result;
  const now = new Date();

  const save = (src: string, name: string) => Object.assign(document.createElement("a"), { href: src, download: name }).click();

  return (
    <div className="container py-8 sm:py-12">
      <style>{`@media print {
        header, footer, .no-print { display: none !important; }
        body { background: #fff !important; }
        .print-only { display: block !important; }
        .card { box-shadow: none !important; break-inside: avoid; }
      }`}</style>
      <div className="mx-auto max-w-6xl">
        <div className="no-print mb-6 flex flex-wrap items-center justify-between gap-3">
          <button onClick={onBack} className="text-sm text-muted hover:text-ink">&larr; Edit photos</button>
          <div className="flex flex-wrap gap-2">
            <Button variant="secondary" onClick={() => window.print()}><Printer className="h-4 w-4" />Print / save PDF</Button>
            <Button variant="ink" onClick={onReset}><RefreshCw className="h-4 w-4" />New screening</Button>
          </div>
        </div>

        <div className="mb-8">
          <p className="eyebrow mb-2 flex items-center gap-2"><Cpu className="h-3.5 w-3.5" />Analysed on this device · {now.toLocaleString()}</p>
          <h1 className="text-2xl font-semibold tracking-tight sm:text-3xl">{label || "Screening result"}</h1>
          <p className="mt-1 text-sm text-muted">{[site, `${ok.length} of ${results.length} photo${results.length > 1 ? "s" : ""} analysed`].filter(Boolean).join(" · ")}</p>
        </div>

        {worst && (
          <div className="card mb-6 flex flex-col gap-4 p-5 sm:flex-row sm:items-center" style={{ borderColor: worst.result.color + "55" }}>
            <RiskBadge tier={worst.result.tier} />
            <div className="flex-1">
              <p className="font-semibold">{worst.result.finding}</p>
              <p className="text-sm text-muted">{worst.result.recommendation}</p>
            </div>
            <p className="text-sm text-muted">Highest score <span className="font-semibold tabular-nums text-ink">{pct(worst.result.probability)}</span></p>
          </div>
        )}

        {results.length > 1 && (
          <div className="no-print mb-6 flex gap-2 overflow-x-auto pb-1">
            {results.map((x, i) => (
              <button key={x.item.id} onClick={() => setActive(i)}
                className={cn("relative h-16 w-16 shrink-0 overflow-hidden rounded-xl ring-2 transition", i === active ? "ring-brand" : "ring-transparent opacity-70 hover:opacity-100")}>
                <img src={x.item.url} alt={`Photo ${i + 1}`} className="h-full w-full object-cover" />
                {x.result && <span className="absolute bottom-1 right-1 h-2.5 w-2.5 rounded-full ring-2 ring-white" style={{ background: x.result.color }} />}
                {x.error && <span className="absolute inset-0 grid place-items-center bg-black/50 text-white"><X className="h-5 w-5" /></span>}
              </button>
            ))}
          </div>
        )}

        <AnimatePresence mode="wait">
          <motion.div key={active} initial={{ opacity: 0, y: 6 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0 }} transition={{ duration: 0.25 }}>
            {current?.error && (
              <div className="card flex items-start gap-3 p-5">
                <AlertTriangle className="mt-0.5 h-5 w-5 shrink-0 text-moderate" />
                <div><p className="font-semibold">This photo wasn't analysed</p><p className="text-sm text-muted">{current.error}</p></div>
              </div>
            )}
            {r && (
              <div className="grid gap-6 lg:grid-cols-[1.25fr_1fr]">
                <div className="card p-4 sm:p-5">
                  <div className="no-print mb-4 overflow-x-auto">
                    <Segmented<View> value={view} onChange={setView} label="Image view" size="sm" options={[
                      { value: "compare", label: "Compare" }, { value: "gradcam", label: "Grad-CAM" },
                      { value: "segmentation", label: "Region" }, { value: "heatmap", label: "Heatmap" },
                    ]} />
                  </div>
                  {view === "compare"
                    ? <CompareSlider before={r.files.original} after={r.files.gradcam} labelAfter="Grad-CAM" />
                    : <img src={r.files[view]} alt={view} className="aspect-square w-full rounded-2xl object-cover" />}
                  <div className="no-print mt-3 flex justify-between text-xs text-muted">
                    <span>{view === "segmentation" ? "Outlined: the area the model relied on most." : "Warm colours: where the model looked."}</span>
                    <button onClick={() => save(r.files[view === "compare" ? "gradcam" : view], `oralaid-${view}.jpg`)} className="inline-flex items-center gap-1 hover:text-ink">
                      <Download className="h-3.5 w-3.5" />Save image
                    </button>
                  </div>
                  <div className="print-only mt-4 hidden">
                    <div className="grid grid-cols-3 gap-2">
                      {(["original", "gradcam", "segmentation"] as const).map((k) => <img key={k} src={r.files[k]} alt={k} className="w-full rounded-lg" />)}
                    </div>
                  </div>
                </div>

                <div className="space-y-6">
                  <div className="card p-5 sm:p-6">
                    <div className="mb-2 flex items-center justify-between"><p className="eyebrow">AI screening result</p><RiskBadge tier={r.tier} size="sm" /></div>
                    <div className="flex justify-center"><ProbabilityMeter value={r.probability} threshold={THRESHOLD} high={HIGH_THRESHOLD} /></div>
                    <p className="mt-4 text-center font-semibold">{r.finding}</p>
                    <p className="mt-1 text-center text-xs text-muted">Uncertainty ± {(r.uncertainty * 100).toFixed(1)} pts · Confidence {pct(r.confidence)}</p>
                    <div className={cn("mt-5 rounded-xl border p-4 text-sm", r.tier === "high" ? "border-high/30 bg-high/5" : "border-line bg-surface-2/50")}>
                      <p className="eyebrow mb-1">Recommendation</p>{r.recommendation}
                    </div>
                  </div>
                  <div className="card p-5 sm:p-6">
                    <div className="mb-2 flex items-center justify-between"><p className="font-semibold">Image quality</p><span className="text-xs text-muted">{r.quality.resolution}</span></div>
                    <QualityBar score={r.quality.score} />
                    {r.quality.issues.map((q) => <p key={q} className="mt-2 flex gap-1.5 text-xs text-moderate"><AlertTriangle className="h-3.5 w-3.5 shrink-0" />{q}</p>)}
                    <dl className="mt-4 grid grid-cols-2 gap-3 text-sm">
                      <div><dt className="text-xs text-muted">Focus area</dt><dd className="tabular-nums">{pct(r.region_area_ratio, 1)}</dd></div>
                      <div><dt className="text-xs text-muted">Time</dt><dd className="tabular-nums">{(r.inference_ms / 1000).toFixed(1)} s</dd></div>
                    </dl>
                  </div>
                  {notes && <div className="card p-5 text-sm"><p className="mb-1 font-semibold">Notes</p><p className="whitespace-pre-wrap text-muted">{notes}</p></div>}
                </div>
              </div>
            )}
          </motion.div>
        </AnimatePresence>

        <p className="mt-8 text-center text-xs text-muted">
          AI-generated screening support from OralNet-B4. Not a diagnosis. Confirm with clinical examination and histopathology.
          Nothing from this screening was uploaded or stored.
        </p>
      </div>
    </div>
  );
}
