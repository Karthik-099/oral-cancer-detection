import { useCallback, useEffect, useRef, useState, type DragEvent, type FormEvent, type ReactNode } from "react";
import { useNavigate, useSearchParams } from "react-router-dom";
import { AnimatePresence, motion } from "framer-motion";
import {
  Camera, Check, Crop, Hand, ImageOff, ImagePlus, Info, Lightbulb, Loader2, Maximize, Ruler, ScanLine, Sparkles, Sun, Trash2, Upload, UserRound, Wind, X,
} from "lucide-react";
import { ApiError, api, type PatientRow } from "@/lib/api";
import { useSeo } from "@/lib/seo";
import { cn } from "@/lib/utils";
import { PageHeader } from "@/components/AppShell";
import { Button, Card, ErrorNote, Field } from "@/components/ui";
import { Modal } from "@/components/Modal";
import { MouthMap, SITES, type Site } from "@/components/MouthMap";
import { ImageCropper } from "@/components/ImageCropper";
import { toast } from "@/components/Toast";
import { CameraCapture } from "@/components/app/CameraCapture";
import { Chip, patientName, RISK_FACTORS, Segmented, Toggle } from "@/components/app/kit";

const MAX_FILES = 6;
const MAX_BYTES = 15 * 1024 * 1024;
const ACCEPT = "image/jpeg,image/png,image/webp,image/heic,image/heif,.heic,.heif";
const OK_TYPES = ["image/jpeg", "image/png", "image/webp", "image/heic", "image/heif"];

interface Item { id: string; file: File; url: string }

const SEX_OPTIONS = [
  { value: "female", label: "Female" },
  { value: "male", label: "Male" },
  { value: "other", label: "Other" },
] as const;
type Sex = (typeof SEX_OPTIONS)[number]["value"];

const TIPS = [
  { icon: Sun, text: "Use bright, even light. Avoid harsh flash glare." },
  { icon: Wind, text: "Gently dry the area so saliva doesn't reflect." },
  { icon: Hand, text: "Retract the cheek or tongue to expose the site." },
  { icon: Maximize, text: "Fill the frame with the lesion and nearby tissue." },
  { icon: Ruler, text: "Keep fingers, instruments and background out of shot. Background objects can mislead the model." },
];

const STEPS = ["Checking image quality", "Running OralNet-B4 with test-time augmentation", "Computing Grad-CAM", "Localising region"];

function isAcceptable(f: File) {
  return OK_TYPES.includes(f.type) || /\.(jpe?g|png|webp|heic|heif)$/i.test(f.name);
}

function friendlyError(e: unknown): string {
  if (!(e instanceof ApiError)) return "Something went wrong while analysing. Please try again.";
  switch (e.status) {
    case 0: return e.message;
    case 422: return /oral|intraoral|mouth/i.test(e.message)
      ? `${e.message} Retake a close-up of the oral tissue, cropping out background.`
      : e.message || "Some details weren't valid. Check the form and try again.";
    case 413: return "One of the images is too large. Each image must be 15 MB or smaller.";
    case 415: return "That file type isn't supported. Use JPEG, PNG, WebP or HEIC images.";
    case 429: return "You've run several analyses in a short time. Wait a minute and try again.";
    case 503: return "The AI model is still warming up. Give it a few seconds and try again.";
    default: return e.message || "Analysis failed. Please try again.";
  }
}

export default function NewCase() {
  useSeo("New analysis", undefined, { noindex: true });
  const navigate = useNavigate();
  const [params] = useSearchParams();
  const welcome = params.get("welcome") === "1";

  const [items, setItems] = useState<Item[]>([]);
  const [drag, setDrag] = useState(false);
  const [camera, setCamera] = useState(false);
  const [cropping, setCropping] = useState<Item | null>(null);
  const [site, setSite] = useState<Site | null>(null);
  const [reference, setReference] = useState(params.get("ref") ?? "");
  const [name, setName] = useState("");
  const [age, setAge] = useState("");
  const [sex, setSex] = useState<Sex | null>(null);
  const [factors, setFactors] = useState<string[]>([]);
  const [notes, setNotes] = useState("");
  const [withLime, setWithLime] = useState(false);
  const [existing, setExisting] = useState<PatientRow | null>(null);
  const [submitting, setSubmitting] = useState(false);
  const [done, setDone] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [showWelcome, setShowWelcome] = useState(welcome);
  const inputRef = useRef<HTMLInputElement>(null);
  const itemsRef = useRef(items);
  itemsRef.current = items;

  useEffect(() => () => itemsRef.current.forEach((i) => URL.revokeObjectURL(i.url)), []);

  // Link to an existing patient when the reference matches one exactly.
  useEffect(() => {
    const ref = reference.trim();
    if (!ref) { setExisting(null); return; }
    const t = setTimeout(() => {
      api.patients(ref).then((r) => {
        const match = r.patients.find((p) => (p.reference ?? "").toLowerCase() === ref.toLowerCase()) ?? null;
        setExisting(match);
        if (match) {
          setName((n) => n || match.name || "");
          setAge((a) => a || (match.age !== null && match.age !== undefined ? String(match.age) : ""));
          setSex((s) => s ?? ((match.sex as Sex | null | undefined) || null));
          setFactors((f) => (f.length ? f : match.risk_factors ?? []));
        }
      }).catch(() => setExisting(null));
    }, 350);
    return () => clearTimeout(t);
  }, [reference]);

  const addFiles = useCallback((list: FileList | File[]) => {
    const incoming = Array.from(list);
    const room = MAX_FILES - itemsRef.current.length;
    const accepted: Item[] = [];
    let rejectedType = 0, rejectedSize = 0;
    for (const f of incoming) {
      if (!isAcceptable(f)) { rejectedType++; continue; }
      if (f.size > MAX_BYTES) { rejectedSize++; continue; }
      if (accepted.length >= room) break;
      accepted.push({ id: `${Date.now()}-${Math.random().toString(36).slice(2)}`, file: f, url: URL.createObjectURL(f) });
    }
    if (rejectedType) toast.error(`${rejectedType} file${rejectedType > 1 ? "s" : ""} skipped. Use JPEG, PNG, WebP or HEIC.`);
    if (rejectedSize) toast.error(`${rejectedSize} file${rejectedSize > 1 ? "s were" : " was"} over 15 MB.`);
    if (incoming.length - rejectedType - rejectedSize > accepted.length) toast.error(`Up to ${MAX_FILES} images per case.`);
    if (accepted.length) { setItems((prev) => [...prev, ...accepted]); setError(null); }
  }, []);

  const remove = (id: string) => setItems((prev) => {
    const it = prev.find((p) => p.id === id);
    if (it) URL.revokeObjectURL(it.url);
    return prev.filter((p) => p.id !== id);
  });

  const replace = (id: string, file: File) => setItems((prev) => prev.map((p) => {
    if (p.id !== id) return p;
    URL.revokeObjectURL(p.url);
    return { ...p, file, url: URL.createObjectURL(file) };
  }));

  const onDrop = (e: DragEvent) => {
    e.preventDefault();
    setDrag(false);
    if (e.dataTransfer.files?.length) addFiles(e.dataTransfer.files);
  };

  const ageNum = age === "" ? null : Number(age);
  const ageInvalid = ageNum !== null && (!Number.isInteger(ageNum) || ageNum < 0 || ageNum > 120);

  const submit = async (e: FormEvent) => {
    e.preventDefault();
    if (!items.length) { setError("Add at least one intraoral image."); return; }
    if (ageInvalid) { setError("Age must be a whole number between 0 and 120."); return; }
    setError(null);
    const fd = new FormData();
    items.forEach((i) => fd.append("files", i.file, i.file.name));
    if (site) fd.append("site", site);
    if (reference.trim()) fd.append("patient_reference", reference.trim());
    if (name.trim()) fd.append("patient_name", name.trim());
    if (ageNum !== null) fd.append("patient_age", String(ageNum));
    if (sex) fd.append("patient_sex", sex);
    if (factors.length) fd.append("risk_factors", JSON.stringify(factors));
    if (notes.trim()) fd.append("notes", notes.trim());
    fd.append("with_lime", withLime ? "true" : "false");
    setSubmitting(true);
    try {
      const res = await api.createCase(fd);
      setDone(true);
      await new Promise((r) => setTimeout(r, 450));
      toast.success("Analysis complete");
      navigate(`/app/cases/${res.id}`);
    } catch (err) {
      const msg = friendlyError(err);
      setError(msg);
      toast.error(msg);
      setSubmitting(false);
    }
  };
  return (
    <form onSubmit={submit} noValidate>
      <PageHeader title="New analysis" subtitle="Add intraoral photos, mark the site and record the clinical context." />

      <AnimatePresence>
        {showWelcome && (
          <motion.div initial={{ opacity: 0, y: -6 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, height: 0, marginBottom: 0 }}
            className="relative mb-6 overflow-hidden rounded-2xl border border-brand/25 bg-brand-soft/60 p-5">
            <button type="button" onClick={() => setShowWelcome(false)} aria-label="Dismiss welcome"
              className="absolute right-3 top-3 grid h-8 w-8 place-items-center rounded-lg text-muted hover:text-ink"><X className="h-4 w-4" /></button>
            <div className="flex gap-4 pr-8">
              <div className="grid h-10 w-10 shrink-0 place-items-center rounded-xl bg-brand text-white dark:text-bg"><Sparkles className="h-5 w-5" /></div>
              <div>
                <h2 className="text-[15px] font-semibold">Welcome to OralAID</h2>
                <p className="mt-1 text-sm text-muted">
                  Run your first analysis below. You'll get a suspicion score, a Grad-CAM map of where the model looked and an image quality check.
                  Only the images are required; everything else helps you track patients over time.
                </p>
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      <div className="grid gap-6 lg:grid-cols-[minmax(0,1fr)_300px]">
        <div className="min-w-0 space-y-6">
          <Section n={1} title="Images" desc={`Up to ${MAX_FILES} photos of the same site. The most concerning image sets the case result.`} done={items.length > 0}>
            <div
              onDragOver={(e) => { e.preventDefault(); setDrag(true); }}
              onDragLeave={() => setDrag(false)}
              onDrop={onDrop}
              className={cn(
                "relative flex flex-col items-center justify-center rounded-2xl border-2 border-dashed px-6 py-10 text-center transition",
                drag ? "border-brand bg-brand/5" : "border-line bg-surface-2/40",
                items.length >= MAX_FILES && "pointer-events-none opacity-50",
              )}
            >
              <div className={cn("grid h-12 w-12 place-items-center rounded-2xl bg-surface text-muted shadow-soft transition", drag && "scale-110 text-brand")}>
                <Upload className="h-5 w-5" />
              </div>
              <p className="mt-4 text-sm font-medium">{drag ? "Drop to add" : "Drag photos here"}</p>
              <p className="mt-1 text-xs text-muted">JPEG, PNG, WebP or HEIC · 15 MB each</p>
              <div className="mt-5 flex flex-wrap justify-center gap-2">
                <Button type="button" variant="secondary" onClick={() => inputRef.current?.click()} disabled={items.length >= MAX_FILES}>
                  <ImagePlus className="h-4 w-4" />Choose files
                </Button>
                <Button type="button" variant="secondary" onClick={() => setCamera(true)} disabled={items.length >= MAX_FILES}>
                  <Camera className="h-4 w-4" />Use camera
                </Button>
              </div>
              <input ref={inputRef} type="file" accept={ACCEPT} multiple className="sr-only" tabIndex={-1} aria-label="Choose images"
                onChange={(e) => { if (e.target.files) addFiles(e.target.files); e.target.value = ""; }} />
            </div>

            {items.length > 0 && (
              <div className="mt-4">
                <div className="mb-2 flex items-center justify-between text-xs text-muted">
                  <span className="tabular-nums">{items.length} of {MAX_FILES} images</span>
                  <button type="button" className="hover:text-ink" onClick={() => { items.forEach((i) => URL.revokeObjectURL(i.url)); setItems([]); }}>Clear all</button>
                </div>
                <ul className="grid grid-cols-2 gap-3 sm:grid-cols-3">
                  <AnimatePresence initial={false}>
                    {items.map((it, idx) => (
                      <motion.li key={it.id} layout initial={{ opacity: 0, scale: 0.96 }} animate={{ opacity: 1, scale: 1 }} exit={{ opacity: 0, scale: 0.94 }}
                        className="group relative aspect-square overflow-hidden rounded-xl border border-line bg-surface-2">
                        <Preview src={it.url} name={it.file.name} />
                        {idx === 0 && <span className="absolute left-2 top-2 rounded-md bg-black/55 px-1.5 py-0.5 text-[10px] font-medium text-white backdrop-blur">Primary</span>}
                        <div className="absolute inset-x-0 bottom-0 flex items-center justify-between gap-1 bg-gradient-to-t from-black/70 to-transparent p-2 pt-6">
                          <span className="truncate text-[10px] text-white/80">{(it.file.size / 1024 / 1024).toFixed(1)} MB</span>
                          <div className="flex gap-1">
                            <button type="button" onClick={() => setCropping(it)} aria-label={`Crop image ${idx + 1}`}
                              className="flex h-7 items-center gap-1 rounded-md bg-white/90 px-2 text-[11px] font-medium text-ink hover:bg-white">
                              <Crop className="h-3 w-3" />Crop
                            </button>
                            <button type="button" onClick={() => remove(it.id)} aria-label={`Remove image ${idx + 1}`}
                              className="grid h-7 w-7 place-items-center rounded-md bg-white/90 text-ink hover:bg-white hover:text-high">
                              <Trash2 className="h-3.5 w-3.5" />
                            </button>
                          </div>
                        </div>
                      </motion.li>
                    ))}
                  </AnimatePresence>
                </ul>
              </div>
            )}
          </Section>

          <Section n={2} title="Anatomical site" desc="Tap the region on the map or choose from the list." done={!!site}>
            <div className="grid items-center gap-6 sm:grid-cols-[220px_1fr]">
              <MouthMap value={site} onChange={setSite} className="mx-auto max-w-[220px]" />
              <div className="flex flex-wrap gap-2">
                {SITES.map((s) => <Chip key={s} active={site === s} onClick={() => setSite(site === s ? null : s)}>{s}</Chip>)}
              </div>
            </div>
          </Section>

          <Section n={3} title="Patient" desc="Optional, but the reference links visits so you can follow change over time." done={!!reference.trim()}>
            <div className="grid gap-4 sm:grid-cols-2">
              <Field label="Reference / MRN" hint={existing ? undefined : "Use your clinic's own identifier. Avoid names if you can."}>
                <input className="input font-mono" value={reference} onChange={(e) => setReference(e.target.value)} maxLength={64} placeholder="e.g. MRN-20431" autoComplete="off" />
              </Field>
              <Field label="Name (optional)">
                <input className="input" value={name} onChange={(e) => setName(e.target.value)} maxLength={120} autoComplete="off" disabled={!!existing} />
              </Field>
              {existing && (
                <div className="flex items-start gap-2 rounded-xl border border-brand/25 bg-brand/5 px-3 py-2.5 text-xs text-ink sm:col-span-2">
                  <UserRound className="mt-0.5 h-3.5 w-3.5 shrink-0 text-brand" />
                  <span>
                    This adds a new visit for <strong>{patientName(existing)}</strong>
                    {existing.case_count ? ` (${existing.case_count} previous case${existing.case_count > 1 ? "s" : ""})` : ""}.
                    Stored patient details are kept.
                  </span>
                </div>
              )}
              <Field label="Age" error={ageInvalid ? "0–120" : null}>
                <input className="input tabular-nums" inputMode="numeric" value={age} disabled={!!existing}
                  onChange={(e) => setAge(e.target.value.replace(/[^\d]/g, "").slice(0, 3))} placeholder="Years" />
              </Field>
              <div>
                <span className="label">Sex</span>
                <Segmented label="Sex" value={sex} onChange={(v) => setSex(sex === v ? null : v)} options={SEX_OPTIONS} className={cn("w-full", existing && "pointer-events-none opacity-60")} />
              </div>
              <div className="sm:col-span-2">
                <span className="label">Risk factors</span>
                <div className={cn("flex flex-wrap gap-2", existing && "pointer-events-none opacity-60")}>
                  {RISK_FACTORS.map((f) => {
                    const on = factors.includes(f);
                    return (
                      <Chip key={f} active={on} onClick={() => setFactors((prev) => (on ? prev.filter((x) => x !== f) : [...prev, f]))}>
                        {on && <Check className="h-3.5 w-3.5" />}{f}
                      </Chip>
                    );
                  })}
                </div>
              </div>
            </div>
          </Section>

          <Section n={4} title="Clinical notes" desc="History, duration, symptoms, appearance. Saved with the case and printed on the report." done={!!notes.trim()}>
            <textarea className="input min-h-[120px] resize-y" value={notes} onChange={(e) => setNotes(e.target.value)} maxLength={4000}
              placeholder="e.g. Non-healing white patch, left lateral tongue, 6 weeks. Painless. Smoker 20 pack-years." aria-label="Clinical notes" />
            <p className="mt-1.5 text-right text-xs tabular-nums text-muted">{notes.length} / 4000</p>
          </Section>

          <Card className="flex items-start gap-4 p-5">
            <Toggle checked={withLime} onChange={setWithLime} label="Include LIME explanation now" />
            <div className="min-w-0">
              <p className="text-sm font-medium">Include LIME explanation now <span className="font-normal text-muted">(slower)</span></p>
              <p className="mt-0.5 text-xs text-muted">Adds a second, perturbation-based explanation. Takes 15–30 s per image. You can also generate it later from the case.</p>
            </div>
          </Card>

          <div className="sticky bottom-0 z-20 -mx-4 border-t border-line bg-bg/90 px-4 py-4 backdrop-blur-xl sm:static sm:mx-0 sm:border-0 sm:bg-transparent sm:p-0 sm:backdrop-blur-none">
            {error && <div className="mb-3"><ErrorNote>{error}</ErrorNote></div>}
            <div className="flex items-center justify-between gap-3">
              <p className="hidden text-xs text-muted sm:block">Decision support only. Confirm findings clinically.</p>
              <Button type="submit" size="lg" className="w-full sm:w-auto" loading={submitting} disabled={!items.length}>
                {!submitting && <ScanLine className="h-4 w-4" />}Analyse {items.length > 1 ? `${items.length} images` : "image"}
              </Button>
            </div>
          </div>
        </div>

        <aside className="space-y-4 lg:sticky lg:top-10 lg:self-start">
          <Card className="p-5">
            <div className="flex items-center gap-2"><Lightbulb className="h-4 w-4 text-moderate" /><h3 className="text-sm font-semibold">Capture tips</h3></div>
            <ul className="mt-4 space-y-3">
              {TIPS.map((t) => (
                <li key={t.text} className="flex gap-3 text-[13px] text-muted">
                  <t.icon className="mt-0.5 h-4 w-4 shrink-0 text-ink/60" />{t.text}
                </li>
              ))}
            </ul>
          </Card>
          <div className="flex gap-3 rounded-2xl border border-line p-4 text-xs text-muted">
            <Info className="h-4 w-4 shrink-0" />
            Images that don't look like oral tissue are rejected before analysis. Crop out background to improve accuracy.
          </div>
        </aside>
      </div>

      <Modal open={camera} onClose={() => setCamera(false)} title="Capture photo" className="sm:max-w-2xl">
        {camera && <CameraCapture onCapture={(f) => { addFiles([f]); toast.success("Photo added"); }} onClose={() => setCamera(false)} />}
      </Modal>

      <Modal open={!!cropping} onClose={() => setCropping(null)} title="Crop image" className="sm:max-w-2xl">
        {cropping && (
          <ImageCropper file={cropping.file} onCancel={() => setCropping(null)}
            onDone={(f) => { replace(cropping.id, f); setCropping(null); if (f !== cropping.file) toast.success("Image cropped"); }} />
        )}
      </Modal>

      <AnimatePresence>{submitting && <AnalysingOverlay image={items[0]?.url} done={done} count={items.length} withLime={withLime} />}</AnimatePresence>
    </form>
  );
}

function Section({ n, title, desc, done, children }: { n: number; title: string; desc: string; done?: boolean; children: ReactNode }) {
  return (
    <Card className="p-5 sm:p-6">
      <div className="mb-5 flex gap-3">
        <span className={cn("grid h-7 w-7 shrink-0 place-items-center rounded-full text-xs font-semibold tabular-nums transition",
          done ? "bg-brand text-white dark:text-bg" : "border border-line text-muted")}>
          {done ? <Check className="h-3.5 w-3.5" /> : n}
        </span>
        <div>
          <h2 className="text-[15px] font-semibold tracking-tight">{title}</h2>
          <p className="mt-0.5 text-[13px] text-muted">{desc}</p>
        </div>
      </div>
      {children}
    </Card>
  );
}

function Preview({ src, name }: { src: string; name: string }) {
  const [failed, setFailed] = useState(false);
  if (failed) {
    return (
      <div className="flex h-full w-full flex-col items-center justify-center gap-1 p-3 text-center text-muted">
        <ImageOff className="h-5 w-5" /><span className="line-clamp-2 text-[11px]">{name}</span>
        <span className="text-[10px]">Preview unavailable</span>
      </div>
    );
  }
  return <img src={src} alt={name} className="h-full w-full object-cover" onError={() => setFailed(true)} />;
}

function AnalysingOverlay({ image, done, count, withLime }: { image?: string; done: boolean; count: number; withLime: boolean }) {
  const [step, setStep] = useState(0);
  useEffect(() => {
    const per = (withLime ? 4200 : 1700) * Math.max(1, Math.min(count, 3));
    const t = setInterval(() => setStep((s) => Math.min(STEPS.length - 1, s + 1)), per / 2);
    return () => clearInterval(t);
  }, [count, withLime]);
  const current = done ? STEPS.length : step;

  return (
    <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
      className="fixed inset-0 z-[85] grid place-items-center bg-bg/90 p-4 backdrop-blur-xl" role="status" aria-live="polite">
      <motion.div initial={{ y: 12, scale: 0.98 }} animate={{ y: 0, scale: 1 }} className="w-full max-w-sm">
        <div className="relative mx-auto aspect-square w-full max-w-[280px] overflow-hidden rounded-3xl border border-line bg-surface-2 shadow-lift">
          {image && <img src={image} alt="" className="h-full w-full object-cover" />}
          <div className="absolute inset-0 bg-[linear-gradient(rgb(var(--brand)/.12)_1px,transparent_1px),linear-gradient(90deg,rgb(var(--brand)/.12)_1px,transparent_1px)] bg-[size:24px_24px]" />
          {!done && (
            <div className="absolute inset-0 animate-scan">
              <div className="h-1/3 bg-gradient-to-b from-transparent via-brand/25 to-brand/50" />
              <div className="h-0.5 bg-brand shadow-[0_0_16px_4px_rgb(var(--brand)/.6)]" />
            </div>
          )}
          <AnimatePresence>
            {done && (
              <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="absolute inset-0 grid place-items-center bg-brand/30">
                <motion.span initial={{ scale: 0.6 }} animate={{ scale: 1 }} className="grid h-14 w-14 place-items-center rounded-full bg-surface text-brand shadow-lift">
                  <Check className="h-7 w-7" />
                </motion.span>
              </motion.div>
            )}
          </AnimatePresence>
        </div>
        <h2 className="display mt-8 text-center text-3xl">{done ? "Analysis complete" : "Analysing"}</h2>
        <p className="mt-1 text-center text-sm text-muted">
          {count > 1 ? `${count} images` : "1 image"}{withLime ? " · including LIME" : ""}
        </p>
        <ol className="mx-auto mt-6 max-w-[300px] space-y-3">
          {STEPS.map((s, i) => (
            <li key={s} className={cn("flex items-center gap-3 text-sm transition-colors", i <= current ? "text-ink" : "text-muted/60")}>
              <span className="grid h-5 w-5 shrink-0 place-items-center">
                {i < current ? <Check className="h-4 w-4 text-brand" />
                  : i === current ? <Loader2 className="h-4 w-4 animate-spin text-brand" />
                    : <span className="h-1.5 w-1.5 rounded-full bg-line" />}
              </span>
              {s}
            </li>
          ))}
        </ol>
      </motion.div>
    </motion.div>
  );
}

