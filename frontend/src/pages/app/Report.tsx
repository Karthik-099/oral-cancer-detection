import { useEffect, useState, type CSSProperties, type ReactNode } from "react";
import { Link, Navigate, useLocation, useParams } from "react-router-dom";
import { ArrowLeft, Printer } from "lucide-react";
import { api, ApiError, fileUrl, type CaseDetail } from "@/lib/api";
import { useAuth } from "@/lib/auth";
import { useSeo } from "@/lib/seo";
import { cn, formatDate, pct } from "@/lib/utils";
import { Logo } from "@/components/Logo";
import { Button, ErrorNote, Spinner } from "@/components/ui";
import { HIGH_THRESHOLD, IMPRESSIONS, SEX_LABEL, THRESHOLD } from "@/components/app/kit";

// The report is always rendered in the light palette, whatever the app theme.
const LIGHT_VARS = {
  "--bg": "250 250 247", "--surface": "255 255 255", "--surface-2": "243 243 238", "--ink": "13 17 28", "--muted": "96 103 117",
  "--line": "226 226 219", "--brand": "12 110 99", "--brand-soft": "220 239 235", "--accent": "240 98 64",
  "--low": "16 160 112", "--moderate": "217 140 8", "--high": "220 56 56", colorScheme: "light",
} as CSSProperties;

const PRINT_CSS = `
@page { size: A4; margin: 14mm 12mm; }
@media print {
  html, body { background: #fff !important; }
  .report-toolbar { display: none !important; }
  .report-shell { background: #fff !important; padding: 0 !important; min-height: 0 !important; }
  .report-paper { box-shadow: none !important; border: 0 !important; border-radius: 0 !important; padding: 0 !important; max-width: none !important; width: auto !important; }
  .report-avoid { break-inside: avoid; page-break-inside: avoid; }
  * { -webkit-print-color-adjust: exact; print-color-adjust: exact; }
}
`;

const TIER_STYLE = {
  low: { box: "border-low/40 bg-low/[0.08]", text: "text-low" },
  moderate: { box: "border-moderate/40 bg-moderate/[0.08]", text: "text-moderate" },
  high: { box: "border-high/40 bg-high/[0.08]", text: "text-high" },
};

export default function Report() {
  const { id = "" } = useParams();
  useSeo(`Report ${id.slice(0, 8)}`, undefined, { noindex: true });
  const location = useLocation();
  const { user, status, bootstrap } = useAuth();
  const [data, setData] = useState<CaseDetail | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => { bootstrap(); }, [bootstrap]);
  useEffect(() => {
    if (status !== "ready" || !user) return;
    api.case(id).then(setData).catch((e) => {
      if (e instanceof ApiError && e.status === 401) return; // signOut listener redirects
      setError(e instanceof ApiError && e.status === 404 ? "Case not found." : "Couldn't load this report. Please try again.");
    });
  }, [id, status, user]);

  if (status !== "ready") return <div className="grid min-h-screen place-items-center bg-white"><Spinner /></div>;
  if (!user) return <Navigate to="/login" replace state={{ from: location.pathname }} />;

  return (
    <div style={LIGHT_VARS} className="report-shell min-h-screen bg-[#ecece6] px-3 py-4 text-ink sm:px-6 sm:py-8">
      <style>{PRINT_CSS}</style>
      <div className="report-toolbar mx-auto mb-4 flex max-w-[210mm] items-center justify-between gap-3">
        <Link to={`/app/cases/${id}`} className="inline-flex items-center gap-1.5 text-sm text-muted hover:text-ink"><ArrowLeft className="h-4 w-4" />Back to case</Link>
        <Button onClick={() => window.print()} disabled={!data}><Printer className="h-4 w-4" />Print / Save as PDF</Button>
      </div>

      <article className="report-paper mx-auto w-full max-w-[210mm] rounded-sm border border-line bg-white p-6 shadow-lift sm:p-[14mm]">
        {error ? <ErrorNote>{error}</ErrorNote> : !data ? (
          <div className="grid min-h-[60vh] place-items-center"><Spinner /></div>
        ) : <ReportBody data={data} clinician={user.full_name} organization={user.organization} />}
      </article>
    </div>
  );
}

function ReportBody({ data, clinician, organization }: { data: CaseDetail; clinician: string; organization?: string | null }) {
  const p = data.patient;
  const t = TIER_STYLE[data.tier];
  const worst = data.images.reduce((a, b) => (b.probability > a.probability ? b : a), data.images[0]);
  const fb = data.feedback;
  const verdict = fb ? { agree: "Agrees with AI result", disagree: "Disagrees with AI result", unsure: "Unsure" }[fb.verdict] : null;

  return (
    <div className="text-[13px] leading-relaxed">
      <header className="flex items-start justify-between gap-4 border-b border-ink pb-4">
        <div>
          <Logo />
          <h1 className="mt-3 text-xl font-semibold tracking-tight">AI Oral Screening Report</h1>
          <p className="text-xs text-muted">Explainable decision support · OralNet-B4</p>
        </div>
        <dl className="text-right text-xs">
          <dt className="text-muted">Case ID</dt><dd className="font-mono font-medium">{data.id}</dd>
          <dt className="mt-1.5 text-muted">Analysed</dt><dd>{formatDate(data.created_at)}</dd>
          <dt className="mt-1.5 text-muted">Report generated</dt><dd>{new Date().toLocaleString(undefined, { dateStyle: "medium", timeStyle: "short" })}</dd>
        </dl>
      </header>

      <section className="report-avoid mt-5 grid grid-cols-2 gap-x-6 gap-y-2 sm:grid-cols-4">
        <Info label="Patient" value={p?.name || "—"} />
        <Info label="Reference" value={p?.reference || "—"} mono />
        <Info label="Age / sex" value={[p?.age ?? null, p?.sex ? SEX_LABEL[p.sex] ?? p.sex : null].filter((v) => v !== null && v !== "").join(" / ") || "—"} />
        <Info label="Anatomical site" value={data.site || "Not specified"} />
        {!!p?.risk_factors?.length && <Info label="Risk factors" value={p.risk_factors.join(", ")} className="col-span-2 sm:col-span-4" />}
        <Info label="Requested by" value={[clinician, organization].filter(Boolean).join(", ")} className="col-span-2" />
        <Info label="Case status" value={data.status.replace(/^./, (c) => c.toUpperCase())} />
        <Info label="Images analysed" value={String(data.images.length)} />
      </section>

      <section className={cn("report-avoid mt-5 grid gap-4 rounded-lg border-2 p-4 sm:grid-cols-[auto_1fr]", t.box)}>
        <div className="sm:border-r sm:border-line sm:pr-5">
          <p className="text-[10px] font-semibold uppercase tracking-[0.14em] text-muted">Risk tier</p>
          <p className={cn("mt-1 text-xl font-bold", t.text)}>{data.label}</p>
          <p className="mt-2 text-[10px] font-semibold uppercase tracking-[0.14em] text-muted">Suspicion score</p>
          <p className="font-display text-4xl leading-none tabular-nums">{Math.round(data.probability * 100)}<span className="text-lg text-muted">%</span></p>
          <ScaleBar value={data.probability} />
        </div>
        <div>
          <p className="text-[10px] font-semibold uppercase tracking-[0.14em] text-muted">Finding</p>
          <p className="mt-0.5 text-[15px] font-medium">{data.finding}</p>
          <p className="mt-3 text-[10px] font-semibold uppercase tracking-[0.14em] text-muted">Recommendation</p>
          <p className={cn("mt-0.5", data.tier === "high" && "font-semibold")}>{data.recommendation}</p>
          {worst && (
            <p className="mt-3 text-xs text-muted">
              Uncertainty ± {(worst.uncertainty * 100).toFixed(1)} pts · confidence {pct(worst.confidence)} · thresholds: flag {Math.round((worst.threshold || THRESHOLD) * 100)}%, high {Math.round(HIGH_THRESHOLD * 100)}%
            </p>
          )}
        </div>
      </section>

      <section className="mt-6">
        <SectionTitle>Images and explanations</SectionTitle>
        <div className="space-y-4">
          {data.images.map((im, i) => (
            <div key={im.index} className="report-avoid rounded-lg border border-line p-3">
              <div className="mb-2 flex flex-wrap items-center justify-between gap-2 text-xs">
                <span className="font-medium">Image {i + 1} of {data.images.length}</span>
                <span className="text-muted">
                  Score <strong className="text-ink tabular-nums">{Math.round(im.probability * 100)}%</strong> · {im.label} · quality {im.quality.score}/100 · attention {pct(im.attention_area_ratio, 1)} · region {pct(im.region_area_ratio, 1)}
                </span>
              </div>
              <div className="grid grid-cols-3 gap-2">
                {([["Original", im.files.original], ["Grad-CAM attention", im.files.gradcam], ["Localised region", im.files.segmentation]] as const).map(([label, src]) => (
                  <figure key={label}>
                    <img src={fileUrl(src)} alt={label} className="aspect-square w-full rounded border border-line object-cover" />
                    <figcaption className="mt-1 text-center text-[10px] text-muted">{label}</figcaption>
                  </figure>
                ))}
              </div>
              {(im.quality.issues.length > 0) && (
                <p className="mt-2 text-[11px] text-moderate">Quality notes: {im.quality.issues.join("; ")}</p>
              )}
            </div>
          ))}
        </div>
      </section>

      <section className="report-avoid mt-6 grid gap-5 sm:grid-cols-2">
        <div>
          <SectionTitle>Clinician review</SectionTitle>
          {fb ? (
            <dl className="space-y-1.5">
              <Row label="Verdict" value={verdict ?? "—"} />
              <Row label="Clinical impression" value={IMPRESSIONS.find((x) => x.value === fb.clinical_impression)?.label ?? "—"} />
              <Row label="Histopathology" value={fb.histopathology || "—"} />
              {fb.explanation_rating ? <Row label="Explanation usefulness" value={`${fb.explanation_rating} / 5`} /> : null}
              {fb.comment && <Row label="Comment" value={fb.comment} />}
            </dl>
          ) : <p className="text-muted">Not yet reviewed by a clinician.</p>}
        </div>
        <div>
          <SectionTitle>Clinical notes</SectionTitle>
          <p className="whitespace-pre-wrap">{data.notes || <span className="text-muted">None recorded.</span>}</p>
        </div>
      </section>

      <section className="report-avoid mt-8 grid grid-cols-2 gap-8 text-xs text-muted">
        <div><div className="h-10 border-b border-line" /><p className="mt-1">Clinician signature</p></div>
        <div><div className="h-10 border-b border-line" /><p className="mt-1">Date</p></div>
      </section>

      <footer className="report-avoid mt-8 border-t border-ink pt-3 text-[11px] text-muted">
        <p className="font-semibold text-ink">AI-generated screening support. Not a diagnosis. Confirm with clinical examination and histopathology.</p>
        <p className="mt-1">
          Grad-CAM and region overlays show where the model focused; they are not lesion boundaries. Generated by OralAID for {[clinician, organization].filter(Boolean).join(", ")}.
          Case {data.id}.
        </p>
      </footer>
    </div>
  );
}

function SectionTitle({ children }: { children: ReactNode }) {
  return <h2 className="mb-2 border-b border-line pb-1 text-[11px] font-semibold uppercase tracking-[0.14em] text-muted">{children}</h2>;
}

function Info({ label, value, mono, className }: { label: string; value: string; mono?: boolean; className?: string }) {
  return (
    <div className={className}>
      <p className="text-[10px] font-semibold uppercase tracking-[0.12em] text-muted">{label}</p>
      <p className={cn("font-medium", mono && "font-mono text-xs")}>{value}</p>
    </div>
  );
}

function Row({ label, value }: { label: string; value: string }) {
  return (
    <div className="grid grid-cols-[130px_1fr] gap-2">
      <dt className="text-muted">{label}</dt><dd className="whitespace-pre-wrap">{value}</dd>
    </div>
  );
}

function ScaleBar({ value }: { value: number }) {
  return (
    <div className="relative mt-2 w-40">
      <div className="flex h-1.5 overflow-hidden rounded-full">
        <span className="bg-low/50" style={{ width: `${THRESHOLD * 100}%` }} />
        <span className="bg-moderate/50" style={{ width: `${(HIGH_THRESHOLD - THRESHOLD) * 100}%` }} />
        <span className="flex-1 bg-high/50" />
      </div>
      <span className="absolute -top-1 h-3.5 w-0.5 -translate-x-1/2 rounded bg-ink" style={{ left: `${value * 100}%` }} />
    </div>
  );
}
