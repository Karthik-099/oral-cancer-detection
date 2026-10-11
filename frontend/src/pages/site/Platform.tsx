import type { ReactNode } from "react";
import { Link } from "react-router-dom";
import {
  ArrowRight, Braces, Camera, ClipboardCheck, Crop, Database, FileSpreadsheet, FileText, Fingerprint, History, KeyRound,
  Lock, LogOut, MapPin, RefreshCw, ScanSearch, ShieldAlert, Sparkles, Timer, Users,
} from "lucide-react";
import { useSeo } from "@/lib/seo";
import { cn } from "@/lib/utils";
import { Reveal } from "@/components/Reveal";
import { Badge, ButtonLink, Kbd } from "@/components/ui";
import { MouthMap } from "@/components/MouthMap";
import { CTABand, CheckList, IconTile, PageHero, Section, SectionHeading } from "@/components/site/marketing";
import { AnalysisMockup, CaseListMockup, FeedbackMockup, ReportMockup, TimelineMockup } from "@/components/site/ProductMockup";

function FeatureRow({ eyebrow, title, body, bullets, visual, flip = false }: {
  eyebrow: string; title: ReactNode; body: string; bullets: ReactNode[]; visual: ReactNode; flip?: boolean;
}) {
  return (
    <div className="grid items-center gap-10 lg:grid-cols-2 lg:gap-20">
      <Reveal className={cn(flip && "lg:order-2")}>
        <p className="eyebrow mb-4 text-brand">{eyebrow}</p>
        <h2 className="display text-4xl sm:text-5xl">{title}</h2>
        <p className="mt-5 text-base leading-relaxed text-muted">{body}</p>
        <CheckList items={bullets} className="mt-7" />
      </Reveal>
      <Reveal delay={0.08} className={cn("relative", flip && "lg:order-1")}>
        <div aria-hidden className="absolute -inset-6 rounded-[36px] bg-gradient-to-br from-brand/10 via-transparent to-accent/5 blur-2xl" />
        <div className="relative">{visual}</div>
      </Reveal>
    </div>
  );
}

const WORKFLOW = [
  { icon: <Camera />, t: "Capture", d: "Upload or camera capture, 1 to 6 images" },
  { icon: <ClipboardCheck />, t: "Quality gate", d: "Sharpness, exposure, resolution" },
  { icon: <Sparkles />, t: "Inference", d: "OralNet-B4 with multi-scale and flip TTA" },
  { icon: <ScanSearch />, t: "Explain", d: "Grad-CAM, region outline, LIME" },
  { icon: <Users />, t: "Review", d: "Clinician verdict, impression, referral" },
  { icon: <RefreshCw />, t: "Learn", d: "Feedback feeds future retraining" },
];

const SECURITY = [
  { icon: <Lock />, t: "Signed, expiring image URLs", d: "Image links are signed and expire after one hour." },
  { icon: <History />, t: "Audit log", d: "Every view, export, edit and key action is recorded." },
  { icon: <ShieldAlert />, t: "Lockout and rate limiting", d: "Repeated failed sign-ins lock the account; APIs are rate limited." },
  { icon: <LogOut />, t: "Sign out everywhere", d: "Invalidate every session instantly from Settings." },
  { icon: <Fingerprint />, t: "bcrypt password hashing", d: "Passwords are never stored in plain text." },
  { icon: <Database />, t: "Per-tenant isolation", d: "Cases and patients are scoped to their owner." },
];

export default function Platform() {
  useSeo("Platform", "Explainable oral cancer screening workflow: quality checks, OralNet-B4 risk tiers, Grad-CAM and LIME, case management, timelines, reports and an API.");
  return (
    <>
      <PageHero eyebrow="Platform" title={<>A screening workflow that <em className="italic text-brand">shows its work.</em></>}
        body="From capture to referral, OralAID gives clinicians a fast, explainable second opinion and a clean record of every decision.">
        <ButtonLink to="/register" size="lg">Start free <ArrowRight className="h-4 w-4" /></ButtonLink>
        <ButtonLink to="/contact" size="lg" variant="secondary">Book a demo</ButtonLink>
      </PageHero>

      {/* Workflow */}
      <Section>
        <SectionHeading eyebrow="Workflow" title="Six steps. One continuous record." align="center" />
        <div className="relative mt-14">
          <div aria-hidden className="absolute left-5 top-0 h-full w-px bg-line md:left-0 md:top-6 md:h-px md:w-full" />
          <ol className="relative grid gap-6 md:grid-cols-6 md:gap-4">
            {WORKFLOW.map((s, i) => (
              <li key={s.t}>
                <Reveal delay={i * 0.06} className="flex gap-4 md:flex-col md:gap-0">
                  <div className="relative z-10 grid h-10 w-10 shrink-0 place-items-center rounded-full border border-line bg-surface text-brand shadow-soft md:h-12 md:w-12 [&_svg]:h-[18px] [&_svg]:w-[18px]">{s.icon}</div>
                  <div className="md:mt-5">
                    <p className="font-mono text-[11px] text-muted">0{i + 1}</p>
                    <h3 className="text-sm font-semibold">{s.t}</h3>
                    <p className="mt-1 text-sm text-muted">{s.d}</p>
                  </div>
                </Reveal>
              </li>
            ))}
          </ol>
        </div>
      </Section>

      <Section className="space-y-28 border-t border-line pt-24 sm:space-y-36">
        <FeatureRow eyebrow="Analysis" title={<>Score, tier and <em className="italic text-brand">uncertainty</em>, not just a label.</>}
          body="OralNet-B4, an EfficientNet-B4 classifier, runs with multi-scale and horizontal-flip test-time augmentation. You get a suspicion probability, an uncertainty estimate from the spread across augmentations, a risk tier and a written recommendation."
          bullets={["Low below 0.30, Moderate from 0.30, High at 0.55 and above", "Case-level result across up to six images", "Per-image quality score with specific issues called out", "Recommendation text that always defers to the clinician"]}
          visual={<AnalysisMockup />} />

        <FeatureRow flip eyebrow="Explainability" title={<>Three views of <em className="italic text-brand">one decision.</em></>}
          body="Grad-CAM shows where the model looked. The Region Localizer turns that attention into a weakly supervised outline using GrabCut. LIME, on demand, scores superpixels by how much they push the call toward suspicious (red) or non-suspicious (green)."
          bullets={["Overlay comparison slider on every image", "Crop-to-region tool to remove fingers, retractors and background", "Explanation usefulness rating captured with feedback"]}
          visual={
            <div className="card p-5 shadow-lift">
              <div className="flex items-center gap-2 text-xs text-muted"><Crop className="h-4 w-4 text-brand" />Crop to region, then re-run</div>
              <div className="mt-4 grid grid-cols-[1fr_auto] items-center gap-5">
                <div className="relative overflow-hidden rounded-xl border border-line">
                  <div className="aspect-[4/3] w-full bg-[radial-gradient(circle_at_50%_55%,#cf7380,#a24456_55%,#4a1a24)]" role="img" aria-label="Illustrative intraoral photo with crop frame" />
                  <div className="absolute inset-[18%_20%_14%_28%] rounded-lg border-2 border-white shadow-[0_0_0_9999px_rgba(0,0,0,.45)]">
                    {["-left-1 -top-1", "-right-1 -top-1", "-bottom-1 -left-1", "-bottom-1 -right-1"].map((p) => <span key={p} className={cn("absolute h-2.5 w-2.5 rounded-sm bg-white", p)} />)}
                  </div>
                </div>
                <div className="hidden w-40 space-y-2 text-xs sm:block">
                  {[["Grad-CAM", "bg-high"], ["Region", "bg-brand"], ["LIME", "bg-moderate"]].map(([l, c]) => (
                    <div key={l} className="flex items-center gap-2 rounded-lg border border-line px-2.5 py-2"><span className={cn("h-2 w-2 rounded-full", c)} />{l}</div>
                  ))}
                </div>
              </div>
            </div>
          } />

        <FeatureRow eyebrow="Case management" title={<>Queues that match <em className="italic text-brand">how clinics work.</em></>}
          body="Every case is open, referred or closed. Filter by tier, status or patient, search instantly with the command palette, and export everything to CSV for audits or research."
          bullets={[<>Command palette: <Kbd>⌘</Kbd> <Kbd>K</Kbd> to jump anywhere</>, "Dashboard analytics: risk mix, trend, agreement rate, average quality", "CSV export of cases and feedback", "Anatomical site tagging with an interactive mouth map"]}
          visual={<CaseListMockup />} />

        <FeatureRow flip eyebrow="Patients" title={<>Follow a lesion <em className="italic text-brand">over time.</em></>}
          body="Link cases to a patient reference and OralAID builds a longitudinal timeline, highlighting when the suspicion score moves between visits."
          bullets={["Patient reference, age, sex and risk factors", "Score trend across visits with tier bands", "Printable report per case for referral letters"]}
          visual={<div className="grid gap-4 sm:grid-cols-2"><TimelineMockup className="sm:col-span-2" /><ReportMockup className="sm:col-span-2" /></div>} />

        <FeatureRow eyebrow="Anatomy" title={<>Where it is matters. <em className="italic text-brand">Record it.</em></>}
          body="Tap the site on a schematic mouth map when capturing. The dashboard reuses the same map as a heat map of where findings cluster across your practice."
          bullets={["Lip, palate, gingiva, buccal mucosa, tongue surfaces, floor of mouth, retromolar trigone", "Site-level analytics on the dashboard"]}
          visual={<div className="card mx-auto max-w-md p-6 shadow-lift"><div className="mb-3 flex items-center gap-2 text-xs text-muted"><MapPin className="h-4 w-4 text-brand" />Select anatomical site</div><MouthMap value="Tongue (lateral)" /></div>} />
      </Section>

      {/* Feedback loop */}
      <Section className="border-y border-line bg-surface/50">
        <div className="grid items-center gap-12 lg:grid-cols-2 lg:gap-20">
          <div>
            <SectionHeading eyebrow="Clinician feedback loop" title={<>Every review makes the <em className="italic text-brand">next model</em> better.</>}
              body="Clinicians record whether they agree, their clinical impression, the histopathology result when it arrives, and how useful the explanation was. With your consent, this ground truth feeds future retraining and evaluation." />
            <Reveal delay={0.1} className="mt-8 grid gap-3 sm:grid-cols-2">
              {[["Verdict", "Agree, disagree or unsure"], ["Impression", "Normal, benign, OPMD or malignant"], ["Histopathology", "The ground truth that matters"], ["Explanation rating", "Was the heatmap useful?"]].map(([t, d]) => (
                <div key={t} className="rounded-xl border border-line bg-surface p-4"><p className="text-sm font-semibold">{t}</p><p className="mt-1 text-xs text-muted">{d}</p></div>
              ))}
            </Reveal>
          </div>
          <Reveal delay={0.08}><FeedbackMockup className="mx-auto max-w-md" /></Reveal>
        </div>
      </Section>

      {/* Integrations */}
      <Section>
        <SectionHeading eyebrow="Integrations" title={<>Fits into <em className="italic text-brand">your stack.</em></>}
          body="A clean REST API and scoped API keys let your EHR, imaging archive or research pipeline submit cases and read results." />
        <div className="mt-12 grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
          {[
            { icon: <Braces />, t: "REST API", d: "JSON endpoints for cases, patients, feedback, dashboard and audit.", tag: "Available" },
            { icon: <KeyRound />, t: "API keys", d: "Create and revoke keys per integration. Send them in the X-API-Key header.", tag: "Available" },
            { icon: <FileSpreadsheet />, t: "CSV export", d: "Cases, tiers, sites and feedback in one file.", tag: "Available" },
            { icon: <FileText />, t: "Printable reports", d: "A clean one-page summary for referral letters and records.", tag: "Available" },
            { icon: <Database />, t: "HL7 FHIR", d: "DiagnosticReport and ImagingStudy resources for EHR integration.", tag: "On the roadmap" },
            { icon: <Timer />, t: "DICOM", d: "Accept DICOM visible-light studies and write back structured results.", tag: "On the roadmap" },
          ].map((x, i) => (
            <Reveal key={x.t} delay={i * 0.04} className="card flex flex-col p-6">
              <div className="flex items-start justify-between gap-3">
                <IconTile tone={x.tag === "Available" ? "brand" : "neutral"}>{x.icon}</IconTile>
                <Badge tone={x.tag === "Available" ? "low" : "neutral"}>{x.tag}</Badge>
              </div>
              <h3 className="mt-5 font-semibold">{x.t}</h3>
              <p className="mt-1.5 text-sm text-muted">{x.d}</p>
            </Reveal>
          ))}
        </div>
        <Reveal className="mt-8"><Link to="/docs" className="inline-flex items-center gap-1.5 text-sm font-medium text-brand hover:underline">Read the API docs <ArrowRight className="h-4 w-4" /></Link></Reveal>
      </Section>

      {/* Security */}
      <Section className="border-t border-line">
        <div className="flex flex-col justify-between gap-6 lg:flex-row lg:items-end">
          <SectionHeading eyebrow="Security" title={<>Built for <em className="italic text-brand">clinical data</em> from day one.</>} />
          <Reveal><ButtonLink to="/security" variant="secondary">Security overview <ArrowRight className="h-4 w-4" /></ButtonLink></Reveal>
        </div>
        <div className="mt-12 grid gap-px overflow-hidden rounded-2xl border border-line bg-line sm:grid-cols-2 lg:grid-cols-3">
          {SECURITY.map((s) => (
            <div key={s.t} className="flex gap-4 bg-surface p-6">
              <IconTile tone="neutral">{s.icon}</IconTile>
              <div><h3 className="text-sm font-semibold">{s.t}</h3><p className="mt-1 text-sm text-muted">{s.d}</p></div>
            </div>
          ))}
        </div>
        <p className="mt-6 flex items-center gap-2 text-xs text-muted"><Lock className="h-3.5 w-3.5" />Deploy in our cloud, your private VPC, or fully on-premise. <Link to="/deployment" className="text-brand hover:underline">Compare options</Link></p>
      </Section>

      <CTABand />
    </>
  );
}
