import { Fragment, useState, type ReactNode } from "react";
import { motion } from "framer-motion";
import { ArrowRight, Check, Minus } from "lucide-react";
import { useSeo } from "@/lib/seo";
import { cn } from "@/lib/utils";
import { Reveal } from "@/components/Reveal";
import { Badge, ButtonLink } from "@/components/ui";
import { CTABand, ClinicalNote, FAQ, PageHero, Section, SectionHeading } from "@/components/site/marketing";

type Billing = "monthly" | "annual";

interface Tier {
  id: string;
  name: string;
  blurb: string;
  monthly: number | null; // USD per month; null = custom
  unit?: string;
  suffix?: string;
  cta: { label: string; to: string };
  featured?: boolean;
  features: string[];
}

const TIERS: Tier[] = [
  {
    id: "starter", name: "Starter", blurb: "For trying OralAID on real cases.", monthly: 0,
    cta: { label: "Start free", to: "/register" },
    features: ["50 analyses per month", "1 user", "Grad-CAM heatmap and region outline", "Quality check on every image", "Community support"],
  },
  {
    id: "clinic", name: "Clinic", blurb: "For practices screening every day.", monthly: 49, unit: "per user / month",
    cta: { label: "Start free trial", to: "/register" }, featured: true,
    features: ["Unlimited analyses (fair use)", "LIME explanations on demand", "Patient timelines", "Printable reports and CSV export", "API access, 1,000 calls / month", "Email support"],
  },
  {
    id: "network", name: "Network", blurb: "For multi-site groups and networks.", monthly: 199, unit: "per month base", suffix: "+ seats",
    cta: { label: "Talk to sales", to: "/contact" },
    features: ["Everything in Clinic", "Multi-site management", "SSO (SAML / OIDC)", "Audit log export", "Higher API limits", "Priority support"],
  },
  {
    id: "enterprise", name: "Enterprise / Private", blurb: "For hospitals, governments and research.", monthly: null,
    cta: { label: "Contact us", to: "/contact" },
    features: ["Dedicated private VPC or on-premise", "Uptime SLA", "Custom validation on your data", "BAA / DPA available", "Security review support", "Named technical contact"],
  },
];

type Val = boolean | string;
const MATRIX: { group: string; rows: { label: string; v: [Val, Val, Val, Val] }[] }[] = [
  { group: "Screening", rows: [
    { label: "Analyses", v: ["50 / month", "Unlimited*", "Unlimited*", "Unlimited"] },
    { label: "Images per case", v: ["Up to 6", "Up to 6", "Up to 6", "Up to 6"] },
    { label: "Image quality check", v: [true, true, true, true] },
    { label: "Risk tier and recommendation", v: [true, true, true, true] },
  ] },
  { group: "Explainability", rows: [
    { label: "Grad-CAM heatmap", v: [true, true, true, true] },
    { label: "Region outline", v: [true, true, true, true] },
    { label: "LIME superpixels", v: [false, true, true, true] },
  ] },
  { group: "Workflow", rows: [
    { label: "Case management", v: [true, true, true, true] },
    { label: "Patient timelines", v: [false, true, true, true] },
    { label: "Printable reports", v: [false, true, true, true] },
    { label: "CSV export", v: [false, true, true, true] },
    { label: "Dashboard analytics", v: ["Basic", true, true, true] },
    { label: "Multi-site", v: [false, false, true, true] },
  ] },
  { group: "Platform", rows: [
    { label: "Users", v: ["1", "Per seat", "Per seat", "Custom"] },
    { label: "API access", v: [false, "1k calls / mo", "Higher limits", "Custom"] },
    { label: "SSO", v: [false, false, true, true] },
    { label: "Audit log", v: ["View", "View", "View + export", "View + export"] },
    { label: "Deployment", v: ["Cloud", "Cloud", "Cloud", "Cloud, VPC or on-prem"] },
    { label: "Custom validation study", v: [false, false, false, true] },
    { label: "BAA / DPA", v: [false, false, "DPA", true] },
  ] },
  { group: "Support", rows: [
    { label: "Support", v: ["Community", "Email", "Priority", "Dedicated + SLA"] },
  ] },
];

function price(t: Tier, billing: Billing) {
  if (t.monthly === null) return null;
  if (t.monthly === 0) return 0;
  return billing === "annual" ? (t.monthly * 10) / 12 : t.monthly;
}

function fmt(n: number) {
  return Number.isInteger(n) ? `$${n}` : `$${n.toFixed(2).replace(/\.00$/, "")}`;
}

function MatrixCell({ v }: { v: Val }) {
  if (v === true) return <Check className="mx-auto h-4 w-4 text-brand" aria-label="Included" />;
  if (v === false) return <Minus className="mx-auto h-4 w-4 text-muted/60" aria-label="Not included" />;
  return <span className="text-[13px]">{v}</span>;
}

const FAQS: { q: string; a: ReactNode }[] = [
  { q: "Are these prices final?", a: "Prices are indicative, in USD, and exclude applicable taxes. Regional pricing for low- and middle-income countries and public health programmes is available on request." },
  { q: "How does annual billing work?", a: "Pay yearly and get two months free: you are billed for 10 months and get 12. You can switch between monthly and annual at renewal." },
  { q: "What does fair use mean on unlimited analyses?", a: "Unlimited covers normal clinical use by the licensed user. We will reach out before taking any action if usage looks automated or shared; API traffic is metered separately." },
  { q: "How are Network seats priced?", a: "The Network plan has a monthly base that covers multi-site management, SSO and audit export, plus per-seat pricing that decreases with volume. We'll quote it with you." },
  { q: "Is OralAID approved for clinical diagnosis?", a: "No. OralAID is clinical decision support for screening and research use. It is not cleared as a medical device in any jurisdiction and does not replace clinical examination or biopsy." },
  { q: "Can I get a BAA or DPA?", a: "A data processing agreement is available on Network and Enterprise. A business associate agreement is available on Enterprise. OralAID is designed to support HIPAA, GDPR and DPDP programmes but is not yet independently certified." },
];

export default function Pricing() {
  useSeo("Pricing", "Simple, transparent pricing for OralAID: free Starter plan, Clinic, Network and Enterprise private deployment options.");
  const [billing, setBilling] = useState<Billing>("monthly");

  return (
    <>
      <PageHero eyebrow="Pricing" title={<>Simple pricing. <em className="italic text-brand">Start free.</em></>}
        body="Every plan includes explainable results, quality checks and the same production model. Upgrade when your practice needs more." />

      <Section className="pt-14 sm:pt-16">
        <Reveal className="flex justify-center">
          <div className="inline-flex items-center rounded-xl border border-line bg-surface p-1 text-sm" role="radiogroup" aria-label="Billing period">
            {(["monthly", "annual"] as const).map((b) => (
              <button key={b} role="radio" aria-checked={billing === b} onClick={() => setBilling(b)}
                className={cn("relative rounded-lg px-4 py-1.5 font-medium transition", billing === b ? "text-bg" : "text-muted hover:text-ink")}>
                {billing === b && <motion.span layoutId="billing-pill" className="absolute inset-0 rounded-lg bg-ink" transition={{ type: "spring", bounce: 0.15, duration: 0.5 }} />}
                <span className="relative flex items-center gap-2">
                  {b === "monthly" ? "Monthly" : "Annual"}
                  {b === "annual" && <span className={cn("rounded-full px-1.5 py-0.5 text-[10px] font-semibold", billing === b ? "bg-bg/20 text-bg" : "bg-accent/10 text-accent")}>2 months free</span>}
                </span>
              </button>
            ))}
          </div>
        </Reveal>

        <div className="mt-12 grid gap-4 md:grid-cols-2 xl:grid-cols-4">
          {TIERS.map((t, i) => {
            const p = price(t, billing);
            return (
              <Reveal key={t.id} delay={i * 0.06}
                className={cn("relative flex flex-col rounded-2xl border p-6 sm:p-7",
                  t.featured ? "glow-border spotlight border-transparent bg-[linear-gradient(180deg,rgb(var(--glow-3)/.12),rgb(var(--surface)))] shadow-[0_30px_80px_-30px_rgb(var(--glow-3)/.6)] lg:-translate-y-3" : "spotlight border-line bg-surface/60")}>
                {t.featured && <Badge tone="brand" className="absolute -top-3 left-6 bg-surface">Most popular</Badge>}
                <h2 className="text-lg font-semibold">{t.name}</h2>
                <p className="mt-1 text-sm text-muted">{t.blurb}</p>
                <div className="mt-6 flex min-h-[72px] flex-col justify-end">
                  {p === null ? (
                    <p className="display text-5xl">Custom</p>
                  ) : (
                    <div className="flex flex-wrap items-baseline gap-x-2">
                      <motion.span key={`${t.id}-${billing}`} initial={{ opacity: 0, y: 6 }} animate={{ opacity: 1, y: 0 }} className="display text-5xl tabular-nums">
                        {p === 0 ? "Free" : fmt(Math.round(p))}
                      </motion.span>
                      {t.unit && <span className="text-sm text-muted">{t.unit}</span>}
                      {t.suffix && <span className="text-sm font-medium text-ink">{t.suffix}</span>}
                    </div>
                  )}
                  <p className="mt-1 h-4 text-xs text-muted">
                    {p !== null && p > 0 && billing === "annual" && t.monthly !== null && `Billed ${fmt(t.monthly * 10)} yearly${t.unit?.startsWith("per user") ? " per user" : ""}`}
                    {p !== null && p > 0 && billing === "monthly" && "Billed monthly"}
                    {p === 0 && "No credit card required"}
                    {p === null && "Annual agreement"}
                  </p>
                </div>
                <ButtonLink to={t.cta.to} variant={t.featured ? "primary" : "secondary"} className="mt-6 w-full">{t.cta.label}</ButtonLink>
                <ul className="mt-7 space-y-2.5 border-t border-line pt-6">
                  {t.features.map((f) => (
                    <li key={f} className="flex gap-2.5 text-sm"><Check className="mt-0.5 h-4 w-4 shrink-0 text-brand" />{f}</li>
                  ))}
                </ul>
              </Reveal>
            );
          })}
        </div>
        <p className="mt-8 text-center text-xs text-muted">Indicative prices in USD, excluding taxes. For screening support and research use; not a diagnostic device.</p>
      </Section>

      <Section className="border-t border-line">
        <SectionHeading eyebrow="Compare plans" title="Every feature, side by side" />
        <Reveal className="mt-10 overflow-x-auto rounded-2xl border border-line bg-surface shadow-soft">
          <table className="w-full min-w-[720px] text-sm">
            <caption className="sr-only">Plan feature comparison</caption>
            <thead>
              <tr className="border-b border-line bg-surface">
                <th scope="col" className="w-[34%] p-4 text-left text-xs font-medium text-muted sm:px-6">Feature</th>
                {TIERS.map((t) => <th key={t.id} scope="col" className={cn("p-4 text-center font-semibold", t.featured && "text-brand")}>{t.name.split(" /")[0]}</th>)}
              </tr>
            </thead>
            <tbody>
              {MATRIX.map((g) => (
                <Fragment key={g.group}>
                  <tr className="bg-surface-2/60">
                    <th colSpan={5} scope="colgroup" className="px-4 py-2.5 text-left text-[11px] font-semibold uppercase tracking-[0.14em] text-muted sm:px-6">{g.group}</th>
                  </tr>
                  {g.rows.map((r) => (
                    <tr key={r.label} className="border-t border-line/70 transition hover:bg-surface-2/40">
                      <th scope="row" className="p-4 text-left font-normal text-ink/90 sm:px-6">{r.label}</th>
                      {r.v.map((v, i) => <td key={i} className={cn("p-4 text-center", TIERS[i].featured && "bg-brand/[0.03]")}><MatrixCell v={v} /></td>)}
                    </tr>
                  ))}
                </Fragment>
              ))}
            </tbody>
          </table>
        </Reveal>
        <p className="mt-4 text-xs text-muted">* Fair use applies. API calls are metered separately.</p>
      </Section>

      <Section className="border-t border-line">
        <div className="grid gap-12 lg:grid-cols-[0.8fr_1.2fr]">
          <div>
            <SectionHeading eyebrow="FAQ" title="Pricing questions" />
            <Reveal delay={0.1} className="mt-8 space-y-4">
              <ButtonLink to="/contact" variant="secondary">Ask us anything <ArrowRight className="h-4 w-4" /></ButtonLink>
              <ClinicalNote className="max-w-sm" />
            </Reveal>
          </div>
          <Reveal><FAQ items={FAQS} /></Reveal>
        </div>
      </Section>

      <CTABand />
    </>
  );
}
