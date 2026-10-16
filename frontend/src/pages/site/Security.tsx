import type { ReactNode } from "react";
import { Link } from "react-router-dom";
import {
  ArrowRight, Building2, Clock, Cloud, Eye, FileLock2, Fingerprint, HardDrive, History, KeyRound, Link2, Lock, LogOut, Mail, Scale,
  ShieldAlert, ShieldCheck, UserCheck, Users,
} from "lucide-react";
import { useSeo } from "@/lib/seo";
import { Reveal } from "@/components/Reveal";
import { Badge, ButtonLink } from "@/components/ui";
import { IconTile, PageHero, Section, SectionHeading } from "@/components/site/marketing";

const DATA: { icon: ReactNode; t: string; d: string }[] = [
  { icon: <Lock />, t: "Encryption in transit", d: "All traffic is served over TLS. Private deployments can add encryption at rest with your own keys." },
  { icon: <Link2 />, t: "Signed, expiring image URLs", d: "Images are never public. Every link is signed and expires after one hour." },
  { icon: <Fingerprint />, t: "bcrypt password hashing", d: "Passwords are hashed with bcrypt and never stored or logged in plain text." },
  { icon: <ShieldAlert />, t: "Account lockout and rate limiting", d: "Repeated failed sign-ins lock the account temporarily. Every API surface is rate limited." },
  { icon: <LogOut />, t: "Sign out everywhere", d: "Revoke every active session at once. Changing your password does the same." },
  { icon: <History />, t: "Audit log", d: "Case views, edits, exports, deletions and key changes are recorded with time and IP." },
  { icon: <Users />, t: "Per-tenant isolation", d: "Every case, patient and image is scoped to its owner and checked on each request." },
  { icon: <KeyRound />, t: "Scoped, revocable API keys", d: "Keys are stored hashed, shown once, and can be revoked instantly." },
  { icon: <Clock />, t: "Demo images deleted after 1 hour", d: "Images uploaded to the public demo are deleted automatically and never used for training." },
  { icon: <FileLock2 />, t: "No training without opt-in", d: "We do not use customer images or feedback to train models unless you explicitly opt in." },
];

const RESPONSIBLE: { icon: ReactNode; t: string; d: ReactNode }[] = [
  { icon: <Eye />, t: "Explainability on every result", d: "Grad-CAM and a region outline accompany every score, with LIME on demand, so clinicians can see what drove the call." },
  { icon: <UserCheck />, t: "Human in the loop", d: "OralAID recommends; clinicians decide. Results are framed as screening support, never as a diagnosis." },
  { icon: <ShieldCheck />, t: "Transparent model cards", d: <>Architecture, inputs, metrics, evaluation caveats and limitations are published on the <Link to="/models" className="text-brand underline">Models page</Link>.</> },
  { icon: <Scale />, t: "Retired models disclosed", d: "Models that failed validation stay listed as retired, with the reason, instead of quietly disappearing." },
];

export default function Security() {
  useSeo("Security and trust", "How OralAID protects clinical images: encryption, signed URLs, audit logs, tenant isolation, deployment options and responsible AI.");
  return (
    <>
      <PageHero eyebrow="Security and trust" title={<>Clinical data deserves <em className="italic text-brand">more than a checkbox.</em></>}
        body="Here is exactly how we handle images, accounts and models, including what we have not done yet." />

      <Section>
        <SectionHeading eyebrow="Data handling" title="How your data is protected" />
        <div className="mt-12 grid gap-px overflow-hidden rounded-2xl border border-line bg-line sm:grid-cols-2">
          {DATA.map((x) => (
            <div key={x.t} className="flex gap-4 bg-surface p-6">
              <IconTile tone="neutral">{x.icon}</IconTile>
              <div><h3 className="font-semibold">{x.t}</h3><p className="mt-1 text-sm leading-relaxed text-muted">{x.d}</p></div>
            </div>
          ))}
        </div>
      </Section>

      <Section className="border-y border-line bg-surface/50">
        <div className="flex flex-col justify-between gap-6 lg:flex-row lg:items-end">
          <SectionHeading eyebrow="Isolation" title={<>Choose how far your data <em className="italic text-brand">travels.</em></>} />
          <Reveal><ButtonLink to="/deployment" variant="secondary">Deployment options <ArrowRight className="h-4 w-4" /></ButtonLink></Reveal>
        </div>
        <div className="mt-12 grid gap-4 md:grid-cols-3">
          {[
            { icon: <Cloud />, t: "Cloud SaaS", d: "Logical per-tenant isolation in our managed environment.", to: "/deployment#cloud" },
            { icon: <Building2 />, t: "Private VPC", d: "Single-tenant infrastructure in your own cloud account and region.", to: "/deployment#vpc" },
            { icon: <HardDrive />, t: "On-premise", d: "Air-gap capable. Images and results never leave your network.", to: "/deployment#onprem" },
          ].map((o, i) => (
            <Reveal key={o.t} delay={i * 0.06}>
              <Link to={o.to} className="card-stripe spotlight flex h-full flex-col p-6 transition hover:-translate-y-0.5 hover:border-brand/30 hover:shadow-lift">
                <IconTile>{o.icon}</IconTile>
                <h3 className="mt-5 font-semibold">{o.t}</h3>
                <p className="mt-1.5 text-sm text-muted">{o.d}</p>
              </Link>
            </Reveal>
          ))}
        </div>
      </Section>

      <Section>
        <SectionHeading eyebrow="Responsible AI" title={<>Trust is earned by <em className="italic text-brand">showing the work.</em></>} />
        <div className="mt-12 grid gap-4 sm:grid-cols-2">
          {RESPONSIBLE.map((r, i) => (
            <Reveal key={r.t} delay={i * 0.05} className="card-stripe spotlight p-6">
              <IconTile>{r.icon}</IconTile>
              <h3 className="mt-5 font-semibold">{r.t}</h3>
              <p className="mt-1.5 text-sm leading-relaxed text-muted">{r.d}</p>
            </Reveal>
          ))}
        </div>
      </Section>

      <Section className="border-t border-line">
        <div className="grid gap-12 lg:grid-cols-2 lg:gap-20">
          <div>
            <SectionHeading eyebrow="Compliance posture" title="Stated honestly" />
            <Reveal delay={0.08} className="mt-8 space-y-4 text-[15px] leading-relaxed text-muted">
              <p>
                OralAID is <span className="font-medium text-ink">designed to support HIPAA, GDPR and India's DPDP Act programmes</span>: access controls,
                audit logging, data minimisation, deletion on request and deployment options that keep data in your jurisdiction.
              </p>
              <p>
                A data processing agreement (DPA) and business associate agreement (BAA) are available on the Enterprise plan.
              </p>
              <p>
                <span className="font-medium text-ink">We are not yet independently certified</span> (for example SOC 2 or ISO 27001), and OralAID is
                not cleared as a medical device in any jurisdiction. We will update this page as that changes.
              </p>
            </Reveal>
            <Reveal delay={0.12} className="mt-6 flex flex-wrap gap-2">
              <Badge tone="brand">HIPAA-ready controls</Badge><Badge tone="brand">GDPR</Badge><Badge tone="brand">DPDP</Badge>
              <Badge>SOC 2: not yet</Badge><Badge>ISO 27001: not yet</Badge><Badge>Medical device clearance: not yet</Badge>
            </Reveal>
          </div>
          <Reveal delay={0.05} className="card-stripe spotlight relative overflow-hidden p-7 sm:p-9">
            <div aria-hidden className="absolute -right-20 -top-20 h-56 w-56 rounded-full bg-brand/10 blur-2xl" />
            <IconTile tone="accent"><ShieldAlert /></IconTile>
            <h2 className="mt-6 text-2xl font-semibold tracking-tight">Responsible disclosure</h2>
            <p className="mt-3 text-sm leading-relaxed text-muted">
              Found a vulnerability? Please email us with steps to reproduce. We acknowledge reports within three working days, keep you updated,
              and will not pursue legal action against good-faith research that avoids patient data, service disruption and social engineering.
            </p>
            <a href="mailto:security@oralaid.ai" className="mt-6 inline-flex items-center gap-2 rounded-xl border border-line bg-surface px-4 py-2.5 font-mono text-sm transition hover:border-brand/40 hover:text-brand">
              <Mail className="h-4 w-4" />security@oralaid.ai
            </a>
          </Reveal>
        </div>
      </Section>
    </>
  );
}
