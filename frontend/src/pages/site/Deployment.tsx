import { useState, type ReactNode } from "react";
import { motion } from "framer-motion";
import { Building2, Check, Cloud, Cpu, Gauge, HardDrive, MemoryStick, Minus, Server, Zap } from "lucide-react";
import { useSeo } from "@/lib/seo";
import { cn } from "@/lib/utils";
import { Reveal } from "@/components/Reveal";
import { Badge } from "@/components/ui";
import { CheckList, IconTile, PageHero, Section, SectionHeading } from "@/components/site/marketing";
import { LeadForm } from "@/components/site/LeadForm";

type Option = "cloud" | "vpc" | "onprem";

const OPTIONS: { id: Option; icon: ReactNode; name: string; tagline: string; body: string; bullets: string[]; best: string }[] = [
  {
    id: "cloud", icon: <Cloud />, name: "Cloud SaaS", tagline: "Start in minutes",
    body: "Our managed, multi-tenant service. Sign up, upload, and screen. We run the infrastructure, patch it, and ship model updates.",
    bullets: ["No installation, works in any modern browser", "Encrypted in transit, signed expiring image URLs", "Per-tenant data isolation and audit log", "Automatic updates to models and app"],
    best: "Independent practices and pilots",
  },
  {
    id: "vpc", icon: <Building2 />, name: "Dedicated private VPC", tagline: "Your cloud, your region",
    body: "A single-tenant deployment inside your own AWS, GCP or Azure account, in the region you choose. Data never touches shared infrastructure.",
    bullets: ["Single tenant, in your account and region", "Your network policies, keys and logging", "SSO with your identity provider", "Scheduled, approved updates"],
    best: "Hospital groups and dental networks",
  },
  {
    id: "onprem", icon: <HardDrive />, name: "On-premise / local", tagline: "Nothing leaves the clinic",
    body: "Docker images that run on a clinic server or workstation. CPU only is fine. Works fully air-gapped, with offline update bundles.",
    bullets: ["Runs on CPU; GPU optional", "Fully air-gapped operation", "No image or result leaves your network", "Offline update bundles you install"],
    best: "Strict data residency and low connectivity",
  },
];

const ROWS: { label: string; values: Record<Option, ReactNode> }[] = [
  { label: "Data residency", values: { cloud: "OralAID-managed region", vpc: "Your chosen cloud region", onprem: "Your premises" } },
  { label: "Isolation", values: { cloud: "Logical, per tenant", vpc: "Single-tenant infrastructure", onprem: "Physical, air-gap capable" } },
  { label: "Updates", values: { cloud: "Continuous, automatic", vpc: "Scheduled, you approve", onprem: "Offline bundles, you install" } },
  { label: "SSO (SAML / OIDC)", values: { cloud: "Network plan and above", vpc: true, onprem: true } },
  { label: "Uptime", values: { cloud: "99.9% target", vpc: "SLA available", onprem: "Depends on your hardware" } },
  { label: "Who manages it", values: { cloud: "OralAID", vpc: "OralAID, in your account", onprem: "Your IT, with our support" } },
  { label: "Time to deploy", values: { cloud: "Minutes", vpc: "Typically weeks", onprem: "Days to weeks" } },
];

/* ------------------------------------------------------- Architecture SVG */

const NODES: Record<Option, { label: string; sub: string }[]> = {
  cloud: [
    { label: "Browser", sub: "Clinician" }, { label: "CDN + TLS", sub: "Edge" }, { label: "API", sub: "Auth, cases, audit" },
    { label: "Inference", sub: "OralNet-B4 workers" }, { label: "Storage", sub: "DB + encrypted images" },
  ],
  vpc: [
    { label: "Browser", sub: "Your staff, via SSO" }, { label: "Load balancer", sub: "Your VPC edge" }, { label: "API", sub: "Single tenant" },
    { label: "Inference", sub: "Autoscaled workers" }, { label: "Storage", sub: "Your DB + buckets" },
  ],
  onprem: [
    { label: "Browser", sub: "Clinic PCs" }, { label: "Reverse proxy", sub: "Local TLS" }, { label: "API", sub: "Docker container" },
    { label: "Inference", sub: "CPU workers" }, { label: "Storage", sub: "Local disk + DB" },
  ],
};

const BOUNDARY: Record<Option, { from: number; label: string }> = {
  cloud: { from: 1, label: "OralAID managed cloud" },
  vpc: { from: 1, label: "Your AWS / GCP / Azure account · single tenant" },
  onprem: { from: 0, label: "Clinic network · air-gap capable" },
};

function ArchDiagram({ option }: { option: Option }) {
  const nodes = NODES[option];
  const b = BOUNDARY[option];
  // Horizontal layout
  const W = 152, NW = 124, NH = 66, Y = 96, X0 = 16;
  // Vertical layout
  const VY = 92, VH = 60, VW = 220, VX = 70, VY0 = 54;
  return (
    <div className="card overflow-hidden p-4 sm:p-6">
      <svg viewBox="0 0 776 220" className="hidden h-auto w-full md:block" role="img" aria-label={`Architecture: ${nodes.map((n) => n.label).join(" to ")}. ${b.label}.`}>
        <rect x={X0 + b.from * W - 10} y={46} width={(5 - b.from) * W - 8} height={150} rx={18} className="fill-brand/[0.04] stroke-brand/40" strokeDasharray="5 5" />
        <text x={X0 + b.from * W + 4} y={68} className="fill-brand text-[11px] font-medium">{b.label}</text>
        {nodes.slice(0, -1).map((_, i) => {
          const x1 = X0 + i * W + NW, x2 = X0 + (i + 1) * W;
          const d = `M ${x1} ${Y + NH / 2} L ${x2} ${Y + NH / 2}`;
          return (
            <g key={i}>
              <path d={d} className="stroke-line" strokeWidth={1.5} />
              <circle r={3} className="fill-brand"><animateMotion dur="1.8s" repeatCount="indefinite" begin={`${i * 0.3}s`} path={d} /></circle>
            </g>
          );
        })}
        {nodes.map((n, i) => (
          <g key={n.label} transform={`translate(${X0 + i * W} ${Y})`}>
            <rect width={NW} height={NH} rx={12} className={cn("stroke-line", i === 3 ? "fill-brand/10" : "fill-surface")} strokeWidth={1} />
            <text x={NW / 2} y={28} textAnchor="middle" className="fill-ink text-[13px] font-semibold">{n.label}</text>
            <text x={NW / 2} y={46} textAnchor="middle" className="fill-muted text-[10.5px]">{n.sub}</text>
          </g>
        ))}
        {option === "vpc" && <text x={X0 + 2 * W + NW / 2} y={30} textAnchor="middle" className="fill-muted text-[10.5px]">Signed images pulled from OralAID registry on approval</text>}
        {option === "onprem" && <text x={X0 + 2 * W + NW / 2} y={30} textAnchor="middle" className="fill-muted text-[10.5px]">No outbound connection required</text>}
        {option === "cloud" && <text x={X0 + 2 * W + NW / 2} y={30} textAnchor="middle" className="fill-muted text-[10.5px]">Rate limited, audited, encrypted in transit</text>}
      </svg>

      <svg viewBox="0 0 360 506" className="h-auto w-full md:hidden" role="img" aria-label={`Architecture: ${nodes.map((n) => n.label).join(" to ")}. ${b.label}.`}>
        <rect x={VX - 22} y={VY0 + b.from * VY - 34} width={VW + 44} height={(5 - b.from) * VY + 12} rx={18} className="fill-brand/[0.04] stroke-brand/40" strokeDasharray="5 5" />
        <text x={VX - 8} y={VY0 + b.from * VY - 14} className="fill-brand text-[11px] font-medium">{b.label.split(" · ")[0]}</text>
        {nodes.slice(0, -1).map((_, i) => {
          const d = `M ${VX + VW / 2} ${VY0 + i * VY + VH} L ${VX + VW / 2} ${VY0 + (i + 1) * VY}`;
          return (
            <g key={i}>
              <path d={d} className="stroke-line" strokeWidth={1.5} />
              <circle r={3} className="fill-brand"><animateMotion dur="1.4s" repeatCount="indefinite" begin={`${i * 0.25}s`} path={d} /></circle>
            </g>
          );
        })}
        {nodes.map((n, i) => (
          <g key={n.label} transform={`translate(${VX} ${VY0 + i * VY})`}>
            <rect width={VW} height={VH} rx={12} className={cn("stroke-line", i === 3 ? "fill-brand/10" : "fill-surface")} strokeWidth={1} />
            <text x={VW / 2} y={26} textAnchor="middle" className="fill-ink text-[14px] font-semibold">{n.label}</text>
            <text x={VW / 2} y={44} textAnchor="middle" className="fill-muted text-[11px]">{n.sub}</text>
          </g>
        ))}
      </svg>
    </div>
  );
}

function Architecture() {
  const [opt, setOpt] = useState<Option>("cloud");
  return (
    <Section className="border-y border-line bg-surface/50">
      <SectionHeading eyebrow="Architecture" title={<>Same pipeline. <em className="italic text-brand">Different walls.</em></>}
        body="Every edition runs the same API, inference workers and storage layer. What changes is where it runs and who holds the keys." />
      <Reveal className="mt-10">
        <div className="mb-5 inline-flex rounded-xl border border-line bg-surface p-1 text-sm" role="tablist" aria-label="Deployment variant">
          {OPTIONS.map((o) => (
            <button key={o.id} role="tab" aria-selected={opt === o.id} onClick={() => setOpt(o.id)}
              className={cn("relative rounded-lg px-3 py-1.5 font-medium transition sm:px-4", opt === o.id ? "text-bg" : "text-muted hover:text-ink")}>
              {opt === o.id && <motion.span layoutId="arch-pill" className="absolute inset-0 rounded-lg bg-ink" transition={{ type: "spring", bounce: 0.15, duration: 0.5 }} />}
              <span className="relative">{o.id === "cloud" ? "Cloud" : o.id === "vpc" ? "VPC" : "On-prem"}</span>
            </button>
          ))}
        </div>
        <ArchDiagram option={opt} />
      </Reveal>
    </Section>
  );
}

function Cell({ v }: { v: ReactNode }) {
  if (v === true) return <Check className="h-4 w-4 text-brand" aria-label="Included" />;
  if (v === false) return <Minus className="h-4 w-4 text-muted" aria-label="Not included" />;
  return <>{v}</>;
}

export default function Deployment() {
  useSeo("Deployment", "Deploy OralAID as cloud SaaS, in a dedicated private VPC on AWS, GCP or Azure, or fully on-premise and air-gapped on CPU.");
  return (
    <>
      <PageHero eyebrow="Deployment" title={<>Runs where your <em className="italic text-brand">data lives.</em></>}
        body="Cloud for speed, a private VPC for control, or on-premise for complete data sovereignty. Same models, same explanations, same workflow." />

      <Section>
        <div className="grid gap-5 lg:grid-cols-3">
          {OPTIONS.map((o, i) => (
            <Reveal key={o.id} delay={i * 0.06} className="h-full">
              <article id={o.id} className="card flex h-full scroll-mt-24 flex-col p-7 transition hover:shadow-lift">
                <div className="flex items-start justify-between gap-3">
                  <IconTile>{o.icon}</IconTile>
                  <Badge tone="brand">{o.tagline}</Badge>
                </div>
                <h2 className="mt-6 text-2xl font-semibold tracking-tight">{o.name}</h2>
                <p className="mt-3 text-sm leading-relaxed text-muted">{o.body}</p>
                <CheckList items={o.bullets} className="mt-6" />
                <p className="mt-auto pt-8 text-xs text-muted"><span className="font-medium text-ink">Best for:</span> {o.best}</p>
              </article>
            </Reveal>
          ))}
        </div>
      </Section>

      <Section className="pt-0 sm:pt-0">
        <SectionHeading eyebrow="Compare" title="Side by side" />
        <Reveal className="mt-10 overflow-x-auto rounded-2xl border border-line bg-surface shadow-soft">
          <table className="w-full min-w-[640px] text-left text-sm">
            <caption className="sr-only">Deployment option comparison</caption>
            <thead>
              <tr className="border-b border-line">
                <th scope="col" className="p-4 text-xs font-medium text-muted sm:p-5">&nbsp;</th>
                {OPTIONS.map((o) => <th key={o.id} scope="col" className="p-4 font-semibold sm:p-5">{o.name}</th>)}
              </tr>
            </thead>
            <tbody className="divide-y divide-line">
              {ROWS.map((r) => (
                <tr key={r.label} className="transition hover:bg-surface-2/50">
                  <th scope="row" className="p-4 font-medium text-muted sm:p-5">{r.label}</th>
                  {OPTIONS.map((o) => <td key={o.id} className="p-4 sm:p-5"><Cell v={r.values[o.id]} /></td>)}
                </tr>
              ))}
            </tbody>
          </table>
        </Reveal>
      </Section>

      <Architecture />

      <Section>
        <div className="grid gap-12 lg:grid-cols-[0.9fr_1.1fr] lg:gap-20">
          <div>
            <SectionHeading eyebrow="On-premise hardware" title={<>A GPU is <em className="italic text-brand">optional.</em></>}
              body="OralNet-B4 runs comfortably on modern CPUs at clinic volumes. Add a GPU only if you need higher throughput or frequent LIME explanations." />
            <Reveal delay={0.1}>
              <CheckList className="mt-8" items={[
                "Linux x86_64 with Docker and Docker Compose (Windows via Docker Desktop)",
                "Roughly 2 to 4 GB of RAM per inference worker",
                "Optional NVIDIA GPU with CUDA for higher throughput",
                "Works with no internet connection; updates ship as signed offline bundles",
              ]} />
            </Reveal>
          </div>
          <div className="grid gap-4 sm:grid-cols-3 lg:grid-cols-1">
            {[
              { icon: <Cpu />, t: "Minimum", d: "Single clinic", specs: [["CPU", "4 cores"], ["RAM", "8 GB"], ["Disk", "40 GB SSD"], ["Workers", "1"]] },
              { icon: <Gauge />, t: "Recommended", d: "Busy clinic or small network", specs: [["CPU", "8+ cores"], ["RAM", "16 GB"], ["Disk", "100 GB SSD"], ["Workers", "2 to 4"]] },
              { icon: <Zap />, t: "Accelerated", d: "High volume or research", specs: [["CPU", "8+ cores"], ["RAM", "32 GB"], ["GPU", "NVIDIA, 8 GB+ VRAM"], ["Workers", "4+"]] },
            ].map((h, i) => (
              <Reveal key={h.t} delay={i * 0.06} className="card p-5">
                <div className="flex items-center gap-3"><IconTile tone="neutral">{h.icon}</IconTile><div><h3 className="font-semibold">{h.t}</h3><p className="text-xs text-muted">{h.d}</p></div></div>
                <dl className="mt-4 grid grid-cols-2 gap-x-4 gap-y-2 text-sm lg:grid-cols-4">
                  {h.specs.map(([k, v]) => <div key={k}><dt className="text-[11px] uppercase tracking-wider text-muted">{k}</dt><dd className="font-medium">{v}</dd></div>)}
                </dl>
              </Reveal>
            ))}
            <p className="flex items-center gap-2 text-xs text-muted sm:col-span-3 lg:col-span-1"><MemoryStick className="h-3.5 w-3.5" />Indicative sizing. We size every private deployment with you.</p>
          </div>
        </div>
      </Section>

      <Section id="talk" className="border-t border-line">
        <div className="grid gap-12 lg:grid-cols-[0.9fr_1.1fr] lg:gap-20">
          <div>
            <SectionHeading eyebrow="Talk to us" title={<>Plan your <em className="italic text-brand">deployment.</em></>}
              body="Tell us about your environment and data requirements. An engineer, not a sales script, will get back to you." />
            <Reveal delay={0.1} className="mt-8 flex items-start gap-3 rounded-2xl border border-line bg-surface-2/50 p-5 text-sm text-muted">
              <Server className="mt-0.5 h-4 w-4 shrink-0 text-brand" />
              Private VPC and on-premise deployments are part of our Enterprise plan and include a validation run on a sample of your own images.
            </Reveal>
          </div>
          <Reveal delay={0.05} className="card p-6 sm:p-8">
            <LeadForm kind="deployment"
              select={{ label: "Preferred option", name: "preferred_option", options: [
                { value: "cloud", label: "Cloud SaaS" }, { value: "vpc", label: "Dedicated private VPC" },
                { value: "onprem", label: "On-premise / local" }, { value: "unsure", label: "Not sure yet" },
              ] }}
              messageLabel="Tell us about your environment" messagePlaceholder="Sites, expected volume, cloud provider, data residency requirements..."
              submitLabel="Send request" successTitle="Request received." successBody="We'll reply within two working days to set up a call." />
          </Reveal>
        </div>
      </Section>
    </>
  );
}
