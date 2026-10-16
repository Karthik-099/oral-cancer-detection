import { Eye, Lock, Microscope, Stethoscope, Wifi } from "lucide-react";
import { useSeo } from "@/lib/seo";
import { Reveal } from "@/components/Reveal";
import { CTABand, IconTile, PageHero, Section, SectionHeading } from "@/components/site/marketing";

const PRINCIPLES = [
  { icon: <Eye />, t: "Transparent", d: "Every result explains itself. Every model has a public card, including the ones that failed." },
  { icon: <Stethoscope />, t: "Clinician-first", d: "We build tools that support clinical judgement, never replace it. The clinician always makes the call." },
  { icon: <Lock />, t: "Private by design", d: "Minimal data, signed links, tenant isolation, and deployments that keep images where they belong." },
  { icon: <Wifi />, t: "Built for low-resource settings", d: "Runs on CPU, works offline on-premise, and is priced for the clinics that need it most." },
];

const WHY = [
  { n: "01", t: "Late diagnosis is common", d: "Oral cancer is often found at an advanced stage, when treatment is harder and outcomes are worse. Early lesions can be subtle and easy to miss." },
  { n: "02", t: "The dental chair is the right place", d: "Dentists and oral health workers see the mouth more often than anyone. Screening there, routinely, is the most practical way to look earlier." },
  { n: "03", t: "Explainability builds trust", d: "A black-box score is hard to act on. Showing what the model saw lets clinicians check it, challenge it, and decide with confidence." },
];

export default function About() {
  useSeo("About", "OralAID's mission is to make expert-level oral cancer screening available in every chair, with AI that explains itself.");
  return (
    <>
      <PageHero eyebrow="About OralAID" title={<>Make expert-level oral cancer screening available in <em className="italic text-brand">every chair.</em></>}
        body="We build explainable AI that helps clinicians decide who needs a specialist, sooner, wherever they practise." />

      <Section>
        <div className="grid gap-12 lg:grid-cols-2 lg:gap-20">
          <Reveal>
            <p className="eyebrow text-brand">Mission</p>
            <p className="display mt-5 text-3xl sm:text-4xl">
              Give every clinician a fast, explainable second opinion on oral lesions, so fewer cancers are found late.
            </p>
          </Reveal>
          <Reveal delay={0.08}>
            <p className="eyebrow text-brand">Vision</p>
            <p className="mt-5 text-lg leading-relaxed text-muted">
              Screening that happens at every dental visit, in every setting, from city hospitals to rural clinics without reliable internet.
              AI that clinicians trust because it shows its reasoning, admits its limits, and improves with their feedback.
            </p>
          </Reveal>
        </div>
      </Section>

      <Section className="border-y border-line bg-surface/50">
        <SectionHeading eyebrow="Why we exist" title={<>Three reasons, <em className="italic text-brand">one product.</em></>} />
        <div className="mt-12 grid gap-4 md:grid-cols-3">
          {WHY.map((w, i) => (
            <Reveal key={w.n} delay={i * 0.06} className="panel-outline p-7">
              <p className="font-mono text-xs text-brand">{w.n}</p>
              <h3 className="mt-4 text-lg font-semibold">{w.t}</h3>
              <p className="mt-2 text-sm leading-relaxed text-muted">{w.d}</p>
            </Reveal>
          ))}
        </div>
      </Section>

      <Section>
        <SectionHeading eyebrow="Principles" title="How we build" />
        <div className="mt-12 grid gap-px overflow-hidden rounded-2xl border border-line bg-line sm:grid-cols-2">
          {PRINCIPLES.map((p) => (
            <div key={p.t} className="bg-surface p-7 sm:p-9">
              <IconTile>{p.icon}</IconTile>
              <h3 className="display mt-6 text-3xl">{p.t}</h3>
              <p className="mt-2 text-sm leading-relaxed text-muted">{p.d}</p>
            </div>
          ))}
        </div>
      </Section>

      <Section className="border-t border-line">
        <Reveal className="mx-auto max-w-3xl text-center">
          <Microscope className="mx-auto h-6 w-6 text-brand" />
          <p className="display mt-6 text-3xl sm:text-4xl">
            "A tool that helps someone look closer, sooner, and refer with confidence. That is the whole job."
          </p>
          <p className="mt-5 text-sm text-muted">What we hold ourselves to. OralAID is screening support, not a diagnosis.</p>
        </Reveal>
      </Section>

      <CTABand title={<>Help us bring screening <em className="italic">to every chair.</em></>}
        body="Clinicians, researchers, public health teams and hardware makers: we would love to hear from you." />
    </>
  );
}
