import { motion } from "framer-motion";
import { ArrowUpRight } from "lucide-react";
import { useSeo } from "@/lib/seo";
import { Reveal } from "@/components/Reveal";
import { CTABand, PageHero, Section, SectionHeading } from "@/components/site/marketing";
import { BURDEN, Counter, SURVIVAL } from "@/components/site/HomeSections";

// Aggregates only: computed from the de-identified project dataset. No individual records are shown.
const DATASET = {
  patients: 608,
  images: 3484,
  labelled: 518,
  suspicious: 249,
  nonSuspicious: 269,
  male: 77,
  medianAge: 47,
  hospital: 87,
  tobacco: 92,
};

const DIAGNOSES = [
  ["Oral cavity normal", 245],
  ["Homogenous leukoplakia", 114],
  ["Smokeless tobacco keratosis", 90],
  ["Squamous cell carcinoma", 26],
  ["Oral submucous fibrosis", 18],
  ["Other / benign", 24],
] as const;

const EXPOSURE = [
  ["Smokeless tobacco (current)", 49],
  ["Areca nut (current)", 30],
  ["Alcohol (current)", 26],
  ["Cigarette / bidi (current)", 17],
] as const;

const SOURCES = [
  ["World Cancer Research Fund: mouth and oral cancer statistics (GLOBOCAN 2022)", "https://www.wcrf.org/preventing-cancer/cancer-statistics/mouth-and-oral-cancer-statistics/"],
  ["Oral cancer burden and mortality in India: a GLOBOCAN 2022 analysis (ecancermedicalscience)", "https://ecancer.org/en/journal/article/2162-oral-cancer-burden-and-mortality-in-india-a-globocan-2022-analysis-with-regional-comparison"],
  ["NCI SEER Cancer Stat Facts: oral cavity and pharynx cancer", "https://seer.cancer.gov/statfacts/html/oralcav.html"],
  ["American Cancer Society: survival rates for oral cavity and oropharyngeal cancers", "https://www.cancer.org/cancer/oral-cavity-and-oropharyngeal-cancer/detection-diagnosis-staging/survival-rates.html"],
];

function Bar({ label, value, max = 100, suffix = "%", delay = 0 }: { label: string; value: number; max?: number; suffix?: string; delay?: number }) {
  return (
    <div>
      <div className="mb-1.5 flex justify-between gap-3 text-sm"><span className="text-ink/90">{label}</span><span className="font-semibold tabular-nums">{value}{suffix}</span></div>
      <div className="h-2 overflow-hidden rounded-full bg-surface-2">
        <motion.div initial={{ width: 0 }} whileInView={{ width: `${(value / max) * 100}%` }} viewport={{ once: true }}
          transition={{ duration: 1.3, delay, ease: [0.16, 1, 0.3, 1] }}
          className="h-full rounded-full bg-[linear-gradient(90deg,rgb(var(--glow-1)),rgb(var(--glow-3)),rgb(var(--glow-2)))]" />
      </div>
    </div>
  );
}

export default function CaseStudy() {
  useSeo("Case study: oral cancer in India and the OralAID dataset",
    "How many people oral cancer affects, why early detection matters, who is at risk, and what we learned building OralAID on specialist-labelled screening data.");
  return (
    <>
      <PageHero eyebrow="Case study" title={<>A visible cancer, <em>found too late.</em></>}
        body="Oral cancer can often be seen in the mouth before it becomes dangerous, yet most cases are still diagnosed after they have spread. This is the problem OralAID exists to help with, and what we learned building it." />

      <Section>
        <SectionHeading eyebrow="01 · The burden" title={<>How many people it <em>affects.</em></>} />
        <div className="mt-12 grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
          {BURDEN.map((b, i) => (
            <Reveal key={b.label} delay={i * 0.07}>
              <div className="card-stripe spotlight h-full p-6 pt-7">
                <p className="stat-num text-4xl sm:text-5xl"><Counter to={b.value} suffix={b.suffix} /></p>
                <p className="mt-3 text-sm">{b.label}</p>
                <p className="mt-4 text-[11px] uppercase tracking-wider text-muted">{b.source}</p>
              </div>
            </Reveal>
          ))}
        </div>
        <Reveal>
          <p className="mt-8 max-w-3xl text-muted">
            India carries the largest share of the world's oral cancer, driven above all by smokeless tobacco and areca nut.
            Men are affected about three times as often as women, and most patients are of working age, so every late
            diagnosis also hits a household's income.
          </p>
        </Reveal>
      </Section>

      <Section>
        <div className="grid gap-12 lg:grid-cols-2">
          <div>
            <SectionHeading eyebrow="02 · Why timing matters" title={<>Early stage changes <em>everything.</em></>}
              body="Five-year relative survival falls sharply once the cancer spreads. Yet only about a quarter of cases are found while still localised." />
          </div>
          <Reveal>
            <div className="card space-y-5 p-6 sm:p-8">
              {SURVIVAL.map((s, i) => <Bar key={s.stage} label={s.stage} value={s.value} delay={i * 0.12} />)}
              <p className="text-xs text-muted">US SEER, oral cavity and pharynx, diagnosed 2015 to 2021.</p>
            </div>
          </Reveal>
        </div>
      </Section>

      <Section>
        <SectionHeading eyebrow="03 · Our data" title={<>Built on real <em>screening data.</em></>}
          body="OralAID's model was trained and evaluated on intraoral photographs from a community oral cancer screening programme in India, each case reviewed by a specialist. The figures below are aggregates; no individual records are published." />
        <div className="mt-12 grid divide-y divide-line border-y border-line sm:grid-cols-2 sm:divide-y-0 lg:grid-cols-4 lg:divide-x">
          {[
            [<Counter to={DATASET.patients} />, "people screened"],
            [<Counter to={DATASET.images} />, "intraoral photographs"],
            [<Counter to={DATASET.labelled} />, `specialist-labelled cases (${DATASET.suspicious} suspicious, ${DATASET.nonSuspicious} not)`],
            [<Counter to={DATASET.tobacco} suffix="%" />, "used tobacco or areca nut, now or in the past"],
          ].map(([v, l], i) => (
            <Reveal key={i} delay={i * 0.07}>
              <div className="px-1 py-8 sm:px-6 lg:first:pl-0"><p className="stat-num text-5xl sm:text-6xl">{v}</p><p className="mt-4 max-w-[15rem] text-sm text-muted">{l}</p></div>
            </Reveal>
          ))}
        </div>
        <div className="mt-6 grid gap-6 lg:grid-cols-2">
          <Reveal>
            <div className="card spotlight h-full space-y-4 p-6 sm:p-8">
              <p className="eyebrow">Provisional diagnosis (people)</p>
              {DIAGNOSES.map(([l, v], i) => <Bar key={l} label={l} value={v} max={245} suffix="" delay={i * 0.06} />)}
            </div>
          </Reveal>
          <Reveal delay={0.08}>
            <div className="panel-outline h-full space-y-4 p-6 sm:p-8">
              <p className="eyebrow">Habits at the time of screening</p>
              {EXPOSURE.map(([l, v], i) => <Bar key={l} label={l} value={v} delay={i * 0.06} />)}
              <p className="pt-2 text-sm text-muted">
                {DATASET.male}% men · median age {DATASET.medianAge} · {DATASET.hospital}% screened in hospital, the rest in field and primary-care camps.
              </p>
            </div>
          </Reveal>
        </div>
      </Section>

      <Section>
        <SectionHeading eyebrow="04 · What we learned" title={<>Honest results, <em>including the misses.</em></>} />
        <ol className="mt-12 grid gap-0 md:grid-cols-3 md:gap-10">
          {[
            ["A first model that failed", "Our first classifier (MobileNetV2) scored below chance on held-out specialist labels (AUC 0.41). We retired it and say so publicly."],
            ["A model that works, with limits", "OralNet-B4 reaches 76.8% held-out validation accuracy and AUC 0.83 on internal evaluation, with 90% precision in the High-risk tier. External validation is still pending."],
            ["Explanations catch shortcuts", "Grad-CAM showed the model sometimes looks at mirrors, retractors or fingers. That is why every result shows its heatmap, and why the app has a crop tool."],
          ].map(([t, b], i) => (
            <Reveal key={t} delay={i * 0.08}>
              <li className="relative border-l border-line py-6 pl-8 md:border-l-0 md:border-t md:pl-0 md:pt-8">
                <span aria-hidden className="absolute -left-px top-6 h-10 w-px bg-[linear-gradient(rgb(var(--glow-1)),rgb(var(--glow-2)))] md:-top-px md:left-0 md:h-px md:w-16 md:bg-[linear-gradient(90deg,rgb(var(--glow-1)),rgb(var(--glow-2)))]" />
                <span className="font-mono text-xs text-brand">0{i + 1}</span>
                <h3 className="mt-2 font-display text-xl font-semibold">{t}</h3>
                <p className="mt-3 text-sm leading-relaxed text-muted">{b}</p>
              </li>
            </Reveal>
          ))}
        </ol>
      </Section>

      <Section>
        <Reveal>
          <div className="card p-6 sm:p-8">
            <p className="eyebrow mb-4">Sources</p>
            <ul className="space-y-2.5 text-sm">
              {SOURCES.map(([t, u]) => (
                <li key={u}><a href={u} target="_blank" rel="noreferrer" className="inline-flex items-start gap-1.5 text-muted transition hover:text-brand">{t}<ArrowUpRight className="mt-0.5 h-3.5 w-3.5 shrink-0" /></a></li>
              ))}
            </ul>
            <p className="mt-5 text-xs text-muted">Dataset figures are aggregates computed from the de-identified project data. Images and records are not distributed.</p>
          </div>
        </Reveal>
      </Section>
      <CTABand />
    </>
  );
}
