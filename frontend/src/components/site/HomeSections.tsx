import { lazy, Suspense, useEffect, useRef, useState } from "react";
import { Link } from "react-router-dom";
import { animate, motion, useInView } from "framer-motion";
import { ArrowRight, Camera, Cpu, MonitorSmartphone, Radio, Stethoscope, WifiOff } from "lucide-react";
import {
  siCaddy, siDocker, siFastapi, siGithubactions, siLetsencrypt, siNetlify, siNumpy, siOpencv, siOracle, siPostgresql,
  siPython, siPytorch, siReact, siScikitlearn, siSqlalchemy, siTailwindcss, siThreedotjs, siTypescript, siVite,
} from "simple-icons";
import { Reveal } from "@/components/Reveal";
import { Section, SectionHeading } from "@/components/site/marketing";
import { MouthMap, SITES, type Site } from "@/components/MouthMap";
import { Spinner } from "@/components/ui";

const loadMouth3D = () => import("@/components/three/Mouth3D");
const Mouth3D = lazy(loadMouth3D);

/** Fetch the 3D chunk while the browser is idle, so it's ready before the user scrolls to it. */
export function usePrefetch3D() {
  useEffect(() => {
    const w = window as Window & { requestIdleCallback?: (cb: () => void, o?: { timeout: number }) => number };
    if (w.requestIdleCallback) w.requestIdleCallback(() => void loadMouth3D(), { timeout: 3000 });
    else setTimeout(() => void loadMouth3D(), 1500);
  }, []);
}

/* ------------------------------------------------------------- Counters */

export function Counter({ to, decimals = 0, suffix = "", prefix = "" }: { to: number; decimals?: number; suffix?: string; prefix?: string }) {
  const ref = useRef<HTMLSpanElement>(null);
  const inView = useInView(ref, { once: true, amount: 0.6 });
  const [v, setV] = useState(0);
  useEffect(() => {
    if (!inView) return;
    const c = animate(0, to, { duration: 1.8, ease: [0.16, 1, 0.3, 1], onUpdate: setV });
    return () => c.stop();
  }, [inView, to]);
  return <span ref={ref} className="tabular-nums">{prefix}{v.toLocaleString("en-US", { minimumFractionDigits: decimals, maximumFractionDigits: decimals })}{suffix}</span>;
}

/* ------------------------------------------------------------ Case study */

export const BURDEN = [
  { value: 389846, label: "new mouth and oral cancer cases worldwide in 2022", source: "WCRF / GLOBOCAN 2022" },
  { value: 143759, label: "new cases in India in 2022, about 1 in 3 worldwide", source: "GLOBOCAN 2022 analysis, ecancermedicalscience" },
  { value: 79979, label: "deaths from oral cancer in India in 2022", source: "GLOBOCAN 2022 analysis, ecancermedicalscience" },
  { value: 75, suffix: "%", label: "of cases in India are in men", source: "GLOBOCAN 2022 analysis, ecancermedicalscience" },
];

export const SURVIVAL = [
  { stage: "Found early (localised)", value: 88.4 },
  { stage: "Spread to nearby tissue", value: 69.4 },
  { stage: "Spread to distant organs", value: 36.9 },
];

export function CaseStudy() {
  return (
    <Section id="case-study" className="relative">
      <SectionHeading eyebrow="Case study" title={<>Why early screening <em>matters.</em></>}
        body="Oral cancer is one of the most common cancers in India, and most of it is linked to tobacco, areca nut and alcohol. It is also one of the most visible: many lesions can be seen in the mouth long before they become dangerous." />

      <div className="mt-14 grid grid-cols-1 divide-y divide-line border-y border-line sm:grid-cols-2 sm:divide-y-0 lg:grid-cols-4 lg:divide-x">
        {BURDEN.map((b, i) => (
          <Reveal key={b.label} delay={i * 0.07} className="px-1 py-8 sm:px-6 lg:first:pl-0">
            <p className="stat-num text-5xl sm:text-6xl"><Counter to={b.value} suffix={b.suffix} /></p>
            <p className="mt-4 max-w-[16rem] text-sm text-ink/85">{b.label}</p>
            <p className="mt-3 font-mono text-[10px] uppercase tracking-wider text-muted">{b.source}</p>
          </Reveal>
        ))}
      </div>

      <div className="mt-6 grid gap-6 lg:grid-cols-[1.1fr_1fr]">
        <Reveal>
          <div className="card spotlight h-full p-6 sm:p-8">
            <p className="eyebrow mb-1">5-year relative survival, by stage at diagnosis</p>
            <p className="mb-6 text-xs text-muted">US SEER, oral cavity and pharynx, 2015 to 2021</p>
            <div className="space-y-5">
              {SURVIVAL.map((s, i) => (
                <div key={s.stage}>
                  <div className="mb-1.5 flex justify-between text-sm"><span>{s.stage}</span><span className="font-semibold tabular-nums">{s.value}%</span></div>
                  <div className="h-2.5 overflow-hidden rounded-full bg-surface-2">
                    <motion.div initial={{ width: 0 }} whileInView={{ width: `${s.value}%` }} viewport={{ once: true }}
                      transition={{ duration: 1.4, delay: i * 0.15, ease: [0.16, 1, 0.3, 1] }}
                      className="h-full rounded-full bg-[linear-gradient(90deg,rgb(var(--glow-1)),rgb(var(--glow-3)),rgb(var(--glow-2)))]" />
                  </div>
                </div>
              ))}
            </div>
            <p className="mt-6 text-sm text-muted">Only about a quarter of cases are found at the early stage. Catching more lesions at a routine dental visit is the gap OralAID is built for.</p>
          </div>
        </Reveal>
        <Reveal delay={0.1}>
          <div className="panel-outline h-full p-6 sm:p-8">
            <p className="eyebrow mb-4">Who is at risk</p>
            <ul className="space-y-3 text-sm">
              {[
                ["Tobacco", "smoked (cigarettes, bidi) and smokeless (gutka, khaini)"],
                ["Areca / betel nut", "including paan; strongly linked to submucous fibrosis"],
                ["Alcohol", "risk multiplies when combined with tobacco"],
                ["Age and sex", "most cases are in adults over 40, and three in four in India are men"],
                ["HPV", "linked to some cancers at the back of the mouth and throat"],
              ].map(([k, v]) => (
                <li key={k} className="flex gap-3"><span className="mt-1.5 h-1.5 w-1.5 shrink-0 rounded-full bg-brand" /><span><span className="font-semibold">{k}:</span> <span className="text-muted">{v}</span></span></li>
              ))}
            </ul>
            <Link to="/case-study" className="mt-7 inline-flex items-center gap-1.5 text-sm font-medium text-brand hover:underline">
              Read the full case study and our dataset <ArrowRight className="h-4 w-4" />
            </Link>
          </div>
        </Reveal>
      </div>
    </Section>
  );
}

/* ---------------------------------------------------------- 3D site picker */

const SITE_NOTES: Partial<Record<Site, string>> = {
  "Buccal mucosa": "Inner cheek. The most common site in India, closely tied to chewing tobacco and areca nut.",
  "Tongue (lateral)": "Sides of the tongue. A high-risk site where red or white patches deserve attention.",
  "Floor of mouth": "Under the tongue. Lesions here can be missed without careful retraction.",
  "Gingiva": "Gums. Persistent lesions that don't heal after dental treatment should be reviewed.",
  "Lip": "Most often affected by sun exposure and smoking.",
  "Hard palate": "Roof of the mouth. Look for ulcers or patches that persist beyond two weeks.",
  "Soft palate": "Back of the roof of the mouth, near the throat.",
  "Retromolar trigone": "Behind the last molars, a hard-to-see site that is easy to overlook.",
  "Tongue (dorsal)": "Top of the tongue. Lesions here are less common but still examined.",
  "Tongue (ventral)": "Underside of the tongue, a high-risk site that needs the tongue lifted to see.",
};

export function Site3DShowcase() {
  const [site, setSite] = useState<Site | null>("Buccal mucosa");
  const ref = useRef<HTMLDivElement>(null);
  const near = useInView(ref, { once: true, margin: "900px" });
  usePrefetch3D();
  return (
    <Section id="anatomy">
      <div className="grid items-center gap-10 lg:grid-cols-[1fr_1.1fr]">
        <div>
          <SectionHeading eyebrow="3D anatomy" title={<>Tag every photo to the <em>right site.</em></>}
            body="Pick the anatomical site on a 3D model of the mouth. Site tags power the dashboard heat map and help track a lesion across visits." />
          <Reveal delay={0.1}>
            <div className="mt-8 flex flex-wrap gap-2">
              {SITES.filter((s) => s !== "Other").map((s) => (
                <button key={s} onClick={() => setSite(s)}
                  className={`rounded-full border px-3 py-1.5 text-xs transition ${site === s ? "border-brand bg-brand/15 text-ink" : "border-line text-muted hover:text-ink"}`}>
                  {s}
                </button>
              ))}
            </div>
            {site && <p className="mt-5 min-h-[3rem] max-w-md text-sm text-muted"><span className="font-semibold text-ink">{site}.</span> {SITE_NOTES[site]}</p>}
          </Reveal>
        </div>
        <Reveal delay={0.05}>
          <div ref={ref} className="card glow-border relative aspect-[4/3.4] overflow-hidden">
            <div aria-hidden className="absolute inset-0 bg-[radial-gradient(circle_at_50%_45%,rgb(var(--glow-3)/.18),transparent_65%)]" />
            {near ? (
              <Suspense fallback={<div className="grid h-full place-items-center"><Spinner /></div>}>
                <Mouth3D value={site} onChange={setSite} className="h-full w-full" />
              </Suspense>
            ) : (
              <MouthMap value={site} onChange={setSite} className="mx-auto mt-10 max-w-xs opacity-60" />
            )}
          </div>
        </Reveal>
      </div>
    </Section>
  );
}

/* ---------------------------------------------------------------- Future */

const FUTURE = [
  { icon: MonitorSmartphone, tag: "Coming soon", title: "OralAID Desktop for Windows", body: "Runs offline on clinic PCs. Batch-import images from folders and keep data on the premises." },
  { icon: Camera, tag: "In development", title: "Intraoral camera integration", body: "The camera captures, our model reads the frame in real time and flags suspicious areas while the patient is in the chair." },
  { icon: Stethoscope, tag: "Partnerships open", title: "Inside dental chairs and units", body: "An embedded module for dental equipment makers, so screening happens at every check-up without an extra step." },
  { icon: Cpu, tag: "Research", title: "Edge AI on-device", body: "A compressed model for edge devices and handheld screeners in rural camps, with no internet needed." },
];

export function FutureVision() {
  return (
    <Section id="future">
      <SectionHeading eyebrow="What's next" title={<>From the browser to <em>the dental chair.</em></>} align="center"
        body="Our goal is screening that happens automatically, wherever a mouth is examined: on a clinic PC, inside the intraoral camera, or on a device in a rural screening camp." />
      <div className="relative mt-16">
        <div className="grid gap-5 sm:grid-cols-2 lg:grid-cols-4">
          {FUTURE.map((f, i) => (
            <Reveal key={f.title} delay={i * 0.1}>
              <div className="card-stripe spotlight group relative h-full p-6 pt-7 transition hover:-translate-y-1">
                <span aria-hidden className="absolute right-4 top-3 font-display text-6xl font-bold text-ink/[.05]">0{i + 1}</span>
                <div className="relative mb-5 grid h-[72px] w-[72px] place-items-center rounded-full border border-brand/30 bg-[radial-gradient(circle_at_30%_30%,rgb(var(--glow-1)/.25),transparent_70%)] transition group-hover:shadow-[0_0_50px_rgb(var(--glow-1)/.35)]">
                  <f.icon className="h-8 w-8 text-brand" />
                </div>
                <span className="text-[11px] font-semibold uppercase tracking-wider text-accent">{f.tag}</span>
                <h3 className="mt-2 font-display text-lg font-semibold">{f.title}</h3>
                <p className="mt-2 text-sm text-muted">{f.body}</p>
              </div>
            </Reveal>
          ))}
        </div>
      </div>
      <Reveal>
        <div className="mt-10 flex flex-col items-center justify-center gap-3 text-sm text-muted sm:flex-row sm:gap-6">
          <span className="inline-flex items-center gap-2"><WifiOff className="h-4 w-4 text-brand" />Offline-first</span>
          <span className="inline-flex items-center gap-2"><Radio className="h-4 w-4 text-brand" />Real-time on the camera feed</span>
          <Link to="/roadmap" className="inline-flex items-center gap-1 font-medium text-brand hover:underline">Join the waitlist <ArrowRight className="h-4 w-4" /></Link>
        </div>
      </Reveal>
    </Section>
  );
}

/* ------------------------------------------------------------ Built with */

const STACK = [
  siPytorch, siPython, siFastapi, siPostgresql, siDocker, siOracle, siNetlify, siReact, siTypescript, siTailwindcss,
  siVite, siThreedotjs, siOpencv, siScikitlearn, siNumpy, siSqlalchemy, siGithubactions, siCaddy, siLetsencrypt,
];

function Logo({ icon }: { icon: (typeof STACK)[number] }) {
  return (
    <div className="group flex shrink-0 items-center gap-4 px-10 sm:px-12" title={icon.title}>
      <svg viewBox="0 0 24 24" className="h-10 w-10 fill-muted transition group-hover:fill-[var(--c)] sm:h-12 sm:w-12" style={{ ["--c" as string]: `#${icon.hex}` }} aria-hidden>
        <path d={icon.path} />
      </svg>
      <span className="whitespace-nowrap font-display text-xl font-semibold text-muted transition group-hover:text-ink sm:text-2xl">{icon.title}</span>
    </div>
  );
}

export function BuiltWith() {
  return (
    <section aria-label="Technologies OralAID is built with" className="relative border-y border-line py-14">
      <p className="mb-9 text-center text-xs font-semibold uppercase tracking-[0.2em] text-muted">Built with open technology</p>
      <div className="relative overflow-hidden [mask-image:linear-gradient(90deg,transparent,black_12%,black_88%,transparent)]">
        <div className="flex w-max animate-marquee hover:[animation-play-state:paused]">
          {[...STACK, ...STACK].map((icon, i) => <Logo key={`${icon.slug}-${i}`} icon={icon} />)}
        </div>
      </div>
      <p className="mt-6 text-center text-[11px] text-muted">Logos are trademarks of their owners and are shown to credit the tools we use. No endorsement or partnership is implied.</p>
    </section>
  );
}
