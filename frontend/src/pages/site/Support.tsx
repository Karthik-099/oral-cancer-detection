import { useState } from "react";
import { ArrowUpRight, Bug, Check, Coffee, Copy, GitPullRequest, Github, Heart, Star } from "lucide-react";
import { useSeo } from "@/lib/seo";
import { Reveal } from "@/components/Reveal";
import { CTABand, Section, SectionHeading } from "@/components/site/marketing";

import { REPO_URL } from "@/lib/links";
// Set in Netlify -> Environment variables. Empty means "not set up yet".
const BMC_URL = (import.meta.env.VITE_BMC_URL as string | undefined) || "";
const UPI_ID = (import.meta.env.VITE_UPI_ID as string | undefined) || "";

const WAYS = [
  { icon: Star, title: "Star the repo", body: "Stars help other clinicians and researchers find the project.", href: REPO_URL, cta: "Star on GitHub" },
  { icon: Bug, title: "Report an issue", body: "Found a bug or a wrong prediction? Open an issue with steps to reproduce.", href: `${REPO_URL}/issues/new`, cta: "Open an issue" },
  { icon: GitPullRequest, title: "Contribute code", body: "UI, model evaluation, docs, translations. Read CONTRIBUTING.md to start.", href: `${REPO_URL}/blob/prod/CONTRIBUTING.md`, cta: "How to contribute" },
];

export default function Support() {
  useSeo("Support OralAID", "OralAID is free and open source. Buy us a coffee, star the repo or contribute to keep oral cancer screening open for everyone.");
  const [copied, setCopied] = useState(false);

  return (
    <>
      <Section className="pt-16 sm:pt-24">
        <div className="mx-auto max-w-3xl text-center">
          <Reveal>
            <div className="mx-auto mb-6 grid h-16 w-16 place-items-center rounded-2xl bg-brand/10 text-brand shadow-[0_0_40px_rgb(var(--glow-1)/.25)]">
              <Coffee className="h-8 w-8" />
            </div>
            <h1 className="display text-5xl sm:text-7xl">Buy us a <em>coffee.</em></h1>
            <p className="mx-auto mt-5 max-w-xl text-lg text-muted">
              OralAID is free and open source. Your support pays for servers, model validation and the time it takes
              to make early oral cancer screening available in every clinic.
            </p>
          </Reveal>

          <Reveal delay={0.1}>
            <div className="card glow-border mx-auto mt-10 max-w-md p-6 text-left sm:p-8">
              {BMC_URL ? (
                <a href={BMC_URL} target="_blank" rel="noreferrer"
                  className="flex w-full items-center justify-center gap-2 rounded-xl bg-[linear-gradient(110deg,rgb(var(--glow-1)),rgb(var(--glow-3))_50%,rgb(var(--glow-2)))] bg-[length:200%_100%] px-6 py-4 text-base font-semibold text-white transition-[background-position] duration-500 hover:bg-right">
                  <Coffee className="h-5 w-5" />Buy us a coffee<ArrowUpRight className="h-4 w-4" />
                </a>
              ) : (
                <p className="rounded-xl border border-dashed border-line px-4 py-4 text-center text-sm text-muted">
                  Coffee donations are opening soon. Meanwhile, starring the repo helps a lot.
                </p>
              )}
              {UPI_ID && (
                <div className="mt-5 rounded-xl border border-line bg-surface-2/60 p-4">
                  <p className="eyebrow mb-2">UPI (India)</p>
                  <div className="flex items-center justify-between gap-3">
                    <code className="truncate font-mono text-sm">{UPI_ID}</code>
                    <button
                      onClick={() => navigator.clipboard.writeText(UPI_ID).then(() => { setCopied(true); setTimeout(() => setCopied(false), 1500); })}
                      className="inline-flex shrink-0 items-center gap-1.5 rounded-lg border border-line px-3 py-1.5 text-xs hover:bg-surface-2">
                      {copied ? <Check className="h-3.5 w-3.5 text-low" /> : <Copy className="h-3.5 w-3.5" />}{copied ? "Copied" : "Copy"}
                    </button>
                  </div>
                </div>
              )}
              <a href={REPO_URL} target="_blank" rel="noreferrer"
                className="mt-5 flex items-center justify-center gap-2 rounded-xl border border-line px-6 py-3 text-sm font-medium transition hover:border-brand/50 hover:bg-surface-2">
                <Github className="h-4 w-4" />Star OralAID on GitHub
              </a>
            </div>
          </Reveal>
        </div>
      </Section>

      <Section>
        <SectionHeading eyebrow="Other ways to help" title={<>Support that <em>isn't money.</em></>} align="center" />
        <div className="mt-12 grid gap-5 md:grid-cols-3">
          {WAYS.map((w, i) => (
            <Reveal key={w.title} delay={i * 0.08}>
              <a href={w.href} target="_blank" rel="noreferrer" className="card group block h-full p-6 transition hover:-translate-y-1 hover:border-brand/40">
                <div className="mb-4 grid h-11 w-11 place-items-center rounded-xl bg-brand/10 text-brand"><w.icon className="h-5 w-5" /></div>
                <h3 className="font-display text-lg font-semibold">{w.title}</h3>
                <p className="mt-2 text-sm text-muted">{w.body}</p>
                <span className="mt-5 inline-flex items-center gap-1 text-sm font-medium text-brand">{w.cta}<ArrowUpRight className="h-4 w-4 transition group-hover:translate-x-0.5 group-hover:-translate-y-0.5" /></span>
              </a>
            </Reveal>
          ))}
        </div>
        <Reveal>
          <p className="mx-auto mt-12 flex max-w-xl items-center justify-center gap-2 text-center text-sm text-muted">
            <Heart className="h-4 w-4 text-high" />Every contribution is used for the project: hosting, validation studies and development.
          </p>
        </Reveal>
      </Section>
      <CTABand />
    </>
  );
}
