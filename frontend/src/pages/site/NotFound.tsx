import { Link } from "react-router-dom";
import { motion } from "framer-motion";
import { ArrowLeft, ArrowRight } from "lucide-react";
import { useSeo } from "@/lib/seo";
import { ButtonLink } from "@/components/ui";

const LINKS = [
  { to: "/platform", label: "Platform", d: "How the workflow fits together" },
  { to: "/models", label: "Models", d: "Model cards and evaluation" },
  { to: "/docs", label: "API docs", d: "Build on OralAID" },
  { to: "/contact", label: "Contact", d: "Talk to a human" },
];

export default function NotFound() {
  useSeo("Page not found", undefined, { noindex: true });
  return (
    <section className="relative overflow-hidden">
      <div aria-hidden className="pointer-events-none absolute inset-0 grid-bg [mask-image:radial-gradient(ellipse_60%_50%_at_50%_30%,black,transparent)]" />
      <div className="container relative flex min-h-[70vh] flex-col items-center justify-center py-20 text-center">
        <motion.div initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.7, ease: [0.2, 0.7, 0.2, 1] }}>
          <div className="relative mx-auto h-28 w-56 overflow-hidden" aria-hidden>
            <p className="display text-[7rem] leading-none text-ink/10">404</p>
            <div className="absolute inset-x-0 top-0 h-1/2 animate-scan bg-gradient-to-b from-transparent via-brand/20 to-transparent">
              <div className="absolute inset-x-0 bottom-1/2 h-px bg-brand" />
            </div>
          </div>
          <p className="eyebrow mt-2 text-brand">Error 404</p>
          <h1 className="display mt-4 text-5xl sm:text-6xl">We looked closely. <em className="italic text-brand">Nothing here.</em></h1>
          <p className="mx-auto mt-5 max-w-md text-base text-muted">The page you're after may have moved, or the link may be mistyped.</p>
          <div className="mt-8 flex flex-wrap justify-center gap-3">
            <ButtonLink to="/" size="lg" variant="ink"><ArrowLeft className="h-4 w-4" />Back to home</ButtonLink>
            <ButtonLink to="/#demo" size="lg" variant="secondary">Try the live demo</ButtonLink>
          </div>
        </motion.div>
        <ul className="mt-16 grid w-full max-w-3xl gap-3 text-left sm:grid-cols-2">
          {LINKS.map((l) => (
            <li key={l.to}>
              <Link to={l.to} className="group flex items-center justify-between gap-3 rounded-2xl border border-line bg-surface p-4 transition hover:border-brand/30 hover:shadow-soft">
                <span><span className="block text-sm font-semibold">{l.label}</span><span className="text-xs text-muted">{l.d}</span></span>
                <ArrowRight className="h-4 w-4 text-muted transition group-hover:translate-x-0.5 group-hover:text-brand" />
              </Link>
            </li>
          ))}
        </ul>
      </div>
    </section>
  );
}
