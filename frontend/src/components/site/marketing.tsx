import { useId, useState, type ReactNode } from "react";
import { AnimatePresence, motion } from "framer-motion";
import { ArrowRight, Check, Plus } from "lucide-react";
import { Reveal } from "@/components/Reveal";
import { ButtonLink } from "@/components/ui";
import { useAuth } from "@/lib/auth";
import { cn } from "@/lib/utils";

/** Eyebrow + serif title + optional lede, used to open every marketing section. */
export function SectionHeading({ eyebrow, title, body, align = "left", className, as = "h2" }: {
  eyebrow?: string;
  title: ReactNode;
  body?: ReactNode;
  align?: "left" | "center";
  className?: string;
  as?: "h1" | "h2";
}) {
  const H = as;
  return (
    <Reveal className={cn("max-w-2xl", align === "center" && "mx-auto text-center", className)}>
      {eyebrow && <p className="eyebrow mb-4 text-brand">{eyebrow}</p>}
      <H className={cn("display", as === "h1" ? "text-5xl sm:text-7xl" : "text-4xl sm:text-5xl")}>{title}</H>
      {body && <p className="mt-5 text-base leading-relaxed text-muted sm:text-lg">{body}</p>}
    </Reveal>
  );
}

/** Top-of-page hero for interior pages: faded grid, h1, lede, actions. */
export function PageHero({ eyebrow, title, body, children, className }: {
  eyebrow: string; title: ReactNode; body?: ReactNode; children?: ReactNode; className?: string;
}) {
  return (
    <section className={cn("relative overflow-hidden border-b border-line", className)}>
      <div aria-hidden className="pointer-events-none absolute inset-0 grid-bg [mask-image:radial-gradient(ellipse_70%_60%_at_50%_0%,black,transparent)]" />
      <div aria-hidden className="pointer-events-none absolute -top-40 left-1/2 h-80 w-[min(900px,120vw)] -translate-x-1/2 rounded-full bg-brand/10 blur-3xl" />
      <div className="container relative py-20 sm:py-28">
        <motion.div initial={{ opacity: 0, y: 14 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.7, ease: [0.2, 0.7, 0.2, 1] }} className="max-w-3xl">
          <p className="eyebrow mb-5 text-brand">{eyebrow}</p>
          <h1 className="display text-5xl sm:text-7xl">{title}</h1>
          {body && <p className="mt-6 max-w-2xl text-base leading-relaxed text-muted sm:text-lg">{body}</p>}
          {children && <div className="mt-8 flex flex-wrap gap-3">{children}</div>}
        </motion.div>
      </div>
    </section>
  );
}

export function Section({ id, className, children }: { id?: string; className?: string; children: ReactNode }) {
  return (
    <section id={id} className={cn("scroll-mt-20 overflow-x-clip py-20 sm:py-28", className)}>
      <div className="container">{children}</div>
    </section>
  );
}

export function CheckList({ items, className }: { items: ReactNode[]; className?: string }) {
  return (
    <ul className={cn("space-y-2.5", className)}>
      {items.map((it, i) => (
        <li key={i} className="flex gap-2.5 text-sm text-ink/90">
          <Check className="mt-0.5 h-4 w-4 shrink-0 text-brand" aria-hidden />
          <span>{it}</span>
        </li>
      ))}
    </ul>
  );
}

/** Accessible accordion. One item open at a time. */
export function FAQ({ items, className }: { items: { q: string; a: ReactNode }[]; className?: string }) {
  const [open, setOpen] = useState<number | null>(0);
  const base = useId();
  return (
    <div className={cn("divide-y divide-line border-y border-line", className)}>
      {items.map((it, i) => {
        const isOpen = open === i;
        const id = `${base}-${i}`;
        return (
          <div key={it.q}>
            <h3>
              <button
                className="group flex w-full items-center justify-between gap-6 py-5 text-left text-[15px] font-medium transition hover:text-brand sm:text-base"
                aria-expanded={isOpen} aria-controls={id} onClick={() => setOpen(isOpen ? null : i)}
              >
                {it.q}
                <span className={cn("grid h-7 w-7 shrink-0 place-items-center rounded-full border border-line text-muted transition group-hover:border-brand/40 group-hover:text-brand", isOpen && "rotate-45 border-brand/40 text-brand")}>
                  <Plus className="h-3.5 w-3.5" />
                </span>
              </button>
            </h3>
            <AnimatePresence initial={false}>
              {isOpen && (
                <motion.div id={id} role="region"
                  initial={{ height: 0, opacity: 0 }} animate={{ height: "auto", opacity: 1 }} exit={{ height: 0, opacity: 0 }}
                  transition={{ duration: 0.3, ease: [0.2, 0.7, 0.2, 1] }} className="overflow-hidden">
                  <div className="max-w-3xl pb-6 pr-10 text-sm leading-relaxed text-muted sm:text-[15px]">{it.a}</div>
                </motion.div>
              )}
            </AnimatePresence>
          </div>
        );
      })}
    </div>
  );
}

/** Closing call to action band. */
export function CTABand({ title, body, className }: { title?: ReactNode; body?: ReactNode; className?: string }) {
  const user = useAuth((s) => s.user);
  return (
    <section className={cn("container py-12 sm:py-20", className)}>
      <Reveal className="glow-border relative overflow-hidden rounded-3xl border border-line bg-surface px-6 py-14 sm:px-14 sm:py-20">
        <div aria-hidden className="pointer-events-none absolute -right-32 -top-32 h-96 w-96 rounded-full bg-[radial-gradient(circle,rgb(var(--glow-1)/.28),transparent_65%)] blur-2xl" />
        <div aria-hidden className="pointer-events-none absolute -bottom-40 left-0 h-96 w-96 rounded-full bg-[radial-gradient(circle,rgb(var(--glow-2)/.22),transparent_65%)] blur-2xl" />
        <div className="relative max-w-2xl">
          <h2 className="display text-4xl sm:text-6xl">{title ?? <>Screen earlier. <em>Explain everything.</em></>}</h2>
          <p className="mt-5 text-base text-muted sm:text-lg">
            {body ?? "Start with 50 free analyses a month. No credit card. Bring your own images and see what the model sees."}
          </p>
          <div className="mt-8 flex flex-wrap gap-3">
            <ButtonLink to={user ? "/app" : "/register"} size="lg">
              {user ? "Open the app" : "Start free"} <ArrowRight className="h-4 w-4" />
            </ButtonLink>
            <ButtonLink to="/contact" size="lg" variant="secondary">Book a demo</ButtonLink>
          </div>
        </div>
      </Reveal>
    </section>
  );
}

/** The standing disclaimer, phrased once and reused. */
export function ClinicalNote({ className }: { className?: string }) {
  return (
    <p className={cn("text-xs leading-relaxed text-muted", className)}>
      OralAID is clinical decision support for screening. It does not diagnose, is not cleared as a medical device, and never replaces
      examination by a qualified clinician. Suspicious findings should be referred to a specialist for evaluation and biopsy.
    </p>
  );
}

/** Small icon tile used in feature grids. */
export function IconTile({ children, tone = "brand", className }: { children: ReactNode; tone?: "brand" | "accent" | "neutral"; className?: string }) {
  const tones = {
    brand: "bg-brand/10 text-brand ring-brand/15",
    accent: "bg-accent/10 text-accent ring-accent/15",
    neutral: "bg-surface-2 text-ink ring-line",
  };
  return <div className={cn("grid h-10 w-10 shrink-0 place-items-center rounded-xl ring-1 ring-inset [&_svg]:h-[18px] [&_svg]:w-[18px]", tones[tone], className)}>{children}</div>;
}
