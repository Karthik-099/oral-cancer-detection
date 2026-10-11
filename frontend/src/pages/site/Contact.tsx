import { Link } from "react-router-dom";
import { BookOpen, Mail, MessageSquare, ShieldAlert } from "lucide-react";
import { useSeo } from "@/lib/seo";
import { Reveal } from "@/components/Reveal";
import { IconTile } from "@/components/site/marketing";
import { LeadForm } from "@/components/site/LeadForm";

const CONTACTS = [
  { icon: <Mail />, t: "General and sales", v: "hello@oralaid.ai", href: "mailto:hello@oralaid.ai" },
  { icon: <ShieldAlert />, t: "Security reports", v: "security@oralaid.ai", href: "mailto:security@oralaid.ai" },
];

export default function Contact() {
  useSeo("Contact", "Book a demo of OralAID, ask a question, or talk to us about partnerships and deployment.");
  return (
    <section className="relative overflow-hidden">
      <div aria-hidden className="pointer-events-none absolute inset-0 grid-bg [mask-image:radial-gradient(ellipse_70%_50%_at_30%_0%,black,transparent)]" />
      <div aria-hidden className="pointer-events-none absolute -top-40 left-0 h-96 w-96 rounded-full bg-brand/10 blur-3xl" />
      <div className="container relative grid gap-14 py-16 sm:py-24 lg:grid-cols-[0.9fr_1.1fr] lg:gap-20">
        <div>
          <p className="eyebrow mb-5 text-brand">Contact</p>
          <h1 className="display text-5xl sm:text-7xl">Let's <em className="italic text-brand">talk.</em></h1>
          <p className="mt-6 max-w-md text-base leading-relaxed text-muted sm:text-lg">
            Book a 30-minute demo on your own images, ask about deployment, or explore a research or hardware partnership.
          </p>
          <ul className="mt-10 space-y-3">
            {CONTACTS.map((c) => (
              <li key={c.v}>
                <a href={c.href} className="group flex items-center gap-4 rounded-2xl border border-line bg-surface p-4 transition hover:border-brand/30 hover:shadow-soft">
                  <IconTile tone="neutral">{c.icon}</IconTile>
                  <span>
                    <span className="block text-xs text-muted">{c.t}</span>
                    <span className="font-mono text-sm transition group-hover:text-brand">{c.v}</span>
                  </span>
                </a>
              </li>
            ))}
          </ul>
          <div className="mt-8 grid gap-3 text-sm text-muted sm:grid-cols-2">
            <Link to="/docs" className="flex items-center gap-2 hover:text-ink"><BookOpen className="h-4 w-4 text-brand" />Developer docs</Link>
            <Link to="/deployment#talk" className="flex items-center gap-2 hover:text-ink"><MessageSquare className="h-4 w-4 text-brand" />Deployment questions</Link>
          </div>
          <p className="mt-10 max-w-md text-xs leading-relaxed text-muted">
            Please don't send patient-identifiable information through this form. For clinical concerns about a patient, consult a specialist directly.
          </p>
        </div>

        <Reveal className="card p-6 shadow-lift sm:p-9">
          <h2 className="text-xl font-semibold tracking-tight">Send us a message</h2>
          <p className="mb-7 mt-1 text-sm text-muted">We usually reply within two working days.</p>
          <LeadForm
            kind={(type) => (type === "demo" ? "demo" : "contact")}
            select={{ label: "What can we help with?", name: "type", options: [
              { value: "demo", label: "Book a demo" },
              { value: "general", label: "General question" },
              { value: "partnership", label: "Partnership" },
              { value: "press", label: "Press" },
            ] }}
            messagePlaceholder="Tell us about your practice, volume and what you'd like to see."
            submitLabel="Send message"
            successTitle="Message sent."
            successBody="Thanks for reaching out. We'll get back to you within two working days."
          />
        </Reveal>
      </div>
    </section>
  );
}
