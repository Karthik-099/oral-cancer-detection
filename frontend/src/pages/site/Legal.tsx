import type { ReactNode } from "react";
import { Link } from "react-router-dom";
import { AlertTriangle } from "lucide-react";
import { useSeo } from "@/lib/seo";
import { cn } from "@/lib/utils";

type Kind = "privacy" | "terms";

interface Doc {
  title: string;
  lede: string;
  sections: { id: string; title: string; body: ReactNode }[];
}

const P = ({ children }: { children: ReactNode }) => <p>{children}</p>;
const UL = ({ items }: { items: ReactNode[] }) => (
  <ul className="list-disc space-y-1.5 pl-5 marker:text-line">{items.map((it, i) => <li key={i}>{it}</li>)}</ul>
);

const PRIVACY: Doc = {
  title: "Privacy Policy",
  lede: "How OralAID collects, uses and protects personal and clinical information.",
  sections: [
    { id: "who", title: "1. Who we are", body: <P>OralAID ("we", "us") provides explainable AI screening support software for oral health professionals. For data you upload as a customer, you are generally the controller and we act as your processor. For account, website and marketing data, we are the controller.</P> },
    { id: "collect", title: "2. Information we collect", body: <>
      <UL items={[
        <><strong>Account data:</strong> name, work email, organization, specialty and a bcrypt hash of your password.</>,
        <><strong>Clinical content:</strong> intraoral images, patient references, age, sex, risk factors, anatomical site, notes, model results and clinician feedback you choose to enter.</>,
        <><strong>Usage and security data:</strong> IP address, request identifiers, audit events (such as sign-ins, case views and exports) and basic device information.</>,
        <><strong>Enquiries:</strong> details you submit through contact, demo, deployment or waitlist forms.</>,
        <><strong>Demo uploads:</strong> images uploaded to the public demo and their results, deleted automatically after one hour.</>,
      ]} />
      <P>We ask you not to upload directly identifying information such as full names or full-face photos unless your deployment and agreements require it.</P>
    </> },
    { id: "use", title: "3. How we use information", body: <UL items={[
      "To provide the service: run analyses, generate explanations, store cases and produce reports.",
      "To secure the service: authenticate users, rate limit, detect abuse and maintain audit logs.",
      "To support you and respond to enquiries.",
      "To improve models only where you have explicitly opted in. We do not train on customer images or feedback by default, and never on public demo uploads.",
    ]} /> },
    { id: "legal-bases", title: "4. Legal bases", body: <P>Where data protection law such as the GDPR or India's DPDP Act applies, we rely on performance of a contract, our legitimate interests in operating and securing the service, compliance with legal obligations, and your consent where required (for example, model-improvement opt-in or marketing emails). Health data is processed on your instructions as a healthcare provider.</P> },
    { id: "sharing", title: "5. Sharing", body: <>
      <P>We do not sell personal data. We share it only with:</P>
      <UL items={["Infrastructure and email sub-processors bound by data processing terms.", "Authorities where required by law.", "A successor entity in a merger or acquisition, under equivalent protections."]} />
    </> },
    { id: "security", title: "6. Security", body: <P>We use encryption in transit, signed and expiring image URLs, hashed passwords and API keys, account lockout, rate limiting, tenant isolation and audit logging. See our <Link to="/security" className="text-brand underline">Security page</Link>. No system is perfectly secure; we will notify affected customers of a personal data breach as required by law.</P> },
    { id: "retention", title: "7. Retention", body: <P>Customer content is retained for the life of your account or until you delete it. Public demo images are deleted after one hour. Audit logs are kept for a period appropriate to security and legal needs. On account closure we delete or return customer content within a reasonable period, except where law requires retention.</P> },
    { id: "rights", title: "8. Your rights", body: <P>Depending on where you are, you may have the right to access, correct, delete, restrict or port your personal data, to object to processing, and to withdraw consent. Patients whose data was uploaded by a clinician should contact that clinician, who controls the data. To exercise rights, email <a href="mailto:hello@oralaid.ai" className="text-brand underline">hello@oralaid.ai</a>.</P> },
    { id: "transfers", title: "9. International transfers", body: <P>Cloud SaaS data may be processed in regions where we or our sub-processors operate, under appropriate safeguards. Customers who need data to stay in a specific jurisdiction can use a private VPC or on-premise deployment.</P> },
    { id: "cookies", title: "10. Cookies and local storage", body: <P>We use local storage for your session token and small interface preferences. We do not use third-party advertising cookies.</P> },
    { id: "children", title: "11. Children", body: <P>The service is for healthcare professionals and is not directed at children. Clinical data about minors may be uploaded by clinicians under their own legal obligations.</P> },
    { id: "changes", title: "12. Changes and contact", body: <P>We will post updates here and notify customers of material changes. Questions: <a href="mailto:hello@oralaid.ai" className="text-brand underline">hello@oralaid.ai</a>. Security issues: <a href="mailto:security@oralaid.ai" className="text-brand underline">security@oralaid.ai</a>.</P> },
  ],
};

const TERMS: Doc = {
  title: "Terms of Service",
  lede: "The agreement between you and OralAID for use of the website, app and API.",
  sections: [
    { id: "acceptance", title: "1. Acceptance", body: <P>By creating an account or using OralAID you agree to these Terms. If you use OralAID on behalf of an organization, you confirm you have authority to bind it. Enterprise customers may have a separate written agreement, which prevails where it conflicts with these Terms.</P> },
    { id: "nature", title: "2. Clinical decision support only", body: <>
      <P><strong>OralAID is a screening aid, not a diagnostic device.</strong> It is not cleared or approved as a medical device in any jurisdiction and is provided for clinical decision support and research use.</P>
      <UL items={[
        "Outputs are probabilistic estimates with known limitations, described on our Models page, and may be wrong.",
        "You remain solely responsible for all clinical decisions, examination, diagnosis, referral and treatment.",
        "Suspicious findings should be referred for specialist evaluation and biopsy. A low-risk result does not rule out disease.",
        "Do not use OralAID as the sole basis for any clinical decision or in emergencies.",
      ]} />
    </> },
    { id: "eligibility", title: "3. Eligibility and accounts", body: <P>The service is intended for qualified healthcare professionals, researchers and their authorised staff. Keep credentials and API keys confidential and notify us promptly of any unauthorised use. You are responsible for activity under your account.</P> },
    { id: "your-data", title: "4. Your data", body: <P>You retain all rights to the content you upload. You grant us a limited licence to process it solely to provide and secure the service. We will use it to improve models only if you opt in. You are responsible for having the necessary consents and legal bases to upload patient data, and for complying with laws that apply to you.</P> },
    { id: "acceptable-use", title: "5. Acceptable use", body: <UL items={[
      "No unlawful, infringing or harmful content, and no images uploaded without appropriate authority.",
      "No attempts to probe, scan or breach security outside our responsible disclosure policy.",
      "No reverse engineering, scraping or extracting model weights, and no use to build a competing model.",
      "No circumventing rate limits, usage caps or access controls.",
    ]} /> },
    { id: "plans", title: "6. Plans, fees and trials", body: <P>Paid plans are billed in advance, monthly or annually, as shown at checkout or in your order. Prices on our website are indicative. Fees are non-refundable except where required by law or stated in your order. We may change prices with at least 30 days' notice, effective at your next renewal.</P> },
    { id: "availability", title: "7. Availability and changes", body: <P>We aim for high availability but the service is provided without an uptime commitment unless an SLA is included in your plan. We may update models and features; material changes to models will be reflected on the Models page.</P> },
    { id: "ip", title: "8. Intellectual property", body: <P>We own the service, software and models. Feedback you give us may be used to improve the service without obligation to you.</P> },
    { id: "warranty", title: "9. Disclaimers", body: <P>To the maximum extent permitted by law, the service is provided "as is" without warranties of any kind, including fitness for a particular purpose, accuracy of outputs, or non-infringement.</P> },
    { id: "liability", title: "10. Limitation of liability", body: <P>To the maximum extent permitted by law, we are not liable for indirect, incidental, special or consequential damages, or for clinical outcomes arising from reliance on outputs. Our total liability in any 12-month period is limited to the fees you paid us in that period.</P> },
    { id: "termination", title: "11. Suspension and termination", body: <P>You may stop using the service at any time. We may suspend or terminate access for material breach or to protect the service or others. On termination you may export your data for 30 days, after which we may delete it.</P> },
    { id: "law", title: "12. Governing law and contact", body: <P>These Terms are governed by the laws specified in your order or, if none, the laws of our place of incorporation. Questions: <a href="mailto:hello@oralaid.ai" className="text-brand underline">hello@oralaid.ai</a>.</P> },
  ],
};

export default function Legal({ kind }: { kind: Kind }) {
  const doc = kind === "privacy" ? PRIVACY : TERMS;
  useSeo(doc.title, doc.lede);
  return (
    <div className="container py-14 sm:py-20">
      <header className="max-w-3xl border-b border-line pb-10">
        <p className="eyebrow mb-4 text-brand">Legal</p>
        <h1 className="display text-5xl sm:text-6xl">{doc.title}</h1>
        <p className="mt-4 text-base text-muted sm:text-lg">{doc.lede}</p>
        <div className="mt-6 flex flex-wrap items-center gap-x-6 gap-y-2 text-sm text-muted">
          <span>Last updated <time dateTime="2026-10-01" className="font-medium text-ink">October 2026</time></span>
          <nav aria-label="Legal documents" className="flex gap-4">
            <Link to="/privacy" className={cn("hover:text-ink", kind === "privacy" && "font-medium text-brand")}>Privacy</Link>
            <Link to="/terms" className={cn("hover:text-ink", kind === "terms" && "font-medium text-brand")}>Terms</Link>
          </nav>
        </div>
      </header>

      <div className="mt-10 grid gap-12 lg:grid-cols-[220px_minmax(0,1fr)] lg:gap-16">
        <aside className="hidden lg:block">
          <nav aria-label="Sections" className="sticky top-24">
            <p className="eyebrow mb-3">Contents</p>
            <ul className="space-y-1.5 text-sm">
              {doc.sections.map((s) => <li key={s.id}><a href={`#${s.id}`} className="text-muted transition hover:text-ink">{s.title}</a></li>)}
            </ul>
          </nav>
        </aside>
        <article className="min-w-0 max-w-3xl">
          <div role="note" className="mb-10 flex gap-3 rounded-2xl border border-moderate/25 bg-moderate/5 p-4 text-sm text-ink/90">
            <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0 text-moderate" />
            <p>This document is a template provided for transparency and must be reviewed by qualified legal counsel before it is relied upon. It does not constitute legal advice.</p>
          </div>
          <div className="space-y-10">
            {doc.sections.map((s) => (
              <section key={s.id} id={s.id} className="scroll-mt-24">
                <h2 className="text-xl font-semibold tracking-tight">{s.title}</h2>
                <div className="mt-3 space-y-3 text-[15px] leading-relaxed text-muted [&_strong]:font-medium [&_strong]:text-ink">{s.body}</div>
              </section>
            ))}
          </div>
        </article>
      </div>
    </div>
  );
}
