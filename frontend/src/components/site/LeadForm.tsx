import { useState, type FormEvent } from "react";
import { Link } from "react-router-dom";
import { AnimatePresence, motion } from "framer-motion";
import { ArrowRight, CheckCircle2 } from "lucide-react";
import { api } from "@/lib/api";
import { Button, ErrorNote, Field } from "@/components/ui";
import { cn } from "@/lib/utils";

interface SelectConfig {
  label: string;
  name: string;
  options: { value: string; label: string }[];
  initial?: string;
}

/**
 * Generic lead capture form posting to /api/leads. `kind` may depend on the
 * selected option (e.g. "Book a demo" -> demo, everything else -> contact).
 * Includes a visually hidden honeypot field named `website`.
 */
export function LeadForm({
  kind, select, withName = true, withOrganization = true, withMessage = true, messageLabel = "Message",
  messagePlaceholder, submitLabel = "Send", successTitle = "Thanks, we'll be in touch.", successBody = "We usually reply within two working days.",
  meta, className, layout = "stack",
}: {
  kind: string | ((selected: string) => string);
  select?: SelectConfig;
  withName?: boolean;
  withOrganization?: boolean;
  withMessage?: boolean;
  messageLabel?: string;
  messagePlaceholder?: string;
  submitLabel?: string;
  successTitle?: string;
  successBody?: string;
  meta?: Record<string, unknown>;
  className?: string;
  layout?: "stack" | "inline";
}) {
  const [name, setName] = useState("");
  const [email, setEmail] = useState("");
  const [organization, setOrganization] = useState("");
  const [message, setMessage] = useState("");
  const [choice, setChoice] = useState(select?.initial ?? select?.options[0]?.value ?? "");
  const [website, setWebsite] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [done, setDone] = useState(false);

  const submit = async (e: FormEvent) => {
    e.preventDefault();
    setError(null);
    setBusy(true);
    try {
      const resolvedKind = typeof kind === "function" ? kind(choice) : kind;
      await api.lead({
        kind: resolvedKind,
        email,
        name: name || undefined,
        organization: organization || undefined,
        message: message || undefined,
        meta: { ...meta, ...(select ? { [select.name]: choice } : {}), page: window.location.pathname },
        website,
      });
      setDone(true);
    } catch (err) {
      setError((err as Error).message || "Something went wrong. Please try again.");
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className={cn("relative", className)}>
      <AnimatePresence mode="wait" initial={false}>
        {done ? (
          <motion.div key="done" initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0 }}
            className="flex flex-col items-start gap-3 py-6" role="status">
            <div className="grid h-11 w-11 place-items-center rounded-full bg-low/10 text-low"><CheckCircle2 className="h-5 w-5" /></div>
            <p className="text-lg font-semibold">{successTitle}</p>
            <p className="text-sm text-muted">{successBody}</p>
            <button type="button" className="text-sm font-medium text-brand hover:underline" onClick={() => { setDone(false); setMessage(""); }}>
              Send another
            </button>
          </motion.div>
        ) : (
          <motion.form key="form" onSubmit={submit} initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
            className={cn(layout === "inline" ? "space-y-3" : "space-y-4")}>
            {/* Honeypot: real users never see or fill this. */}
            <div aria-hidden className="absolute -left-[10000px] top-auto h-px w-px overflow-hidden">
              <label>Website<input type="text" name="website" tabIndex={-1} autoComplete="off" value={website} onChange={(e) => setWebsite(e.target.value)} /></label>
            </div>

            {select && (
              <Field label={select.label}>
                <select className="input appearance-none" value={choice} onChange={(e) => setChoice(e.target.value)} name={select.name}>
                  {select.options.map((o) => <option key={o.value} value={o.value}>{o.label}</option>)}
                </select>
              </Field>
            )}

            <div className={cn("grid gap-4", (withName || withOrganization) && "sm:grid-cols-2")}>
              {withName && (
                <Field label="Name">
                  <input className="input" name="name" autoComplete="name" value={name} onChange={(e) => setName(e.target.value)} required maxLength={120} />
                </Field>
              )}
              <Field label="Work email">
                <input className="input" type="email" name="email" autoComplete="email" value={email} onChange={(e) => setEmail(e.target.value)} required placeholder="you@clinic.org" />
              </Field>
              {withOrganization && (
                <Field label="Organization" className={cn(withName && "sm:col-span-2")}>
                  <input className="input" name="organization" autoComplete="organization" value={organization} onChange={(e) => setOrganization(e.target.value)} maxLength={160} />
                </Field>
              )}
            </div>

            {withMessage && (
              <Field label={messageLabel}>
                <textarea className="input min-h-[120px] resize-y" name="message" value={message} onChange={(e) => setMessage(e.target.value)}
                  placeholder={messagePlaceholder} maxLength={4000} />
              </Field>
            )}

            <ErrorNote>{error}</ErrorNote>

            <div className="flex flex-wrap items-center gap-3 pt-1">
              <Button type="submit" loading={busy} size="lg" variant="ink">{submitLabel} {!busy && <ArrowRight className="h-4 w-4" />}</Button>
              <p className="text-xs text-muted">We only use your details to reply. See our <Link to="/privacy" className="underline hover:text-ink">privacy policy</Link>.</p>
            </div>
          </motion.form>
        )}
      </AnimatePresence>
    </div>
  );
}
