import { useState, type FormEvent } from "react";
import { Link, useLocation, useNavigate } from "react-router-dom";
import { ArrowRight, Eye, EyeOff } from "lucide-react";
import { api } from "@/lib/api";
import { useAuth } from "@/lib/auth";
import { useSeo } from "@/lib/seo";
import { Button, ErrorNote, Field } from "@/components/ui";
import { AuthLayout } from "@/components/app/AuthLayout";
import { errorMessage, SPECIALTIES } from "@/components/app/kit";
import { PasswordChecklist, passwordChecks } from "@/components/app/PasswordChecklist";

export default function Register() {
  useSeo("Create account", undefined, { noindex: true });
  const navigate = useNavigate();
  const location = useLocation();
  const setSession = useAuth((s) => s.setSession);
  const [form, setForm] = useState({ full_name: "", email: "", password: "", organization: "", specialty: "" });
  const [consent, setConsent] = useState(false);
  const [show, setShow] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const set = (k: keyof typeof form) => (e: { target: { value: string } }) => setForm((f) => ({ ...f, [k]: e.target.value }));

  const pwOk = passwordChecks(form.password, form.email).every((c) => c.ok);
  const valid = form.full_name.trim() && /\S+@\S+\.\S+/.test(form.email) && pwOk && consent;

  const submit = async (e: FormEvent) => {
    e.preventDefault();
    if (!valid) return;
    setError(null);
    setLoading(true);
    try {
      const res = await api.register({
        full_name: form.full_name.trim(),
        email: form.email.trim(),
        password: form.password,
        organization: form.organization.trim() || undefined,
        specialty: form.specialty || undefined,
      });
      setSession(res.token, res.user);
      const from = (location.state as { from?: string } | null)?.from;
      navigate(from || "/app/new?welcome=1", { replace: true });
    } catch (err) {
      setError(errorMessage(err, "Couldn't create your account. Please try again."));
    } finally {
      setLoading(false);
    }
  };

  return (
    <AuthLayout
      title="Create your workspace"
      subtitle="Start screening with explainable AI in a few minutes."
      footer={<>Already have an account? <Link to="/login" state={location.state} className="font-medium text-brand hover:underline">Sign in</Link></>}
    >
      <form onSubmit={submit} className="space-y-4" noValidate>
        <Field label="Full name">
          <input className="input" autoComplete="name" required value={form.full_name} onChange={set("full_name")} placeholder="Dr. Asha Rao" maxLength={120} />
        </Field>
        <Field label="Work email">
          <input className="input" type="email" autoComplete="email" required value={form.email} onChange={set("email")} placeholder="you@clinic.org" />
        </Field>
        <div>
          <label className="label" htmlFor="reg-pw">Password</label>
          <div className="relative">
            <input id="reg-pw" className="input pr-11" type={show ? "text" : "password"} autoComplete="new-password" required
              value={form.password} onChange={set("password")} maxLength={128} />
            <button type="button" onClick={() => setShow((s) => !s)} aria-label={show ? "Hide password" : "Show password"}
              className="absolute right-1.5 top-1/2 grid h-8 w-8 -translate-y-1/2 place-items-center rounded-lg text-muted hover:text-ink">
              {show ? <EyeOff className="h-4 w-4" /> : <Eye className="h-4 w-4" />}
            </button>
          </div>
          <PasswordChecklist password={form.password} email={form.email} />
        </div>
        <div className="grid gap-4 sm:grid-cols-2">
          <Field label="Organization">
            <input className="input" autoComplete="organization" value={form.organization} onChange={set("organization")} placeholder="Clinic or hospital" maxLength={160} />
          </Field>
          <Field label="Specialty">
            <select className="input" value={form.specialty} onChange={set("specialty")}>
              <option value="">Select...</option>
              {SPECIALTIES.map((s) => <option key={s} value={s}>{s}</option>)}
            </select>
          </Field>
        </div>
        <label className="flex cursor-pointer items-start gap-3 rounded-xl border border-line bg-surface-2/40 p-3 text-sm text-muted">
          <input type="checkbox" checked={consent} onChange={(e) => setConsent(e.target.checked)} className="mt-0.5 h-4 w-4 shrink-0 accent-[rgb(var(--brand))]" />
          <span>
            I agree to the <Link to="/terms" target="_blank" className="font-medium text-ink underline underline-offset-2">Terms</Link> and{" "}
            <Link to="/privacy" target="_blank" className="font-medium text-ink underline underline-offset-2">Privacy Policy</Link>, and understand
            OralAID supports, and does not replace, clinical judgement.
          </span>
        </label>
        <ErrorNote>{error}</ErrorNote>
        <Button type="submit" size="lg" className="w-full" loading={loading} disabled={!valid}>
          Create account{!loading && <ArrowRight className="h-4 w-4" />}
        </Button>
      </form>
    </AuthLayout>
  );
}
