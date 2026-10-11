import { useEffect, useState, type FormEvent } from "react";
import { Link, useLocation, useNavigate } from "react-router-dom";
import { ArrowRight, Eye, EyeOff } from "lucide-react";
import { api } from "@/lib/api";
import { useAuth } from "@/lib/auth";
import { useSeo } from "@/lib/seo";
import { Button, ErrorNote, Field } from "@/components/ui";
import { AuthLayout } from "@/components/app/AuthLayout";
import { errorMessage } from "@/components/app/kit";

export default function Login() {
  useSeo("Sign in", undefined, { noindex: true });
  const navigate = useNavigate();
  const location = useLocation();
  const { user, status, setSession } = useAuth();
  const from = (location.state as { from?: string } | null)?.from || "/app";
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [show, setShow] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (status === "ready" && user) navigate(from, { replace: true });
  }, [status, user, from, navigate]);

  const submit = async (e: FormEvent) => {
    e.preventDefault();
    setError(null);
    setLoading(true);
    try {
      const res = await api.login({ email: email.trim(), password });
      setSession(res.token, res.user);
      navigate(from, { replace: true });
    } catch (err) {
      setError(errorMessage(err, "Couldn't sign you in. Please try again."));
    } finally {
      setLoading(false);
    }
  };

  return (
    <AuthLayout
      title="Welcome back"
      subtitle="Sign in to your OralAID workspace."
      footer={<>New to OralAID? <Link to="/register" state={location.state} className="font-medium text-brand hover:underline">Create an account</Link></>}
    >
      <form onSubmit={submit} className="space-y-4" noValidate>
        <Field label="Work email">
          <input className="input" type="email" autoComplete="email" required autoFocus value={email}
            onChange={(e) => setEmail(e.target.value)} placeholder="you@clinic.org" />
        </Field>
        <Field label="Password">
          <div className="relative">
            <input className="input pr-11" type={show ? "text" : "password"} autoComplete="current-password" required value={password}
              onChange={(e) => setPassword(e.target.value)} />
            <button type="button" onClick={() => setShow((s) => !s)} aria-label={show ? "Hide password" : "Show password"}
              className="absolute right-1.5 top-1/2 grid h-8 w-8 -translate-y-1/2 place-items-center rounded-lg text-muted hover:text-ink">
              {show ? <EyeOff className="h-4 w-4" /> : <Eye className="h-4 w-4" />}
            </button>
          </div>
        </Field>
        <ErrorNote>{error}</ErrorNote>
        <Button type="submit" size="lg" className="w-full" loading={loading} disabled={!email || !password}>
          Sign in{!loading && <ArrowRight className="h-4 w-4" />}
        </Button>
      </form>
    </AuthLayout>
  );
}
