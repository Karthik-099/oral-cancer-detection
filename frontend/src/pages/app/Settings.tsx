import { useEffect, useState, type FormEvent, type ReactNode } from "react";
import { Link, useLocation, useNavigate } from "react-router-dom";
import {
  Activity, AlertTriangle, ArrowUpRight, Check, Copy, CreditCard, KeyRound, LogOut, Plus, Shield, Trash2, UserRound,
} from "lucide-react";
import { API_BASE, api, type ApiKey } from "@/lib/api";
import { useAuth } from "@/lib/auth";
import { useSeo } from "@/lib/seo";
import { cn, formatDate, timeAgo } from "@/lib/utils";
import { PageHeader } from "@/components/AppShell";
import { Badge, Button, Card, ErrorNote, Field, buttonClass } from "@/components/ui";
import { Modal } from "@/components/Modal";
import { toast } from "@/components/Toast";
import { ConfirmDialog, errorMessage, SPECIALTIES } from "@/components/app/kit";
import { PasswordChecklist, passwordChecks } from "@/components/app/PasswordChecklist";

const SECTIONS = [
  { id: "profile", label: "Profile", icon: UserRound },
  { id: "security", label: "Security", icon: Shield },
  { id: "keys", label: "API keys", icon: KeyRound },
  { id: "activity", label: "Activity", icon: Activity },
  { id: "plan", label: "Plan", icon: CreditCard },
] as const;

export default function Settings() {
  useSeo("Settings", undefined, { noindex: true });
  const { hash } = useLocation();
  const [active, setActive] = useState<string>(hash.replace("#", "") || "profile");

  useEffect(() => {
    const id = hash.replace("#", "");
    if (!id) return;
    setActive(id);
    // Wait a frame so async sections have laid out before scrolling.
    const t = setTimeout(() => document.getElementById(id)?.scrollIntoView({ behavior: "smooth", block: "start" }), 120);
    return () => clearTimeout(t);
  }, [hash]);

  useEffect(() => {
    const obs = new IntersectionObserver(
      (entries) => {
        const visible = entries.filter((e) => e.isIntersecting).sort((a, b) => a.boundingClientRect.top - b.boundingClientRect.top)[0];
        if (visible) setActive(visible.target.id);
      },
      { rootMargin: "-20% 0px -65% 0px" },
    );
    SECTIONS.forEach((s) => { const el = document.getElementById(s.id); if (el) obs.observe(el); });
    return () => obs.disconnect();
  }, []);

  return (
    <div>
      <PageHeader title="Settings" subtitle="Manage your profile, security, integrations and preferences." />
      <div className="lg:grid lg:grid-cols-[200px_minmax(0,1fr)] lg:gap-10">
        <nav aria-label="Settings sections" className="sticky top-14 z-10 -mx-4 mb-6 overflow-x-auto border-b border-line bg-bg/90 px-4 py-2 backdrop-blur lg:top-10 lg:mx-0 lg:mb-0 lg:self-start lg:border-0 lg:bg-transparent lg:p-0 lg:backdrop-blur-none">
          <ul className="flex gap-1 lg:flex-col">
            {SECTIONS.map((s) => (
              <li key={s.id}>
                <a href={`#${s.id}`}
                  onClick={(e) => { e.preventDefault(); setActive(s.id); history.replaceState(null, "", `#${s.id}`); document.getElementById(s.id)?.scrollIntoView({ behavior: "smooth", block: "start" }); }}
                  className={cn("flex items-center gap-2.5 whitespace-nowrap rounded-lg px-3 py-2 text-sm transition",
                    active === s.id ? "bg-surface-2 font-medium text-ink" : "text-muted hover:text-ink")}>
                  <s.icon className="h-4 w-4" />{s.label}
                </a>
              </li>
            ))}
          </ul>
        </nav>
        <div className="min-w-0 space-y-6">
          <Profile />
          <Security />
          <Keys />
          <ActivityLog />
          <Plan />
        </div>
      </div>
    </div>
  );
}

function Section({ id, title, desc, children, footer }: { id: string; title: string; desc: string; children: ReactNode; footer?: ReactNode }) {
  return (
    <section id={id} className="scroll-mt-28 lg:scroll-mt-10">
      <Card className="overflow-hidden">
        <div className="border-b border-line px-5 py-4 sm:px-6">
          <h2 className="text-[15px] font-semibold">{title}</h2>
          <p className="mt-0.5 text-[13px] text-muted">{desc}</p>
        </div>
        <div className="p-5 sm:p-6">{children}</div>
        {footer && <div className="flex flex-wrap items-center justify-end gap-2 border-t border-line bg-surface-2/40 px-5 py-3 sm:px-6">{footer}</div>}
      </Card>
    </section>
  );
}

function Profile() {
  const { user, setUser } = useAuth();
  const [form, setForm] = useState({ full_name: user?.full_name ?? "", organization: user?.organization ?? "", specialty: user?.specialty ?? "" });
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const dirty = form.full_name !== (user?.full_name ?? "") || form.organization !== (user?.organization ?? "") || form.specialty !== (user?.specialty ?? "");

  const save = async (e: FormEvent) => {
    e.preventDefault();
    if (!form.full_name.trim()) { setError("Name can't be empty."); return; }
    setSaving(true);
    setError(null);
    try {
      const u = await api.updateMe({ full_name: form.full_name.trim(), organization: form.organization.trim(), specialty: form.specialty });
      setUser(u);
      setForm({ full_name: u.full_name ?? "", organization: u.organization ?? "", specialty: u.specialty ?? "" });
      toast.success("Profile updated");
    } catch (err) {
      setError(errorMessage(err, "Couldn't update your profile."));
    } finally {
      setSaving(false);
    }
  };

  return (
    <form onSubmit={save}>
      <Section id="profile" title="Profile" desc="Shown on reports you print and in your workspace."
        footer={<Button type="submit" loading={saving} disabled={!dirty}>Save changes</Button>}>
        <div className="grid gap-4 sm:grid-cols-2">
          <Field label="Full name">
            <input className="input" value={form.full_name} maxLength={120} onChange={(e) => setForm({ ...form, full_name: e.target.value })} autoComplete="name" />
          </Field>
          <Field label="Email" hint="Contact support to change your sign-in email.">
            <input className="input opacity-70" value={user?.email ?? ""} disabled readOnly />
          </Field>
          <Field label="Organization">
            <input className="input" value={form.organization} maxLength={160} onChange={(e) => setForm({ ...form, organization: e.target.value })} autoComplete="organization" />
          </Field>
          <Field label="Specialty">
            <select className="input" value={form.specialty} onChange={(e) => setForm({ ...form, specialty: e.target.value })}>
              <option value="">Select...</option>
              {SPECIALTIES.map((s) => <option key={s} value={s}>{s}</option>)}
              {form.specialty && !(SPECIALTIES as readonly string[]).includes(form.specialty) && <option value={form.specialty}>{form.specialty}</option>}
            </select>
          </Field>
        </div>
        {error && <div className="mt-4"><ErrorNote>{error}</ErrorNote></div>}
      </Section>
    </form>
  );
}

function Security() {
  const { user, setSession, signOut } = useAuth();
  const navigate = useNavigate();
  const [current, setCurrent] = useState("");
  const [next, setNext] = useState("");
  const [confirm, setConfirm] = useState("");
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [confirmAll, setConfirmAll] = useState(false);
  const [signingOut, setSigningOut] = useState(false);

  const valid = current && passwordChecks(next, user?.email).every((c) => c.ok) && next === confirm;

  const change = async (e: FormEvent) => {
    e.preventDefault();
    if (!valid) return;
    setSaving(true);
    setError(null);
    try {
      const res = await api.changePassword({ current_password: current, new_password: next });
      setSession(res.token, res.user);
      setCurrent(""); setNext(""); setConfirm("");
      toast.success("Password changed. Other devices have been signed out.");
    } catch (err) {
      setError(errorMessage(err, "Couldn't change your password."));
    } finally {
      setSaving(false);
    }
  };

  const logoutAll = async () => {
    setSigningOut(true);
    try {
      await api.logoutAll();
      signOut();
      toast.success("Signed out of all devices");
      navigate("/login", { replace: true });
    } catch (err) {
      toast.error(errorMessage(err, "Couldn't sign out of all devices."));
      setSigningOut(false);
    }
  };

  return (
    <section id="security" className="scroll-mt-28 space-y-6 lg:scroll-mt-10">
      <form onSubmit={change}>
        <Card className="overflow-hidden">
          <div className="border-b border-line px-5 py-4 sm:px-6">
            <h2 className="text-[15px] font-semibold">Password</h2>
            <p className="mt-0.5 text-[13px] text-muted">Changing your password signs out every other session.</p>
          </div>
          <div className="grid gap-4 p-5 sm:grid-cols-2 sm:p-6">
            <Field label="Current password" className="sm:col-span-2 sm:max-w-[calc(50%-0.5rem)]">
              <input className="input" type="password" autoComplete="current-password" value={current} onChange={(e) => setCurrent(e.target.value)} />
            </Field>
            <div>
              <Field label="New password">
                <input className="input" type="password" autoComplete="new-password" value={next} maxLength={128} onChange={(e) => setNext(e.target.value)} />
              </Field>
              {next && <PasswordChecklist password={next} email={user?.email} />}
            </div>
            <Field label="Confirm new password" error={confirm && confirm !== next ? "Passwords don't match" : null}>
              <input className="input" type="password" autoComplete="new-password" value={confirm} maxLength={128} onChange={(e) => setConfirm(e.target.value)} />
            </Field>
            {error && <div className="sm:col-span-2"><ErrorNote>{error}</ErrorNote></div>}
          </div>
          <div className="flex justify-end border-t border-line bg-surface-2/40 px-5 py-3 sm:px-6">
            <Button type="submit" loading={saving} disabled={!valid}>Update password</Button>
          </div>
        </Card>
      </form>

      <Card className="flex flex-col gap-4 p-5 sm:flex-row sm:items-center sm:justify-between sm:p-6">
        <div>
          <h2 className="text-[15px] font-semibold">Sign out of all devices</h2>
          <p className="mt-0.5 text-[13px] text-muted">Ends every active session, including this one. API keys keep working.</p>
        </div>
        <Button variant="secondary" onClick={() => setConfirmAll(true)} className="text-high"><LogOut className="h-4 w-4" />Sign out everywhere</Button>
      </Card>

      <ConfirmDialog open={confirmAll} onClose={() => setConfirmAll(false)} onConfirm={logoutAll} loading={signingOut}
        title="Sign out of all devices?" confirmLabel="Sign out everywhere"
        body="You'll need to sign in again on every browser and device, including this one." />
    </section>
  );
}

function Keys() {
  const [keys, setKeys] = useState<ApiKey[] | null>(null);
  const [name, setName] = useState("");
  const [creating, setCreating] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [created, setCreated] = useState<ApiKey | null>(null);
  const [revoke, setRevoke] = useState<ApiKey | null>(null);
  const [revoking, setRevoking] = useState(false);

  useEffect(() => {
    api.keys().then((r) => setKeys(r.keys)).catch((e) => { setError(errorMessage(e, "Couldn't load API keys.")); setKeys([]); });
  }, []);

  const create = async (e: FormEvent) => {
    e.preventDefault();
    if (!name.trim()) return;
    setCreating(true);
    setError(null);
    try {
      const k = await api.createKey(name.trim());
      setKeys((prev) => [{ ...k, key: undefined }, ...(prev ?? [])]);
      setCreated(k);
      setName("");
    } catch (err) {
      setError(errorMessage(err, "Couldn't create the key."));
    } finally {
      setCreating(false);
    }
  };

  const doRevoke = async () => {
    if (!revoke) return;
    setRevoking(true);
    try {
      await api.revokeKey(revoke.id);
      setKeys((prev) => prev?.map((k) => (k.id === revoke.id ? { ...k, revoked: true } : k)) ?? null);
      toast.success(`Revoked "${revoke.name}"`);
      setRevoke(null);
    } catch (err) {
      toast.error(errorMessage(err, "Couldn't revoke the key."));
    } finally {
      setRevoking(false);
    }
  };

  const base = API_BASE || window.location.origin;
  const snippet = `curl -X POST ${base}/api/cases \\
  -H "X-API-Key: $ORALAID_API_KEY" \\
  -F "files=@lesion.jpg" \\
  -F "site=Tongue (lateral)" \\
  -F "patient_reference=MRN-20431"`;

  const activeKeys = keys?.filter((k) => !k.revoked) ?? [];
  const revokedKeys = keys?.filter((k) => k.revoked) ?? [];

  return (
    <Section id="keys" title="API keys" desc="Connect EHR, PACS or research pipelines. Keys act with your account's permissions.">
      <form onSubmit={create} className="flex flex-col gap-2 sm:flex-row">
        <input className="input" value={name} onChange={(e) => setName(e.target.value)} maxLength={60} placeholder="Key name, e.g. Clinic EHR integration" aria-label="New key name" />
        <Button type="submit" loading={creating} disabled={!name.trim()} className="shrink-0"><Plus className="h-4 w-4" />Create key</Button>
      </form>
      {error && <div className="mt-3"><ErrorNote>{error}</ErrorNote></div>}

      <div className="mt-5">
        {keys === null ? (
          <div className="space-y-2">{[0, 1].map((i) => <div key={i} className="skeleton h-14" />)}</div>
        ) : keys.length === 0 ? (
          <div className="rounded-xl border border-dashed border-line px-4 py-8 text-center">
            <KeyRound className="mx-auto h-5 w-5 text-muted" />
            <p className="mt-2 text-sm font-medium">No API keys yet</p>
            <p className="mt-0.5 text-xs text-muted">Create one to send images from another system.</p>
          </div>
        ) : (
          <ul className="divide-y divide-line overflow-hidden rounded-xl border border-line">
            {[...activeKeys, ...revokedKeys].map((k) => (
              <li key={k.id} className={cn("flex flex-col gap-2 p-3.5 sm:flex-row sm:items-center sm:gap-4", k.revoked && "opacity-55")}>
                <div className="flex min-w-0 flex-1 items-center gap-3">
                  <span className="grid h-9 w-9 shrink-0 place-items-center rounded-lg bg-surface-2 text-muted"><KeyRound className="h-4 w-4" /></span>
                  <div className="min-w-0">
                    <div className="flex items-center gap-2">
                      <p className="truncate text-sm font-medium">{k.name}</p>
                      {k.revoked && <Badge>Revoked</Badge>}
                    </div>
                    <p className="font-mono text-xs text-muted">{k.prefix}…</p>
                  </div>
                </div>
                <div className="flex items-center justify-between gap-4 pl-12 text-xs text-muted sm:pl-0">
                  <span title={formatDate(k.created_at)}>Created {timeAgo(k.created_at)}</span>
                  <span>{k.last_used_at ? `Used ${timeAgo(k.last_used_at)}` : "Never used"}</span>
                  {!k.revoked && (
                    <button onClick={() => setRevoke(k)} className="flex items-center gap-1 rounded-md px-2 py-1 font-medium text-high hover:bg-high/10" aria-label={`Revoke ${k.name}`}>
                      <Trash2 className="h-3.5 w-3.5" />Revoke
                    </button>
                  )}
                </div>
              </li>
            ))}
          </ul>
        )}
      </div>

      <div className="mt-6">
        <p className="label">Usage</p>
        <CodeBlock code={snippet} />
        <p className="mt-2 text-xs text-muted">Full reference in the <Link to="/docs" className="font-medium text-brand hover:underline">API documentation</Link>.</p>
      </div>

      <Modal open={!!created} onClose={() => setCreated(null)} title="Your new API key">
        {created?.key && (
          <div className="space-y-4">
            <div className="flex gap-3 rounded-xl border border-moderate/30 bg-moderate/10 p-3 text-sm">
              <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0 text-moderate" />
              <span>Copy this key now. For your security it <strong>won't be shown again</strong>. Store it in a secret manager, never in source code.</span>
            </div>
            <CopyField value={created.key} />
            <div className="flex justify-end"><Button onClick={() => setCreated(null)}>I've saved it</Button></div>
          </div>
        )}
      </Modal>

      <ConfirmDialog open={!!revoke} onClose={() => setRevoke(null)} onConfirm={doRevoke} loading={revoking}
        title={`Revoke "${revoke?.name ?? ""}"?`} confirmLabel="Revoke key"
        body="Any integration using this key will stop working immediately. This can't be undone." />
    </Section>
  );
}

function useCopy() {
  const [copied, setCopied] = useState(false);
  const copy = async (text: string) => {
    try {
      await navigator.clipboard.writeText(text);
      setCopied(true);
      setTimeout(() => setCopied(false), 1600);
    } catch {
      toast.error("Couldn't copy. Select the text and copy it manually.");
    }
  };
  return { copied, copy };
}

function CopyField({ value }: { value: string }) {
  const { copied, copy } = useCopy();
  return (
    <div className="flex items-center gap-2 rounded-xl border border-line bg-surface-2 p-1.5 pl-3">
      <code className="min-w-0 flex-1 select-all break-all font-mono text-[13px]">{value}</code>
      <Button size="sm" variant={copied ? "secondary" : "primary"} onClick={() => copy(value)} className="shrink-0">
        {copied ? <><Check className="h-3.5 w-3.5" />Copied</> : <><Copy className="h-3.5 w-3.5" />Copy</>}
      </Button>
    </div>
  );
}

function CodeBlock({ code }: { code: string }) {
  const { copied, copy } = useCopy();
  return (
    <div className="relative overflow-hidden rounded-xl border border-line bg-surface-2/70">
      <button onClick={() => copy(code)} aria-label="Copy snippet"
        className="absolute right-2 top-2 grid h-8 w-8 place-items-center rounded-lg text-muted transition hover:bg-surface hover:text-ink">
        {copied ? <Check className="h-4 w-4 text-low" /> : <Copy className="h-4 w-4" />}
      </button>
      <pre className="overflow-x-auto p-4 pr-12 font-mono text-[12px] leading-relaxed text-ink/90"><code>{code}</code></pre>
    </div>
  );
}

const ACTIONS: Record<string, string> = {
  "auth.register": "Created account",
  "auth.login": "Signed in",
  "auth.login_failed": "Failed sign-in attempt",
  "auth.password_changed": "Changed password",
  "auth.logout_all": "Signed out of all devices",
  "account.update": "Updated profile",
  "apikey.create": "Created API key",
  "apikey.revoke": "Revoked API key",
  "case.create": "Ran analysis",
  "case.view": "Viewed case",
  "case.update": "Updated case",
  "case.feedback": "Added feedback",
  "case.delete": "Deleted case",
  "case.export": "Exported cases",
};
const humanize = (a: string) => ACTIONS[a] ?? a.replace(/[._]/g, " ").replace(/^./, (c) => c.toUpperCase());

function ActivityLog() {
  const [events, setEvents] = useState<{ action: string; target: string | null; ip: string; created_at: string }[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [showAll, setShowAll] = useState(false);

  useEffect(() => {
    api.audit().then((r) => setEvents(r.events)).catch((e) => { setError(errorMessage(e, "Couldn't load activity.")); setEvents([]); });
  }, []);

  const shown = showAll ? events : events?.slice(0, 10);

  return (
    <Section id="activity" title="Activity" desc="Recent security and data access events on your account."
      footer={events && events.length > 10 ? <Button variant="ghost" size="sm" onClick={() => setShowAll((s) => !s)}>{showAll ? "Show less" : `Show all ${events.length}`}</Button> : undefined}>
      {error && <ErrorNote>{error}</ErrorNote>}
      {shown === null || shown === undefined ? (
        <div className="space-y-3">{[0, 1, 2, 3].map((i) => <div key={i} className="skeleton h-9" />)}</div>
      ) : shown.length === 0 ? (
        <p className="py-4 text-center text-sm text-muted">No activity yet.</p>
      ) : (
        <ol className="relative space-y-0.5">
          {shown.map((e, i) => {
            const failed = e.action.includes("failed");
            const isCase = e.action.startsWith("case.") && e.target && e.action !== "case.delete";
            return (
              <li key={`${e.created_at}-${i}`} className="flex items-center gap-3 rounded-lg px-1 py-2 text-sm">
                <span className={cn("h-2 w-2 shrink-0 rounded-full", failed ? "bg-high" : e.action.startsWith("auth") ? "bg-brand" : "bg-line")} />
                <div className="min-w-0 flex-1">
                  <p className={cn("truncate", failed && "text-high")}>
                    {humanize(e.action)}
                    {e.target && (isCase
                      ? <Link to={`/app/cases/${e.target}`} className="ml-1.5 font-mono text-xs text-muted hover:text-brand">{e.target.slice(0, 10)}</Link>
                      : <span className="ml-1.5 font-mono text-xs text-muted">{e.target.slice(0, 16)}</span>)}
                  </p>
                </div>
                <span className="hidden font-mono text-[11px] text-muted sm:inline">{e.ip}</span>
                <span className="w-20 shrink-0 text-right text-xs text-muted" title={formatDate(e.created_at)}>{timeAgo(e.created_at)}</span>
              </li>
            );
          })}
        </ol>
      )}
    </Section>
  );
}

function Plan() {
  const user = useAuth((s) => s.user);
  const plan = user?.plan ?? "starter";
  return (
    <Section id="plan" title="Plan" desc="Your subscription and usage limits.">
      <div className="flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
        <div className="flex items-center gap-4">
          <div className="grid h-12 w-12 place-items-center rounded-2xl bg-brand/10 text-brand"><CreditCard className="h-5 w-5" /></div>
          <div>
            <div className="flex items-center gap-2">
              <p className="text-base font-semibold capitalize">{plan}</p>
              <Badge tone="brand">Current plan</Badge>
            </div>
            <p className="mt-0.5 text-[13px] text-muted">Member since {user ? new Date(user.created_at.endsWith("Z") ? user.created_at : `${user.created_at}Z`).toLocaleDateString(undefined, { month: "long", year: "numeric" }) : "–"}</p>
          </div>
        </div>
        <Link to="/pricing" className={buttonClass("secondary")}>Compare plans<ArrowUpRight className="h-4 w-4" /></Link>
      </div>
    </Section>
  );
}
