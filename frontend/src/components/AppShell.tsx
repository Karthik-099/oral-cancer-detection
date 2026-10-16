import { useEffect, useMemo, useRef, useState } from "react";
import { Link, NavLink, Navigate, Outlet, useLocation, useNavigate } from "react-router-dom";
import { AnimatePresence, motion } from "framer-motion";
import {
  BookOpen, Command, FileStack, Home, KeyRound, LayoutDashboard, LogOut, Menu, Plus, Search, Settings, Users, X, Cpu,
} from "lucide-react";
import { useAuth } from "@/lib/auth";
import { cn } from "@/lib/utils";
import { Logo } from "./Logo";
import { Kbd, Spinner, buttonClass } from "./ui";

const NAV = [
  { to: "/app", label: "Overview", icon: LayoutDashboard, end: true },
  { to: "/app/cases", label: "Cases", icon: FileStack },
  { to: "/app/patients", label: "Patients", icon: Users },
  { to: "/app/settings", label: "Settings", icon: Settings },
];

export function RequireAuth() {
  const { user, status, bootstrap } = useAuth();
  const location = useLocation();
  useEffect(() => { bootstrap(); }, [bootstrap]);
  if (status !== "ready") return <div className="grid min-h-screen place-items-center"><Spinner /></div>;
  if (!user) return <Navigate to="/login" replace state={{ from: location.pathname }} />;
  return <AppShell />;
}

function CommandPalette({ open, onClose }: { open: boolean; onClose: () => void }) {
  const navigate = useNavigate();
  const signOut = useAuth((s) => s.signOut);
  const [q, setQ] = useState("");
  const [active, setActive] = useState(0);
  const inputRef = useRef<HTMLInputElement>(null);

  const commands = useMemo(() => [
    { label: "New analysis", hint: "N", icon: Plus, run: () => navigate("/app/new") },
    { label: "Overview", icon: LayoutDashboard, run: () => navigate("/app") },
    { label: "All cases", icon: FileStack, run: () => navigate("/app/cases") },
    { label: "High-risk cases", icon: FileStack, run: () => navigate("/app/cases?tier=high") },
    { label: "Awaiting review", icon: FileStack, run: () => navigate("/app/cases?review=1") },
    { label: "Patients", icon: Users, run: () => navigate("/app/patients") },
    { label: "API keys", icon: KeyRound, run: () => navigate("/app/settings#keys") },
    { label: "Settings", icon: Settings, run: () => navigate("/app/settings") },
    { label: "Model cards", icon: Cpu, run: () => navigate("/models") },
    { label: "API documentation", icon: BookOpen, run: () => navigate("/docs") },
    { label: "Website", icon: Home, run: () => navigate("/") },
    { label: "Sign out", icon: LogOut, run: () => { signOut(); navigate("/"); } },
  ], [navigate, signOut]);

  const looksLikeSearch = q.trim().length > 1;
  const filtered = commands.filter((c) => c.label.toLowerCase().includes(q.toLowerCase()));
  const items = looksLikeSearch
    ? [...filtered, { label: `Search cases for "${q.trim()}"`, icon: Search, run: () => navigate(`/app/cases?q=${encodeURIComponent(q.trim())}`) }]
    : filtered;

  useEffect(() => { if (open) { setQ(""); setActive(0); setTimeout(() => inputRef.current?.focus(), 10); } }, [open]);
  useEffect(() => setActive(0), [q]);

  const run = (i: number) => { items[i]?.run(); onClose(); };

  return (
    <AnimatePresence>
      {open && (
        <motion.div className="fixed inset-0 z-[95] flex items-start justify-center px-4 pt-[14vh]" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}>
          <div className="absolute inset-0 bg-black/60 backdrop-blur-sm" onClick={onClose} />
          <motion.div initial={{ scale: 0.97, y: -8 }} animate={{ scale: 1, y: 0 }} exit={{ scale: 0.98 }}
            className="relative w-full max-w-lg overflow-hidden rounded-2xl border border-line bg-surface shadow-lift" role="dialog" aria-label="Command palette">
            <div className="flex items-center gap-3 border-b border-line px-4">
              <Search className="h-4 w-4 text-muted" />
              <input ref={inputRef} value={q} onChange={(e) => setQ(e.target.value)} placeholder="Jump to, or search patients and case IDs..."
                className="h-12 flex-1 bg-transparent text-sm outline-none placeholder:text-muted"
                onKeyDown={(e) => {
                  if (e.key === "ArrowDown") { e.preventDefault(); setActive((a) => Math.min(items.length - 1, a + 1)); }
                  if (e.key === "ArrowUp") { e.preventDefault(); setActive((a) => Math.max(0, a - 1)); }
                  if (e.key === "Enter") run(active);
                  if (e.key === "Escape") onClose();
                }} />
              <Kbd>esc</Kbd>
            </div>
            <ul className="max-h-80 overflow-auto p-2">
              {items.map((c, i) => (
                <li key={c.label}>
                  <button onMouseEnter={() => setActive(i)} onClick={() => run(i)}
                    className={cn("flex w-full items-center gap-3 rounded-lg px-3 py-2.5 text-left text-sm", i === active ? "bg-surface-2 text-ink" : "text-muted")}>
                    <c.icon className="h-4 w-4" /><span className="flex-1">{c.label}</span>
                  </button>
                </li>
              ))}
              {!items.length && <li className="px-3 py-6 text-center text-sm text-muted">No matches</li>}
            </ul>
          </motion.div>
        </motion.div>
      )}
    </AnimatePresence>
  );
}

function AppShell() {
  const { user, signOut } = useAuth();
  const navigate = useNavigate();
  const { pathname } = useLocation();
  const [palette, setPalette] = useState(false);
  const [mobile, setMobile] = useState(false);

  useEffect(() => setMobile(false), [pathname]);
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const typing = /input|textarea|select/i.test((e.target as HTMLElement).tagName);
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "k") { e.preventDefault(); setPalette((p) => !p); }
      else if (!typing && e.key.toLowerCase() === "n" && !e.metaKey && !e.ctrlKey) navigate("/app/new");
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [navigate]);

  const initials = (user?.full_name || user?.email || "?").split(/\s+/).map((p) => p[0]).slice(0, 2).join("").toUpperCase();

  const sidebar = (
    <div className="flex h-full flex-col gap-1 p-3">
      <Link to="/app" className="mb-4 px-2 pt-2"><Logo /></Link>
      <Link to="/app/new" className={buttonClass("primary", "md", "mb-3 w-full justify-start")}><Plus className="h-4 w-4" />New analysis<span className="ml-auto opacity-70"><Kbd>N</Kbd></span></Link>
      <button onClick={() => setPalette(true)} className="mb-3 flex items-center gap-2 rounded-xl border border-line bg-surface px-3 py-2 text-sm text-muted transition hover:text-ink">
        <Search className="h-4 w-4" />Search<span className="ml-auto flex items-center gap-0.5"><Kbd><Command className="inline h-3 w-3" /></Kbd><Kbd>K</Kbd></span>
      </button>
      {NAV.map((n) => (
        <NavLink key={n.to} to={n.to} end={n.end}
          className={({ isActive }) => cn("flex items-center gap-3 rounded-xl px-3 py-2 text-sm transition", isActive ? "bg-surface-2 font-medium text-ink" : "text-muted hover:bg-surface-2/60 hover:text-ink")}>
          <n.icon className="h-4 w-4" />{n.label}
        </NavLink>
      ))}
      <div className="mt-auto space-y-3">
        <div className="rounded-xl border border-line bg-surface-2/50 p-3 text-xs text-muted">
          Results support, and never replace, clinical judgement. Refer anything that worries you.
        </div>
        <div className="flex items-center gap-3 rounded-xl px-2 py-2">
          <div className="grid h-9 w-9 shrink-0 place-items-center rounded-full bg-brand/15 text-xs font-semibold text-brand">{initials}</div>
          <div className="min-w-0 flex-1">
            <p className="truncate text-sm font-medium">{user?.full_name}</p>
            <p className="truncate text-xs text-muted">{user?.organization || user?.email}</p>
          </div>
          <button onClick={() => { signOut(); navigate("/"); }} className="grid h-9 w-9 place-items-center rounded-lg text-muted hover:bg-surface-2 hover:text-ink" aria-label="Sign out">
            <LogOut className="h-4 w-4" />
          </button>
        </div>
      </div>
    </div>
  );

  return (
    <div className="min-h-screen lg:grid lg:grid-cols-[260px_1fr]">
      <aside className="sticky top-0 hidden h-screen border-r border-line bg-bg lg:block">{sidebar}</aside>
      <div className="sticky top-0 z-40 flex h-14 items-center gap-3 border-b border-line bg-bg/85 px-4 backdrop-blur-xl lg:hidden">
        <button onClick={() => setMobile(true)} aria-label="Open menu" className="grid h-9 w-9 place-items-center rounded-lg hover:bg-surface-2"><Menu className="h-5 w-5" /></button>
        <Logo />
        <button onClick={() => setPalette(true)} className="ml-auto grid h-9 w-9 place-items-center rounded-lg hover:bg-surface-2" aria-label="Search"><Search className="h-4 w-4" /></button>
        <Link to="/app/new" className={buttonClass("primary", "sm")}><Plus className="h-4 w-4" />New</Link>
      </div>
      <AnimatePresence>
        {mobile && (
          <motion.div className="fixed inset-0 z-50 lg:hidden" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}>
            <div className="absolute inset-0 bg-black/60" onClick={() => setMobile(false)} />
            <motion.aside initial={{ x: -280 }} animate={{ x: 0 }} exit={{ x: -280 }} transition={{ type: "spring", damping: 30, stiffness: 300 }}
              className="absolute inset-y-0 left-0 w-[280px] border-r border-line bg-bg">
              <button onClick={() => setMobile(false)} className="absolute right-3 top-4 rounded-lg p-1 text-muted" aria-label="Close menu"><X className="h-5 w-5" /></button>
              {sidebar}
            </motion.aside>
          </motion.div>
        )}
      </AnimatePresence>
      <main className="min-w-0">
        <motion.div key={pathname} initial={{ opacity: 0, y: 6 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.3 }}
          className="mx-auto max-w-[1200px] px-4 py-6 sm:px-6 lg:px-10 lg:py-10">
          <Outlet />
        </motion.div>
      </main>
      <CommandPalette open={palette} onClose={() => setPalette(false)} />
    </div>
  );
}

export function PageHeader({ title, subtitle, actions }: { title: string; subtitle?: string; actions?: React.ReactNode }) {
  return (
    <div className="mb-8 flex flex-col gap-4 sm:flex-row sm:items-end sm:justify-between">
      <div>
        <h1 className="text-2xl font-semibold tracking-tight sm:text-[28px]">{title}</h1>
        {subtitle && <p className="mt-1 text-sm text-muted">{subtitle}</p>}
      </div>
      {actions && <div className="flex flex-wrap gap-2">{actions}</div>}
    </div>
  );
}
