import { useEffect, useState, type FormEvent } from "react";
import { Link, NavLink, Outlet, useLocation } from "react-router-dom";
import { AnimatePresence, motion } from "framer-motion";
import { ArrowRight, Menu, Moon, Sun, X } from "lucide-react";
import { api } from "@/lib/api";
import { useAuth } from "@/lib/auth";
import { useBackend } from "@/lib/backend";
import { useTheme } from "@/lib/theme";
import { cn } from "@/lib/utils";
import { Logo } from "./Logo";
import { ButtonLink } from "./ui";
import { toast } from "./Toast";

const NAV = [
  { to: "/screen", label: "Screen free" },
  { to: "/platform", label: "Platform" },
  { to: "/models", label: "Models" },
  { to: "/deployment", label: "Deployment" },
  { to: "/pricing", label: "Pricing" },
  { to: "/roadmap", label: "Roadmap" },
  { to: "/docs", label: "Docs" },
];

export function ThemeButton({ className }: { className?: string }) {
  const { dark, toggle } = useTheme();
  return (
    <button onClick={toggle} className={cn("grid h-9 w-9 place-items-center rounded-lg text-muted transition hover:bg-surface-2 hover:text-ink", className)}
      aria-label={dark ? "Switch to light theme" : "Switch to dark theme"}>
      {dark ? <Sun className="h-4 w-4" /> : <Moon className="h-4 w-4" />}
    </button>
  );
}

function Header() {
  const [open, setOpen] = useState(false);
  const [scrolled, setScrolled] = useState(false);
  const user = useAuth((s) => s.user);
  const backend = useBackend();
  const { pathname } = useLocation();

  useEffect(() => setOpen(false), [pathname]);
  useEffect(() => {
    const on = () => setScrolled(window.scrollY > 8);
    on();
    window.addEventListener("scroll", on, { passive: true });
    return () => window.removeEventListener("scroll", on);
  }, []);

  return (
    <header className={cn("sticky top-0 z-50 transition-all", scrolled ? "border-b border-line bg-bg/80 backdrop-blur-xl" : "border-b border-transparent")}>
      <div className="container flex h-16 items-center gap-6">
        <Link to="/" aria-label="OralAID home"><Logo /></Link>
        <nav className="hidden items-center gap-1 lg:flex">
          {NAV.map((n) => (
            <NavLink key={n.to} to={n.to}
              className={({ isActive }) => cn("rounded-lg px-3 py-1.5 text-sm transition", isActive ? "text-ink" : "text-muted hover:text-ink")}>
              {n.label}
            </NavLink>
          ))}
        </nav>
        <div className="ml-auto flex items-center gap-2">
          <ThemeButton className="hidden sm:grid" />
          {backend !== "online" ? (
            <ButtonLink to="/screen" size="sm" variant="ink" className="hidden sm:inline-flex">Start screening <ArrowRight className="h-3.5 w-3.5" /></ButtonLink>
          ) : user ? (
            <ButtonLink to="/app" size="sm" variant="ink">Open app <ArrowRight className="h-3.5 w-3.5" /></ButtonLink>
          ) : (
            <>
              <Link to="/login" className="hidden px-3 text-sm text-muted transition hover:text-ink sm:block">Sign in</Link>
              <ButtonLink to="/register" size="sm" variant="ink" className="hidden sm:inline-flex">Start free</ButtonLink>
            </>
          )}
          <button className="grid h-9 w-9 place-items-center rounded-lg hover:bg-surface-2 lg:hidden" onClick={() => setOpen((o) => !o)} aria-label="Menu" aria-expanded={open}>
            {open ? <X className="h-5 w-5" /> : <Menu className="h-5 w-5" />}
          </button>
        </div>
      </div>
      <AnimatePresence>
        {open && (
          <motion.nav initial={{ height: 0, opacity: 0 }} animate={{ height: "auto", opacity: 1 }} exit={{ height: 0, opacity: 0 }}
            className="overflow-hidden border-t border-line bg-bg lg:hidden">
            <div className="container flex flex-col py-3">
              {NAV.map((n) => <Link key={n.to} to={n.to} className="rounded-lg px-2 py-3 text-[15px]">{n.label}</Link>)}
              <div className="mt-3 flex gap-2 border-t border-line pt-4">
                {backend !== "online" ? (
                  <ButtonLink to="/screen" variant="ink" className="flex-1">Start screening</ButtonLink>
                ) : (
                  <>
                    <ButtonLink to={user ? "/app" : "/register"} variant="ink" className="flex-1">{user ? "Open app" : "Start free"}</ButtonLink>
                    {!user && <ButtonLink to="/login" variant="secondary" className="flex-1">Sign in</ButtonLink>}
                  </>
                )}
                <ThemeButton />
              </div>
            </div>
          </motion.nav>
        )}
      </AnimatePresence>
    </header>
  );
}

function Newsletter() {
  const [email, setEmail] = useState("");
  const [busy, setBusy] = useState(false);
  const submit = async (e: FormEvent) => {
    e.preventDefault();
    setBusy(true);
    try {
      await api.lead({ kind: "newsletter", email });
      toast.success("You're on the list. One email a month, no spam.");
      setEmail("");
    } catch (err) {
      toast.error((err as Error).message);
    } finally {
      setBusy(false);
    }
  };
  return (
    <form onSubmit={submit} className="flex max-w-sm gap-2">
      <input type="email" required value={email} onChange={(e) => setEmail(e.target.value)} placeholder="you@clinic.com" className="input" aria-label="Email" />
      <button disabled={busy} className="shrink-0 rounded-xl bg-ink px-4 text-sm font-medium text-bg transition hover:bg-ink/90 disabled:opacity-50">Subscribe</button>
    </form>
  );
}

const FOOTER = [
  { title: "Product", links: [["Platform", "/platform"], ["Models", "/models"], ["Pricing", "/pricing"], ["Roadmap", "/roadmap"], ["Desktop app", "/roadmap#desktop"]] },
  { title: "Deploy", links: [["Cloud", "/deployment#cloud"], ["Private VPC", "/deployment#vpc"], ["On-premise", "/deployment#onprem"], ["Security", "/security"]] },
  { title: "Developers", links: [["API docs", "/docs"], ["Quickstart", "/docs#quickstart"], ["Status", "/docs#status"]] },
  { title: "Company", links: [["About", "/about"], ["Contact", "/contact"], ["Privacy", "/privacy"], ["Terms", "/terms"]] },
];

function Footer() {
  return (
    <footer className="mt-24 border-t border-line">
      <div className="container grid gap-12 py-16 lg:grid-cols-[1.3fr_2fr]">
        <div className="space-y-5">
          <Logo />
          <p className="max-w-sm text-sm text-muted">Explainable AI that helps clinicians catch oral cancer earlier, wherever they practise.</p>
          <Newsletter />
        </div>
        <div className="grid grid-cols-2 gap-8 sm:grid-cols-4">
          {FOOTER.map((col) => (
            <div key={col.title}>
              <p className="eyebrow mb-4">{col.title}</p>
              <ul className="space-y-2.5">
                {col.links.map(([label, to]) => (
                  <li key={to}><Link to={to} className="text-sm text-muted transition hover:text-ink">{label}</Link></li>
                ))}
              </ul>
            </div>
          ))}
        </div>
      </div>
      <div className="border-t border-line">
        <div className="container flex flex-col gap-3 py-6 text-xs text-muted sm:flex-row sm:items-center sm:justify-between">
          <p>&copy; {new Date().getFullYear()} OralAID. Clinical decision support only. Not a substitute for examination or biopsy.</p>
          <p>Research use. Not cleared as a medical device in any jurisdiction.</p>
        </div>
      </div>
    </footer>
  );
}

export default function SiteLayout() {
  const { pathname, hash } = useLocation();
  useEffect(() => {
    if (hash) document.getElementById(hash.slice(1))?.scrollIntoView({ behavior: "smooth" });
    else window.scrollTo(0, 0);
  }, [pathname, hash]);
  return (
    <div className="flex min-h-screen flex-col">
      <a href="#main" className="sr-only focus:not-sr-only focus:fixed focus:left-4 focus:top-4 focus:z-[100] focus:rounded-lg focus:bg-surface focus:px-4 focus:py-2">Skip to content</a>
      <Header />
      <main id="main" className="flex-1"><Outlet /></main>
      <Footer />
    </div>
  );
}
