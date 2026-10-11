import { useEffect, useState } from "react";
import { Link, useSearchParams } from "react-router-dom";
import { motion } from "framer-motion";
import { ChevronRight, Plus, Search, Users, X } from "lucide-react";
import { api, type PatientRow } from "@/lib/api";
import { useSeo } from "@/lib/seo";
import { timeAgo } from "@/lib/utils";
import { PageHeader } from "@/components/AppShell";
import { Badge, Button, ButtonLink, EmptyState, ErrorNote } from "@/components/ui";
import { RiskBadge } from "@/components/clinical";
import { errorMessage, patientName, SEX_LABEL } from "@/components/app/kit";

export default function Patients() {
  useSeo("Patients", undefined, { noindex: true });
  const [params, setParams] = useSearchParams();
  const q = params.get("q") ?? "";
  const [search, setSearch] = useState(q);
  const [rows, setRows] = useState<PatientRow[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [reload, setReload] = useState(0);

  useEffect(() => {
    const t = setTimeout(() => {
      if (search.trim() !== q) setParams(search.trim() ? { q: search.trim() } : {}, { replace: true });
    }, 300);
    return () => clearTimeout(t);
  }, [search, q, setParams]);

  useEffect(() => {
    let cancelled = false;
    setRows(null);
    setError(null);
    api.patients(q || undefined)
      .then((r) => { if (!cancelled) setRows(r.patients); })
      .catch((e) => { if (!cancelled) { setError(errorMessage(e, "Couldn't load patients.")); setRows([]); } });
    return () => { cancelled = true; };
  }, [q, reload]);

  return (
    <div>
      <PageHeader title="Patients" subtitle="Everyone you've screened, linked by reference so visits form a timeline."
        actions={<ButtonLink to="/app/new"><Plus className="h-4 w-4" />New analysis</ButtonLink>} />

      <div className="relative mb-5">
        <Search className="pointer-events-none absolute left-3.5 top-1/2 h-4 w-4 -translate-y-1/2 text-muted" />
        <input className="input pl-10 pr-10" type="search" value={search} onChange={(e) => setSearch(e.target.value)} maxLength={80}
          placeholder="Search by name or reference" aria-label="Search patients" />
        {search && (
          <button onClick={() => setSearch("")} aria-label="Clear search"
            className="absolute right-2 top-1/2 grid h-7 w-7 -translate-y-1/2 place-items-center rounded-md text-muted hover:text-ink"><X className="h-4 w-4" /></button>
        )}
      </div>

      {error && <div className="mb-4 flex items-center gap-3"><ErrorNote>{error}</ErrorNote><Button size="sm" variant="secondary" onClick={() => setReload((r) => r + 1)}>Retry</Button></div>}

      {rows === null ? (
        <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-3">
          {Array.from({ length: 6 }).map((_, i) => (
            <div key={i} className="card space-y-3 p-5">
              <div className="flex items-center gap-3"><div className="skeleton h-10 w-10 rounded-full" /><div className="flex-1 space-y-2"><div className="skeleton h-3.5 w-1/2" /><div className="skeleton h-3 w-1/3" /></div></div>
              <div className="skeleton h-5 w-2/3 rounded-full" />
            </div>
          ))}
        </div>
      ) : rows.length === 0 ? (
        q ? (
          <EmptyState icon={<Search className="h-5 w-5" />} title="No patients match" body={`Nothing found for "${q}". Try a reference number or part of a name.`}
            action={<Button variant="secondary" onClick={() => setSearch("")}>Clear search</Button>} />
        ) : (
          <EmptyState icon={<Users className="h-5 w-5" />} title="No patients yet"
            body="Patients are created when you run an analysis. Add a reference or MRN so repeat visits link into one timeline."
            action={<ButtonLink to="/app/new"><Plus className="h-4 w-4" />New analysis</ButtonLink>} />
        )
      ) : (
        <motion.ul initial="hidden" animate="show" variants={{ show: { transition: { staggerChildren: 0.03 } } }}
          className="grid gap-3 sm:grid-cols-2 xl:grid-cols-3">
          {rows.map((p) => {
            const label = patientName(p);
            const initials = label.split(/[\s-]+/).map((s) => s[0]).filter(Boolean).slice(0, 2).join("").toUpperCase();
            return (
              <motion.li key={p.id} variants={{ hidden: { opacity: 0, y: 6 }, show: { opacity: 1, y: 0 } }}>
                <Link to={`/app/patients/${p.id}`} className="card group flex h-full flex-col p-5 transition hover:-translate-y-0.5 hover:shadow-lift">
                  <div className="flex items-start gap-3">
                    <div className="grid h-10 w-10 shrink-0 place-items-center rounded-full bg-brand/10 text-xs font-semibold text-brand">{initials || "?"}</div>
                    <div className="min-w-0 flex-1">
                      <p className="truncate text-sm font-semibold">{label}</p>
                      <p className="truncate text-xs text-muted">
                        {[p.name && p.reference ? p.reference : null, p.age !== null && p.age !== undefined ? `${p.age} y` : null, p.sex ? SEX_LABEL[p.sex] ?? p.sex : null]
                          .filter(Boolean).join(" · ") || "No demographics"}
                      </p>
                    </div>
                    {p.latest_tier && <RiskBadge tier={p.latest_tier} size="sm" />}
                  </div>
                  {!!p.risk_factors?.length && (
                    <div className="mt-3 flex flex-wrap gap-1">
                      {p.risk_factors.slice(0, 3).map((f) => <Badge key={f}>{f}</Badge>)}
                      {p.risk_factors.length > 3 && <Badge>+{p.risk_factors.length - 3}</Badge>}
                    </div>
                  )}
                  <div className="mt-auto flex items-center justify-between pt-4 text-xs text-muted">
                    <span className="tabular-nums">{p.case_count} visit{p.case_count === 1 ? "" : "s"}{p.latest_at ? ` · last ${timeAgo(p.latest_at)}` : ""}</span>
                    <ChevronRight className="h-4 w-4 transition group-hover:translate-x-0.5 group-hover:text-ink" />
                  </div>
                </Link>
              </motion.li>
            );
          })}
        </motion.ul>
      )}
    </div>
  );
}
