import { useEffect, useMemo, useState } from "react";
import { useNavigate, useSearchParams } from "react-router-dom";
import { motion } from "framer-motion";
import { ChevronLeft, ChevronRight, CircleDashed, Download, FileStack, Plus, Search, X } from "lucide-react";
import { api, fileUrl, type CaseSummary } from "@/lib/api";
import { useSeo } from "@/lib/seo";
import { cn, downloadBlob, formatDate, timeAgo } from "@/lib/utils";
import { PageHeader } from "@/components/AppShell";
import { Button, ButtonLink, EmptyState, ErrorNote } from "@/components/ui";
import { RiskBadge } from "@/components/clinical";
import { toast } from "@/components/Toast";
import { errorMessage, FeedbackStatus, patientMeta, patientName, ProbabilityBar, Thumb } from "@/components/app/kit";

const LIMIT = 25;
const TIERS = [
  { value: "", label: "All" },
  { value: "high", label: "High", dot: "bg-high" },
  { value: "moderate", label: "Moderate", dot: "bg-moderate" },
  { value: "low", label: "Low", dot: "bg-low" },
];
const STATUSES = [
  { value: "", label: "Any status" },
  { value: "open", label: "Open" },
  { value: "referred", label: "Referred" },
  { value: "closed", label: "Closed" },
];

export default function Cases() {
  useSeo("Cases", undefined, { noindex: true });
  const navigate = useNavigate();
  const [params, setParams] = useSearchParams();
  const q = params.get("q") ?? "";
  const tier = ["high", "moderate", "low"].includes(params.get("tier") ?? "") ? params.get("tier")! : "";
  const status = ["open", "referred", "closed"].includes(params.get("status") ?? "") ? params.get("status")! : "";
  const review = params.get("review") === "1";
  const page = Math.max(0, Number(params.get("page") ?? 0) || 0);

  const [search, setSearch] = useState(q);
  const [rows, setRows] = useState<CaseSummary[] | null>(null);
  const [total, setTotal] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const [exporting, setExporting] = useState(false);
  const [reload, setReload] = useState(0);

  const update = (patch: Record<string, string | null>, resetPage = true) => {
    setParams((prev) => {
      const next = new URLSearchParams(prev);
      for (const [k, v] of Object.entries(patch)) v ? next.set(k, v) : next.delete(k);
      if (resetPage) next.delete("page");
      return next;
    }, { replace: true });
  };

  // Keep the input in sync when the URL changes elsewhere (for example the command palette).
  useEffect(() => { setSearch(q); }, [q]);
  useEffect(() => {
    if (search.trim() === q) return;
    const t = setTimeout(() => update({ q: search.trim() || null }), 300);
    return () => clearTimeout(t);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [search]);

  useEffect(() => {
    let cancelled = false;
    setError(null);
    setRows(null);
    (async () => {
      try {
        const res = await api.cases({
          tier: tier || undefined, status: status || undefined, q: q || undefined,
          review: review || undefined, limit: LIMIT, offset: page * LIMIT,
        });
        if (!cancelled) { setTotal(res.total); setRows(res.cases); }
      } catch (e) {
        if (!cancelled) { setError(errorMessage(e, "Couldn't load cases.")); setRows([]); }
      }
    })();
    return () => { cancelled = true; };
  }, [q, tier, status, review, page, reload]);

  const exportCsv = async () => {
    setExporting(true);
    try {
      downloadBlob(await api.exportCsv({ tier: tier || undefined, status: status || undefined, q: q || undefined, review: review || undefined }), `oralaid-cases-${new Date().toISOString().slice(0, 10)}.csv`);
      toast.success("CSV exported");
    } catch (e) {
      toast.error(errorMessage(e, "Export failed."));
    } finally {
      setExporting(false);
    }
  };

  const filtered = !!(q || tier || status || review);
  const pages = Math.max(1, Math.ceil(total / LIMIT));
  const range = useMemo(() => (total ? `${page * LIMIT + 1}–${Math.min(total, page * LIMIT + LIMIT)} of ${total}` : "0 cases"), [page, total]);

  return (
    <div>
      <PageHeader
        title="Cases"
        subtitle="Every analysis you've run, newest first."
        actions={<>
          <Button variant="secondary" onClick={exportCsv} loading={exporting}>{!exporting && <Download className="h-4 w-4" />}Export CSV</Button>
          <ButtonLink to="/app/new"><Plus className="h-4 w-4" />New analysis</ButtonLink>
        </>}
      />

      <div className="mb-5 space-y-3">
        <div className="relative">
          <Search className="pointer-events-none absolute left-3.5 top-1/2 h-4 w-4 -translate-y-1/2 text-muted" />
          <input className="input pl-10 pr-10" value={search} onChange={(e) => setSearch(e.target.value)} maxLength={80}
            placeholder="Search by patient name, reference or case ID" aria-label="Search cases" type="search" />
          {search && (
            <button onClick={() => { setSearch(""); update({ q: null }); }} aria-label="Clear search"
              className="absolute right-2 top-1/2 grid h-7 w-7 -translate-y-1/2 place-items-center rounded-md text-muted hover:text-ink"><X className="h-4 w-4" /></button>
          )}
        </div>
        <div className="flex flex-wrap items-center gap-2">
          <div className="flex flex-wrap gap-1.5" role="group" aria-label="Risk tier">
            {TIERS.map((t) => (
              <button key={t.label} onClick={() => update({ tier: t.value || null })} aria-pressed={tier === t.value}
                className={cn("inline-flex h-8 items-center gap-1.5 rounded-full border px-3 text-[13px] font-medium transition",
                  tier === t.value ? "border-ink bg-ink text-bg" : "border-line bg-surface text-muted hover:text-ink")}>
                {t.dot && <span className={cn("h-1.5 w-1.5 rounded-full", t.dot)} />}{t.label}
              </button>
            ))}
          </div>
          <span className="mx-1 hidden h-5 w-px bg-line sm:block" />
          <select value={status} onChange={(e) => update({ status: e.target.value || null })} aria-label="Status"
            className="h-8 rounded-full border border-line bg-surface px-3 text-[13px] font-medium text-ink focus:border-brand focus:outline-none">
            {STATUSES.map((s) => <option key={s.label} value={s.value}>{s.label}</option>)}
          </select>
          <button onClick={() => update({ review: review ? null : "1" })} aria-pressed={review}
            className={cn("inline-flex h-8 items-center gap-1.5 rounded-full border px-3 text-[13px] font-medium transition",
              review ? "border-accent/40 bg-accent/10 text-accent" : "border-line bg-surface text-muted hover:text-ink")}>
            <CircleDashed className="h-3.5 w-3.5" />Awaiting review
          </button>
          {filtered && (
            <button onClick={() => { setSearch(""); setParams(new URLSearchParams(), { replace: true }); }} className="ml-auto text-[13px] text-muted hover:text-ink">
              Reset filters
            </button>
          )}
        </div>
      </div>

      {error && <div className="mb-4 flex items-center gap-3"><ErrorNote>{error}</ErrorNote><Button variant="secondary" size="sm" onClick={() => setReload((r) => r + 1)}>Retry</Button></div>}

      {rows === null ? <TableSkeleton /> : rows.length === 0 ? (
        filtered ? (
          <EmptyState icon={<Search className="h-5 w-5" />} title="No matching cases"
            body={review ? "Nothing is waiting for review with these filters. Nice work." : "Try a different search or clear the filters."}
            action={<Button variant="secondary" onClick={() => { setSearch(""); setParams(new URLSearchParams(), { replace: true }); }}>Clear filters</Button>} />
        ) : (
          <EmptyState icon={<FileStack className="h-5 w-5" />} title="No cases yet"
            body="Run an analysis and it will appear here with its risk tier, score and review status."
            action={<ButtonLink to="/app/new"><Plus className="h-4 w-4" />New analysis</ButtonLink>} />
        )
      ) : (
        <motion.div key={`${page}-${tier}-${status}-${review}-${q}`} initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ duration: 0.2 }}>
          {/* Desktop table */}
          <div className="card hidden overflow-hidden md:block">
            <table className="w-full text-left text-sm">
              <thead className="border-b border-line bg-surface-2/50 text-xs text-muted">
                <tr>
                  <th className="py-2.5 pl-4 pr-2 font-medium">Case</th>
                  <th className="px-2 py-2.5 font-medium">Patient</th>
                  <th className="px-2 py-2.5 font-medium">Site</th>
                  <th className="px-2 py-2.5 font-medium">Tier</th>
                  <th className="px-2 py-2.5 font-medium">Score</th>
                  <th className="px-2 py-2.5 font-medium">Review</th>
                  <th className="py-2.5 pl-2 pr-4 text-right font-medium">Date</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-line">
                {rows.map((c) => (
                  <tr key={c.id} onClick={() => navigate(`/app/cases/${c.id}`)}
                    onKeyDown={(e) => e.key === "Enter" && navigate(`/app/cases/${c.id}`)} tabIndex={0}
                    className="cursor-pointer transition hover:bg-surface-2/60 focus-visible:bg-surface-2/60 focus-visible:outline-none">
                    <td className="py-2.5 pl-4 pr-2">
                      <div className="flex items-center gap-3">
                        <Thumb src={fileUrl(c.thumbnail)} className="h-10 w-10" />
                        <div>
                          <span className="font-mono text-xs">{c.id.slice(0, 10)}</span>
                          {c.image_count > 1 && <p className="text-[11px] text-muted">{c.image_count} images</p>}
                        </div>
                      </div>
                    </td>
                    <td className="max-w-[200px] px-2 py-2.5">
                      <p className="truncate font-medium">{patientName(c.patient)}</p>
                      <p className="truncate text-xs text-muted">{patientMeta(c.patient) || "—"}</p>
                    </td>
                    <td className="px-2 py-2.5 text-muted">{c.site || "—"}</td>
                    <td className="px-2 py-2.5"><RiskBadge tier={c.tier} size="sm" /></td>
                    <td className="px-2 py-2.5"><ProbabilityBar value={c.probability} /></td>
                    <td className="px-2 py-2.5"><FeedbackStatus feedback={c.feedback} tier={c.tier} /></td>
                    <td className="whitespace-nowrap py-2.5 pl-2 pr-4 text-right text-xs text-muted" title={formatDate(c.created_at)}>{timeAgo(c.created_at)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          {/* Mobile cards */}
          <ul className="space-y-2 md:hidden">
            {rows.map((c) => (
              <li key={c.id}>
                <button onClick={() => navigate(`/app/cases/${c.id}`)} className="card flex w-full items-center gap-3 p-3 text-left active:scale-[.99]">
                  <Thumb src={fileUrl(c.thumbnail)} className="h-14 w-14" />
                  <div className="min-w-0 flex-1">
                    <div className="flex items-center justify-between gap-2">
                      <p className="truncate text-sm font-medium">{patientName(c.patient)}</p>
                      <RiskBadge tier={c.tier} size="sm" />
                    </div>
                    <p className="mt-0.5 truncate text-xs text-muted">{[c.site, timeAgo(c.created_at)].filter(Boolean).join(" · ")}</p>
                    <div className="mt-1.5 flex items-center justify-between gap-2">
                      <ProbabilityBar value={c.probability} />
                      <FeedbackStatus feedback={c.feedback} tier={c.tier} compact />
                    </div>
                  </div>
                </button>
              </li>
            ))}
          </ul>

          <div className="mt-4 flex items-center justify-between gap-3 text-sm">
            <span className="text-xs tabular-nums text-muted">{range}</span>
            {pages > 1 && (
              <div className="flex items-center gap-1">
                <Button variant="secondary" size="sm" disabled={page === 0} onClick={() => update({ page: page - 1 ? String(page - 1) : null }, false)} aria-label="Previous page">
                  <ChevronLeft className="h-4 w-4" />
                </Button>
                <span className="px-2 text-xs tabular-nums text-muted">Page {page + 1} of {pages}</span>
                <Button variant="secondary" size="sm" disabled={page + 1 >= pages} onClick={() => update({ page: String(page + 1) }, false)} aria-label="Next page">
                  <ChevronRight className="h-4 w-4" />
                </Button>
              </div>
            )}
          </div>
        </motion.div>
      )}
    </div>
  );
}

function TableSkeleton() {
  return (
    <div className="card divide-y divide-line overflow-hidden">
      {Array.from({ length: 6 }).map((_, i) => (
        <div key={i} className="flex items-center gap-4 p-3 sm:px-4">
          <div className="skeleton h-10 w-10" />
          <div className="flex-1 space-y-2"><div className="skeleton h-3.5 w-40 max-w-full" /><div className="skeleton h-3 w-24" /></div>
          <div className="skeleton hidden h-5 w-16 rounded-full sm:block" />
          <div className="skeleton hidden h-3 w-20 md:block" />
        </div>
      ))}
    </div>
  );
}
