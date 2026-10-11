import { Link } from "react-router-dom";
import { ChevronRight } from "lucide-react";
import { fileUrl, type CaseSummary } from "@/lib/api";
import { timeAgo } from "@/lib/utils";
import { RiskBadge } from "@/components/clinical";
import { patientMeta, patientName, Thumb } from "./kit";

/** Compact clickable case row used in dashboard lists and timelines. */
export function CaseItem({ c, meta }: { c: CaseSummary; meta?: string }) {
  return (
    <Link to={`/app/cases/${c.id}`}
      className="group flex items-center gap-3 rounded-xl px-2 py-2 transition hover:bg-surface-2/70 focus-visible:bg-surface-2/70">
      <Thumb src={fileUrl(c.thumbnail)} alt="" className="h-11 w-11" />
      <div className="min-w-0 flex-1">
        <div className="flex items-center gap-2">
          <p className="truncate text-sm font-medium">{patientName(c.patient)}</p>
          <span className="hidden font-mono text-[11px] text-muted sm:inline">{c.id.slice(0, 8)}</span>
        </div>
        <p className="truncate text-xs text-muted">
          {meta ?? [c.site, patientMeta(c.patient), timeAgo(c.created_at)].filter(Boolean).join(" · ")}
        </p>
      </div>
      <div className="flex shrink-0 items-center gap-2">
        <span className="hidden text-sm tabular-nums text-muted sm:inline">{Math.round(c.probability * 100)}%</span>
        <RiskBadge tier={c.tier} size="sm" />
        <ChevronRight className="h-4 w-4 text-muted opacity-0 transition group-hover:opacity-100" />
      </div>
    </Link>
  );
}

export function CaseItemSkeleton() {
  return (
    <div className="flex items-center gap-3 px-2 py-2">
      <div className="skeleton h-11 w-11" />
      <div className="flex-1 space-y-2"><div className="skeleton h-3.5 w-1/3" /><div className="skeleton h-3 w-1/2" /></div>
      <div className="skeleton h-5 w-14 rounded-full" />
    </div>
  );
}

/** Shared recharts tooltip shell so every chart uses the same surface. */
export function ChartTooltip({ title, rows }: { title: string; rows: { label: string; value: string | number; color?: string }[] }) {
  return (
    <div className="min-w-[140px] rounded-xl border border-line bg-surface px-3 py-2 text-xs shadow-lift">
      <p className="mb-1.5 font-medium text-ink">{title}</p>
      {rows.map((r) => (
        <div key={r.label} className="flex items-center justify-between gap-4 py-0.5">
          <span className="flex items-center gap-1.5 text-muted">
            {r.color && <span className="h-2 w-2 rounded-sm" style={{ background: r.color }} />}{r.label}
          </span>
          <span className="font-medium tabular-nums text-ink">{r.value}</span>
        </div>
      ))}
    </div>
  );
}
