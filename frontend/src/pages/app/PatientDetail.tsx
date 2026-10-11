import { useEffect, useState } from "react";
import { Link, useParams } from "react-router-dom";
import { ArrowDownRight, ArrowLeft, ArrowRight, ArrowUpRight, Minus, Plus, UserRound } from "lucide-react";
import { CartesianGrid, Line, LineChart, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { api, ApiError, type CaseSummary, type PatientRef } from "@/lib/api";
import { useSeo } from "@/lib/seo";
import { cn, formatDate } from "@/lib/utils";
import { ButtonLink, Card, EmptyState, ErrorNote } from "@/components/ui";
import { RiskBadge } from "@/components/clinical";
import { CaseItem, CaseItemSkeleton, ChartTooltip } from "@/components/app/CaseItem";
import { errorMessage, HIGH_THRESHOLD, patientMeta, patientName, SectionTitle, THRESHOLD, TIER_RGB } from "@/components/app/kit";

interface Timeline { patient: PatientRef; cases: CaseSummary[]; latest_change: number | null }

const shortDate = (iso: string) => new Date(iso.endsWith("Z") ? iso : `${iso}Z`).toLocaleDateString(undefined, { month: "short", day: "numeric" });

export default function PatientDetail() {
  const { id } = useParams();
  const pid = Number(id);
  const [data, setData] = useState<Timeline | null>(null);
  const [error, setError] = useState<{ status: number; message: string } | null>(null);
  useSeo(data ? patientName(data.patient) : "Patient", undefined, { noindex: true });

  useEffect(() => {
    let cancelled = false;
    setData(null);
    setError(null);
    if (!Number.isInteger(pid)) { setError({ status: 404, message: "" }); return; }
    api.timeline(pid).then((d) => { if (!cancelled) setData(d); }).catch((e) => {
      if (!cancelled) setError({ status: e instanceof ApiError ? e.status : 0, message: errorMessage(e, "Couldn't load this patient.") });
    });
    return () => { cancelled = true; };
  }, [pid]);

  const back = (
    <Link to="/app/patients" className="mb-4 inline-flex items-center gap-1.5 text-[13px] text-muted transition hover:text-ink">
      <ArrowLeft className="h-3.5 w-3.5" />Patients
    </Link>
  );

  if (error) {
    return (
      <div>{back}{error.status === 404
        ? <EmptyState icon={<UserRound className="h-5 w-5" />} title="Patient not found" body="This patient may belong to another account."
          action={<ButtonLink to="/app/patients" variant="secondary">All patients</ButtonLink>} />
        : <ErrorNote>{error.message}</ErrorNote>}
      </div>
    );
  }

  const p = data?.patient;
  const cases = data?.cases ?? [];
  const latest = cases[cases.length - 1];
  const newVisit = `/app/new${p?.reference ? `?ref=${encodeURIComponent(p.reference)}` : ""}`;

  return (
    <div>
      {back}
      <div className="mb-8 flex flex-col gap-4 sm:flex-row sm:items-end sm:justify-between">
        {data && p ? (
          <div className="flex items-center gap-4">
            <div className="grid h-14 w-14 place-items-center rounded-2xl bg-brand/10 text-brand"><UserRound className="h-6 w-6" /></div>
            <div className="min-w-0">
              <h1 className="truncate text-2xl font-semibold tracking-tight sm:text-[28px]">{patientName(p)}</h1>
              <p className="mt-0.5 flex flex-wrap items-center gap-x-2 text-sm text-muted">
                {p.reference && p.name && <span className="font-mono text-xs">{p.reference}</span>}
                {patientMeta({ ...p, reference: null }) && <span>{patientMeta({ ...p, reference: null })}</span>}
                {!patientMeta(p) && !(p.reference && p.name) && <span>No demographics recorded</span>}
              </p>
            </div>
          </div>
        ) : (
          <div className="flex items-center gap-4"><div className="skeleton h-14 w-14 rounded-2xl" /><div className="space-y-2"><div className="skeleton h-7 w-48" /><div className="skeleton h-4 w-32" /></div></div>
        )}
        <ButtonLink to={newVisit}><Plus className="h-4 w-4" />New visit</ButtonLink>
      </div>

      <div className="mb-4 grid grid-cols-2 gap-3 sm:grid-cols-4 sm:gap-4">
        <Stat label="Visits" value={data ? String(cases.length) : null} />
        <Stat label="Latest score" value={latest ? `${Math.round(latest.probability * 100)}%` : data ? "–" : null} extra={latest && <RiskBadge tier={latest.tier} size="sm" />} />
        <Stat label="Change since last" value={null} extra={data ? <Change value={data.latest_change} /> : <div className="skeleton h-7 w-20" />} />
        <Stat label="Last visit" value={latest ? shortDate(latest.created_at) : data ? "–" : null} />
      </div>

      <div className="grid gap-4 lg:grid-cols-[minmax(0,1.5fr)_minmax(0,1fr)]">
        <Card className="p-5">
          <SectionTitle action={<span className="text-xs text-muted">Suspicion score per visit</span>}>Trajectory</SectionTitle>
          {!data ? <div className="skeleton h-[260px]" /> : cases.length < 2 ? (
            <div className="grid h-[260px] place-items-center rounded-xl border border-dashed border-line text-center">
              <div className="max-w-xs px-6">
                <p className="text-sm font-medium">{cases.length ? "One visit so far" : "No visits yet"}</p>
                <p className="mt-1 text-xs text-muted">Run another analysis with reference <span className="font-mono">{p?.reference || "(none)"}</span> to start tracking change over time.</p>
                <ButtonLink to={newVisit} size="sm" variant="secondary" className="mt-4">Add a visit<ArrowRight className="h-3.5 w-3.5" /></ButtonLink>
              </div>
            </div>
          ) : <TrajectoryChart cases={cases} />}
        </Card>
        <Card className="p-5">
          <SectionTitle>Visits</SectionTitle>
          {!data ? <div className="-mx-2">{[0, 1, 2].map((i) => <CaseItemSkeleton key={i} />)}</div> : cases.length ? (
            <div className="-mx-2 space-y-0.5">
              {[...cases].reverse().map((c, i) => (
                <CaseItem key={c.id} c={c} meta={[`Visit ${cases.length - i}`, formatDate(c.created_at), c.site].filter(Boolean).join(" · ")} />
              ))}
            </div>
          ) : <p className="py-6 text-center text-sm text-muted">No visits recorded.</p>}
        </Card>
      </div>
    </div>
  );
}

function Stat({ label, value, extra }: { label: string; value: string | null; extra?: React.ReactNode }) {
  return (
    <div className="card p-4">
      <p className="text-[13px] text-muted">{label}</p>
      <div className="mt-2 flex items-center gap-2">
        {value !== null && <span className="text-2xl font-semibold tabular-nums tracking-tight">{value}</span>}
        {value === null && !extra && <div className="skeleton h-7 w-16" />}
        {extra}
      </div>
    </div>
  );
}

function Change({ value }: { value: number | null }) {
  if (value === null) return <span className="text-2xl font-semibold text-muted">–</span>;
  const pts = Math.round(value * 1000) / 10;
  const Icon = Math.abs(pts) < 0.5 ? Minus : pts > 0 ? ArrowUpRight : ArrowDownRight;
  const tone = Math.abs(pts) < 0.5 ? "text-muted" : pts > 0 ? "text-high" : "text-low";
  return (
    <span className={cn("flex items-center gap-1 text-2xl font-semibold tabular-nums tracking-tight", tone)} title="Percentage points vs previous visit">
      <Icon className="h-5 w-5" />{pts > 0 ? "+" : ""}{pts.toFixed(1)}<span className="text-sm font-medium">pts</span>
    </span>
  );
}

type TipProps = { active?: boolean; payload?: { payload?: Row }[] };
interface Row { i: number; date: string; p: number; tier: CaseSummary["tier"]; id: string }

function TrajectoryChart({ cases }: { cases: CaseSummary[] }) {
  const rows: Row[] = cases.map((c, i) => ({ i: i + 1, date: shortDate(c.created_at), p: c.probability, tier: c.tier, id: c.id }));
  const th = THRESHOLD;
  return (
    <div className="h-[260px]">
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={rows} margin={{ top: 12, right: 40, bottom: 0, left: -20 }}>
          <CartesianGrid vertical={false} stroke="rgb(var(--line))" strokeDasharray="3 3" />
          <XAxis dataKey="date" stroke="rgb(var(--muted))" fontSize={11} tickLine={false} axisLine={false} />
          <YAxis domain={[0, 1]} ticks={[0, th, HIGH_THRESHOLD, 1]} tickFormatter={(v: number) => `${Math.round(v * 100)}%`}
            stroke="rgb(var(--muted))" fontSize={11} tickLine={false} axisLine={false} />
          <ReferenceLine y={th} stroke={TIER_RGB.moderate} strokeOpacity={0.6} strokeDasharray="4 3"
            label={{ value: "Flag", position: "right", fill: "rgb(var(--muted))", fontSize: 10 }} />
          <ReferenceLine y={HIGH_THRESHOLD} stroke={TIER_RGB.high} strokeOpacity={0.6} strokeDasharray="4 3"
            label={{ value: "High", position: "right", fill: "rgb(var(--muted))", fontSize: 10 }} />
          <Tooltip cursor={{ stroke: "rgb(var(--line))" }}
            content={({ active, payload }: TipProps) => {
              const r = payload?.[0]?.payload;
              if (!active || !r) return null;
              return <ChartTooltip title={`Visit ${r.i} · ${r.date}`} rows={[{ label: "Score", value: `${Math.round(r.p * 100)}%`, color: TIER_RGB[r.tier] }]} />;
            }} />
          <Line type="monotone" dataKey="p" stroke="rgb(var(--ink))" strokeOpacity={0.55} strokeWidth={2} isAnimationActive
            dot={(props: { cx?: number; cy?: number; payload?: Row; index?: number }) => (
              <circle key={props.index} cx={props.cx} cy={props.cy} r={5.5} fill={props.payload ? TIER_RGB[props.payload.tier] : "transparent"}
                stroke="rgb(var(--surface))" strokeWidth={2} />
            )}
            activeDot={{ r: 7, stroke: "rgb(var(--surface))", strokeWidth: 2 }} />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
