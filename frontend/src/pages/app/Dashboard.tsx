import { useCallback, useEffect, useMemo, useState } from "react";
import { Link } from "react-router-dom";
import { motion } from "framer-motion";
import {
  Activity, AlertTriangle, ArrowRight, Check, CheckCircle2, Clock, Command, FileStack, Gauge, KeyRound, MessageSquareText, Plus,
  ScanLine, Sparkles, Timer, X,
} from "lucide-react";
import { Bar, BarChart, CartesianGrid, Cell, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { api, type Dashboard as DashboardData } from "@/lib/api";
import { useAuth } from "@/lib/auth";
import { useSeo } from "@/lib/seo";
import { cn, pct } from "@/lib/utils";
import { ButtonLink, Card, EmptyState, ErrorNote, Kbd, buttonClass } from "@/components/ui";
import { MouthMap } from "@/components/MouthMap";
import { CaseItem, CaseItemSkeleton, ChartTooltip } from "@/components/app/CaseItem";
import { errorMessage, HIGH_THRESHOLD, SectionTitle, Segmented, StatTile, THRESHOLD, TIER_RGB, tierOf } from "@/components/app/kit";

const PERIODS = [
  { value: "7", label: "7d" },
  { value: "30", label: "30d" },
  { value: "90", label: "90d" },
] as const;
type Period = (typeof PERIODS)[number]["value"];

const PALETTE_FLAG = "oralaid-palette-tried";
const DISMISS_FLAG = "oralaid-onboarding-dismissed";
const readFlag = (k: string) => { try { return localStorage.getItem(k) === "1"; } catch { return false; } };
const writeFlag = (k: string) => { try { localStorage.setItem(k, "1"); } catch { /* ignore */ } };

const shortDate = (d: string) => new Date(`${d}T00:00:00`).toLocaleDateString(undefined, { month: "short", day: "numeric" });
const formatMs = (ms: number | null) => (ms === null ? "–" : ms >= 1000 ? `${(ms / 1000).toFixed(1)}s` : `${ms}ms`);

function greeting() {
  const h = new Date().getHours();
  return h < 12 ? "Good morning" : h < 18 ? "Good afternoon" : "Good evening";
}

export default function Dashboard() {
  useSeo("Overview", undefined, { noindex: true });
  const user = useAuth((s) => s.user);
  const [period, setPeriod] = useState<Period>("30");
  const [data, setData] = useState<DashboardData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const load = useCallback(async (days: number) => {
    setLoading(true);
    setError(null);
    try {
      setData(await api.dashboard(days));
    } catch (e) {
      setError(errorMessage(e, "Couldn't load your overview."));
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => { load(Number(period)); }, [period, load]);

  const firstName = (user?.full_name || "").replace(/^(dr\.?|prof\.?)\s+/i, "").split(/\s+/)[0];
  const today = new Date().toLocaleDateString(undefined, { weekday: "long", month: "long", day: "numeric" });

  const series = useMemo(() => {
    const t = data?.trend ?? [];
    return {
      total: t.map((d) => d.low + d.moderate + d.high),
      high: t.map((d) => d.high),
      highInPeriod: t.reduce((a, d) => a + d.high, 0),
    };
  }, [data]);

  const isEmpty = !!data && data.totals.cases === 0;

  return (
    <div>
      <div className="mb-8 flex flex-col gap-4 sm:flex-row sm:items-end sm:justify-between">
        <div>
          <p className="eyebrow">{today}</p>
          <h1 className="mt-2 text-2xl font-semibold tracking-tight sm:text-[28px]">
            {greeting()}{firstName ? `, ${firstName}` : ""}
          </h1>
          <p className="mt-1 text-sm text-muted">
            {data && !isEmpty
              ? data.totals.awaiting_review
                ? `${data.totals.awaiting_review} flagged case${data.totals.awaiting_review === 1 ? "" : "s"} waiting for your review.`
                : "Your review queue is clear."
              : "Here is how your screening is going."}
          </p>
        </div>
        <div className="flex items-center gap-2">
          <Segmented label="Period" size="sm" value={period} onChange={setPeriod} options={PERIODS} />
          <ButtonLink to="/app/new" size="sm" className="hidden sm:inline-flex"><Plus className="h-4 w-4" />New analysis</ButtonLink>
        </div>
      </div>

      {error && <div className="mb-6"><ErrorNote>{error}</ErrorNote></div>}

      {data && <Onboarding data={data} />}

      <div className="grid grid-cols-2 gap-3 sm:gap-4 lg:grid-cols-3 xl:grid-cols-6">
        <StatTile loading={loading && !data} label="Cases" icon={<FileStack className="h-4 w-4" />}
          value={data?.totals.last_period ?? 0} spark={series.total.some(Boolean) ? series.total : undefined}
          sub={`${data?.totals.cases ?? 0} all time`} />
        <StatTile loading={loading && !data} label="High risk" tone="high" icon={<AlertTriangle className="h-4 w-4" />}
          value={series.highInPeriod} spark={series.high.some(Boolean) ? series.high : undefined} sparkColor="rgb(var(--high))"
          sub={`${data?.totals.referred ?? 0} referred`} />
        <StatTile loading={loading && !data} label="Awaiting review" tone={data?.totals.awaiting_review ? "moderate" : undefined}
          icon={<Clock className="h-4 w-4" />} value={data?.totals.awaiting_review ?? 0} sub="Moderate or high, no feedback" />
        <StatTile loading={loading && !data} label="Clinician agreement" icon={<CheckCircle2 className="h-4 w-4" />}
          value={pct(data?.agreement_rate)} sub={data?.reviewed ? `${data.reviewed} reviewed` : "No reviews yet"} />
        <StatTile loading={loading && !data} label="Avg image quality" icon={<Gauge className="h-4 w-4" />}
          value={data?.avg_quality ?? "–"} sub={data?.avg_quality !== null && data?.avg_quality !== undefined ? "out of 100" : "In this period"} />
        <StatTile loading={loading && !data} label="Avg inference" icon={<Timer className="h-4 w-4" />}
          value={formatMs(data?.avg_inference_ms ?? null)} sub="Per image, with TTA" />
      </div>

      {isEmpty ? (
        <div className="mt-6">
          <EmptyState
            icon={<ScanLine className="h-5 w-5" />}
            title="No analyses yet"
            body="Upload or capture an intraoral photo to get a suspicion score, Grad-CAM attention map and a localised region in under a minute."
            action={<ButtonLink to="/app/new"><Plus className="h-4 w-4" />Run your first analysis</ButtonLink>}
          />
        </div>
      ) : (
        <>
          <div className="mt-6 grid gap-4 lg:grid-cols-[minmax(0,1.6fr)_minmax(0,1fr)]">
            <Card className="p-5">
              <SectionTitle action={<Legend />}>Screening volume by tier</SectionTitle>
              {data ? <TrendChart data={data} /> : <div className="skeleton h-[240px]" />}
            </Card>
            <Card className="p-5">
              <SectionTitle action={<span className="text-xs text-muted">All time</span>}>Score distribution</SectionTitle>
              {data ? <Histogram data={data} /> : <div className="skeleton h-[240px]" />}
            </Card>
          </div>

          <div className="mt-4 grid gap-4 lg:grid-cols-3">
            <Card className="p-5 lg:col-span-1">
              <SectionTitle action={data?.queue.length ? <Link to="/app/cases?review=1" className="text-xs font-medium text-brand hover:underline">View all</Link> : undefined}>
                Review queue
              </SectionTitle>
              {!data ? <ListSkeleton /> : data.queue.length ? (
                <div className="-mx-2 space-y-0.5">{data.queue.map((c) => <CaseItem key={c.id} c={c} />)}</div>
              ) : (
                <div className="flex flex-col items-center py-8 text-center">
                  <div className="grid h-10 w-10 place-items-center rounded-full bg-low/10 text-low"><Check className="h-5 w-5" /></div>
                  <p className="mt-3 text-sm font-medium">All caught up</p>
                  <p className="mt-1 text-xs text-muted">Flagged cases without feedback appear here.</p>
                </div>
              )}
            </Card>
            <Card className="p-5 lg:col-span-1">
              <SectionTitle action={<Link to="/app/cases" className="text-xs font-medium text-brand hover:underline">All cases</Link>}>Recent cases</SectionTitle>
              {!data ? <ListSkeleton /> : (
                <div className="-mx-2 space-y-0.5">{data.recent.map((c) => <CaseItem key={c.id} c={c} />)}</div>
              )}
            </Card>
            <Card className="p-5">
              <SectionTitle action={<span className="text-xs text-muted">All time</span>}>Where cases cluster</SectionTitle>
              {data ? <SiteHeat data={data} /> : <div className="skeleton aspect-square" />}
            </Card>
          </div>
        </>
      )}
    </div>
  );
}

function ListSkeleton() {
  return <div className="-mx-2">{[0, 1, 2, 3].map((i) => <CaseItemSkeleton key={i} />)}</div>;
}

function Legend() {
  return (
    <div className="flex items-center gap-3 text-xs text-muted">
      {(["low", "moderate", "high"] as const).map((t) => (
        <span key={t} className="flex items-center gap-1.5 capitalize"><span className="h-2 w-2 rounded-sm" style={{ background: TIER_RGB[t] }} />{t}</span>
      ))}
    </div>
  );
}

const axis = { stroke: "rgb(var(--muted))", fontSize: 11, tickLine: false, axisLine: false } as const;

type TipProps = { active?: boolean; payload?: any[]; label?: string | number };

function TrendChart({ data }: { data: DashboardData }) {
  const rows = data.trend;
  const interval = rows.length > 31 ? Math.ceil(rows.length / 8) : rows.length > 10 ? 4 : 0;
  const total = rows.reduce((a, d) => a + d.low + d.moderate + d.high, 0);
  return (
    <div className="relative h-[240px]">
      {total === 0 && (
        <div className="absolute inset-0 z-10 grid place-items-center text-sm text-muted">No cases in this period</div>
      )}
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={rows} margin={{ top: 4, right: 4, bottom: 0, left: -24 }} barCategoryGap={rows.length > 31 ? 1 : "20%"}>
          <CartesianGrid vertical={false} stroke="rgb(var(--line))" strokeDasharray="3 3" />
          <XAxis dataKey="date" tickFormatter={shortDate} interval={interval} {...axis} />
          <YAxis allowDecimals={false} {...axis} />
          <Tooltip cursor={{ fill: "rgb(var(--surface-2))" }}
            content={({ active, payload, label }: TipProps) =>
              active && payload?.length ? (
                <ChartTooltip title={shortDate(String(label))} rows={[...payload].reverse().map((p) => ({
                  label: String(p.dataKey).replace(/^./, (s) => s.toUpperCase()), value: p.value as number, color: p.color as string,
                }))} />
              ) : null} />
          <Bar dataKey="low" stackId="t" fill={TIER_RGB.low} stroke="rgb(var(--surface))" strokeWidth={1} />
          <Bar dataKey="moderate" stackId="t" fill={TIER_RGB.moderate} stroke="rgb(var(--surface))" strokeWidth={1} />
          <Bar dataKey="high" stackId="t" fill={TIER_RGB.high} stroke="rgb(var(--surface))" strokeWidth={1} radius={[3, 3, 0, 0]} />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

function Histogram({ data }: { data: DashboardData }) {
  const rows = data.histogram.map((h) => ({ x: Number(h.bin) + 0.05, count: h.count, bin: h.bin }));
  const threshold = data.threshold || THRESHOLD;
  return (
    <div className="h-[240px]">
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={rows} margin={{ top: 16, right: 8, bottom: 0, left: -24 }}>
          <CartesianGrid vertical={false} stroke="rgb(var(--line))" strokeDasharray="3 3" />
          <XAxis dataKey="x" type="number" domain={[0, 1]} ticks={[0, threshold, HIGH_THRESHOLD, 1]}
            tickFormatter={(v: number) => `${Math.round(v * 100)}%`} {...axis} />
          <YAxis allowDecimals={false} {...axis} />
          <Tooltip cursor={{ fill: "rgb(var(--surface-2))" }}
            content={({ active, payload }: TipProps) => {
              const r = payload?.[0]?.payload as (typeof rows)[number] | undefined;
              if (!active || !r) return null;
              const lo = Math.round(Number(r.bin) * 100);
              return <ChartTooltip title={`Score ${lo}–${lo + 10}%`} rows={[{ label: "Cases", value: r.count, color: TIER_RGB[tierOf(r.x)] }]} />;
            }} />
          <ReferenceLine x={threshold} stroke="rgb(var(--ink))" strokeOpacity={0.45} strokeDasharray="4 3"
            label={{ value: "Flag", position: "top", fill: "rgb(var(--muted))", fontSize: 10 }} />
          <ReferenceLine x={HIGH_THRESHOLD} stroke="rgb(var(--ink))" strokeOpacity={0.45} strokeDasharray="4 3"
            label={{ value: "High", position: "top", fill: "rgb(var(--muted))", fontSize: 10 }} />
          <Bar dataKey="count" radius={[3, 3, 0, 0]} barSize={18}>
            {rows.map((r) => <Cell key={r.bin} fill={TIER_RGB[tierOf(r.x)]} fillOpacity={0.85} />)}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

function SiteHeat({ data }: { data: DashboardData }) {
  const counts = Object.fromEntries(data.sites.map((s) => [s.site, s.count]));
  const max = Math.max(1, ...data.sites.map((s) => s.count));
  return (
    <div>
      <MouthMap counts={counts} className="mx-auto max-w-[240px]" />
      <ul className="mt-4 space-y-2">
        {data.sites.slice(0, 5).map((s) => (
          <li key={s.site} className="flex items-center gap-3 text-xs">
            <span className="w-32 truncate text-muted">{s.site}</span>
            <div className="h-1.5 flex-1 overflow-hidden rounded-full bg-surface-2">
              <motion.div className="h-full rounded-full bg-high/70" initial={{ width: 0 }} animate={{ width: `${(s.count / max) * 100}%` }} transition={{ duration: 0.6 }} />
            </div>
            <span className="w-6 text-right tabular-nums text-ink">{s.count}</span>
          </li>
        ))}
      </ul>
    </div>
  );
}

function Onboarding({ data }: { data: DashboardData }) {
  const [hasKey, setHasKey] = useState(false);
  const [tried, setTried] = useState(() => readFlag(PALETTE_FLAG));
  const [dismissed, setDismissed] = useState(() => readFlag(DISMISS_FLAG));

  useEffect(() => {
    api.keys().then((r) => setHasKey(r.keys.length > 0)).catch(() => undefined);
  }, []);
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "k") { writeFlag(PALETTE_FLAG); setTried(true); }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  const steps = [
    { done: data.totals.cases > 0, label: "Run your first analysis", body: "Upload or capture an intraoral photo.", icon: ScanLine, to: "/app/new" },
    { done: data.reviewed > 0, label: "Add clinical feedback", body: "Agree or disagree to track model agreement.", icon: MessageSquareText, to: data.recent[0] ? `/app/cases/${data.recent[0].id}` : "/app/cases" },
    { done: hasKey, label: "Create an API key", body: "Connect your EHR or imaging system.", icon: KeyRound, to: "/app/settings#keys" },
    { done: tried, label: "Try the command palette", body: "Press Ctrl/Cmd + K anywhere in the app.", icon: Command, to: null },
  ];
  const doneCount = steps.filter((s) => s.done).length;
  if (dismissed || doneCount === steps.length || data.totals.cases > 5) return null;

  return (
    <motion.div initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} className="card relative mb-6 overflow-hidden p-5 sm:p-6">
      <div className="pointer-events-none absolute -right-20 -top-24 h-64 w-64 rounded-full bg-brand/10 blur-3xl" />
      <button onClick={() => { writeFlag(DISMISS_FLAG); setDismissed(true); }} aria-label="Dismiss getting started"
        className="absolute right-3 top-3 grid h-8 w-8 place-items-center rounded-lg text-muted hover:bg-surface-2 hover:text-ink">
        <X className="h-4 w-4" />
      </button>
      <div className="relative flex flex-col gap-1 pr-8 sm:flex-row sm:items-center sm:gap-4">
        <div className="flex items-center gap-2"><Sparkles className="h-4 w-4 text-brand" /><h2 className="text-[15px] font-semibold">Get started with OralAID</h2></div>
        <span className="text-xs text-muted tabular-nums">{doneCount} of {steps.length} complete</span>
      </div>
      <div className="relative mt-3 h-1 overflow-hidden rounded-full bg-surface-2">
        <motion.div className="h-full rounded-full bg-brand" initial={{ width: 0 }} animate={{ width: `${(doneCount / steps.length) * 100}%` }} />
      </div>
      <div className="relative mt-5 grid gap-3 sm:grid-cols-2 xl:grid-cols-4">
        {steps.map((s) => {
          const inner = (
            <>
              <span className={cn("grid h-8 w-8 shrink-0 place-items-center rounded-lg", s.done ? "bg-low/15 text-low" : "bg-surface-2 text-muted")}>
                {s.done ? <Check className="h-4 w-4" /> : <s.icon className="h-4 w-4" />}
              </span>
              <span className="min-w-0">
                <span className={cn("block text-sm font-medium", s.done && "text-muted line-through decoration-line")}>{s.label}</span>
                <span className="mt-0.5 block text-xs text-muted">
                  {s.to === null && !s.done ? <>Press <Kbd>Ctrl</Kbd> <Kbd>K</Kbd> or <Kbd>Cmd</Kbd> <Kbd>K</Kbd></> : s.body}
                </span>
              </span>
              {s.to && !s.done && <ArrowRight className="ml-auto h-4 w-4 shrink-0 text-muted transition group-hover:translate-x-0.5 group-hover:text-ink" />}
            </>
          );
          const cls = "group flex items-start gap-3 rounded-xl border border-line bg-surface p-3 text-left transition";
          return s.to && !s.done
            ? <Link key={s.label} to={s.to} className={cn(cls, "hover:border-ink/20 hover:shadow-soft")}>{inner}</Link>
            : <div key={s.label} className={cls}>{inner}</div>;
        })}
      </div>
      {data.totals.cases === 0 && (
        <div className="relative mt-5 flex flex-wrap items-center gap-3 text-xs text-muted">
          <Activity className="h-3.5 w-3.5" />Results are decision support only. Always confirm with clinical examination and histopathology.
          <Link to="/app/new" className={buttonClass("primary", "sm", "ml-auto")}>Start an analysis</Link>
        </div>
      )}
    </motion.div>
  );
}
