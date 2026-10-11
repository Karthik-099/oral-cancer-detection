import { useCallback, useEffect, useState, type ReactNode } from "react";
import { Link } from "react-router-dom";
import { Activity, BookOpen, Cpu, KeyRound, RefreshCw, Server } from "lucide-react";
import { api } from "@/lib/api";
import { useSeo } from "@/lib/seo";
import { cn } from "@/lib/utils";
import { Badge, Button } from "@/components/ui";
import { CodeBlock, CodeTabs } from "@/components/site/CodeBlock";

const BASE = "https://oralaid.ai";

const TOC = [
  { id: "overview", label: "Overview" },
  { id: "authentication", label: "Authentication" },
  { id: "quickstart", label: "Quickstart" },
  { id: "response", label: "Response schema" },
  { id: "endpoints", label: "Endpoints" },
  { id: "errors", label: "Errors" },
  { id: "rate-limits", label: "Rate limits" },
  { id: "status", label: "API status" },
];
const TOC_IDS = TOC.map((t) => t.id);

const CURL = `curl -X POST ${BASE}/api/cases \\
  -H "X-API-Key: $ORALAID_API_KEY" \\
  -F "files=@lesion_front.jpg" \\
  -F "files=@lesion_close.jpg" \\
  -F "patient_reference=P-2231" \\
  -F "patient_age=54" \\
  -F "patient_sex=male" \\
  -F "site=Tongue (lateral)" \\
  -F 'risk_factors=["tobacco","areca nut"]' \\
  -F "notes=Non-healing ulcer, 3 weeks" \\
  -F "with_lime=false"`;

const PYTHON = `import json, os, requests

resp = requests.post(
    "${BASE}/api/cases",
    headers={"X-API-Key": os.environ["ORALAID_API_KEY"]},
    files=[
        ("files", open("lesion_front.jpg", "rb")),
        ("files", open("lesion_close.jpg", "rb")),
    ],
    data={
        "patient_reference": "P-2231",
        "patient_age": 54,
        "patient_sex": "male",
        "site": "Tongue (lateral)",
        "risk_factors": json.dumps(["tobacco", "areca nut"]),
        "notes": "Non-healing ulcer, 3 weeks",
        "with_lime": "false",
    },
    timeout=120,
)
resp.raise_for_status()
case = resp.json()
print(case["tier"], round(case["probability"], 2), case["recommendation"])`;

const JS = `const form = new FormData();
form.append("files", fileInput.files[0]);          // File or Blob
form.append("patient_reference", "P-2231");
form.append("patient_age", "54");
form.append("patient_sex", "male");
form.append("site", "Tongue (lateral)");
form.append("risk_factors", JSON.stringify(["tobacco", "areca nut"]));
form.append("with_lime", "false");

const res = await fetch("${BASE}/api/cases", {
  method: "POST",
  headers: { "X-API-Key": process.env.ORALAID_API_KEY },
  body: form,                                     // let fetch set the boundary
});
if (!res.ok) throw new Error((await res.json()).detail);
const caseResult = await res.json();
console.log(caseResult.tier, caseResult.probability);`;

const RESPONSE = `{
  "id": "9f3a1c2e7b4d4a10",
  "created_at": "2026-10-11T09:24:51",
  "site": "Tongue (lateral)",
  "status": "open",                      // open | referred | closed
  "probability": 0.71,                   // highest across images
  "tier": "high",                        // low | moderate | high
  "label": "High risk",
  "finding": "Suspicious oral lesion features",
  "color": "#ef4444",
  "recommendation": "Refer to an oral medicine / maxillofacial specialist for evaluation and possible biopsy.",
  "image_count": 2,
  "thumbnail": "/files/9f3a..._original.jpg?exp=...&sig=...",
  "patient": { "id": 42, "reference": "P-2231", "age": 54, "sex": "male",
               "risk_factors": ["tobacco", "areca nut"] },
  "feedback": null,
  "notes": "Non-healing ulcer, 3 weeks",
  "images": [
    {
      "index": 0,
      "probability": 0.71,
      "uncertainty": 0.05,
      "confidence": 0.86,
      "tier": "high",
      "threshold": 0.3,
      "quality": { "score": 84, "sharpness": 212.4, "brightness": 131.0,
                   "resolution": "1600x1200", "issues": [] },
      "region_area_ratio": 0.12,
      "attention_area_ratio": 0.18,
      "inference_ms": 1840,
      "files": {
        "original":     "/files/..._original.jpg?exp=...&sig=...",
        "gradcam":      "/files/..._gradcam.png?exp=...&sig=...",
        "heatmap":      "/files/..._heatmap.png?exp=...&sig=...",
        "segmentation": "/files/..._segmentation.png?exp=...&sig=..."
      }
    }
  ]
}`;

const AUTH_LOGIN = `curl -X POST ${BASE}/api/auth/login \\
  -H "Content-Type: application/json" \\
  -d '{"email":"you@clinic.org","password":"..."}'

# => { "token": "eyJhbGciOi...", "user": { ... } }

curl ${BASE}/api/cases -H "Authorization: Bearer eyJhbGciOi..."`;

const AUTH_KEY = `curl ${BASE}/api/cases -H "X-API-Key: oak_..."`;

const ENDPOINTS: { m: string; path: string; d: string }[] = [
  { m: "GET", path: "/api/health", d: "Service and model readiness. No auth." },
  { m: "GET", path: "/api/models", d: "Model cards, metrics and thresholds. No auth." },
  { m: "POST", path: "/api/cases", d: "Analyse 1 to 6 images and create a case (multipart)." },
  { m: "GET", path: "/api/cases", d: "List cases. Query: tier, status, q, limit, offset." },
  { m: "GET", path: "/api/cases/{id}", d: "Case detail with per-image results and signed URLs." },
  { m: "PATCH", path: "/api/cases/{id}", d: "Update status (open, referred, closed) or notes." },
  { m: "POST", path: "/api/cases/{id}/lime", d: "Compute LIME superpixel explanations for the case." },
  { m: "PUT", path: "/api/cases/{id}/feedback", d: "Record clinician verdict, impression, histopathology, rating." },
  { m: "DELETE", path: "/api/cases/{id}", d: "Delete a case and its images." },
  { m: "GET", path: "/api/cases/export.csv", d: "Export all cases and feedback as CSV." },
  { m: "GET", path: "/api/patients", d: "List patients. Query: q." },
  { m: "GET", path: "/api/patients/{id}/timeline", d: "A patient's cases over time and latest score change." },
  { m: "GET", path: "/api/dashboard", d: "Aggregates for the dashboard. Query: days." },
  { m: "GET", path: "/api/keys", d: "List your API keys (prefix only)." },
  { m: "POST", path: "/api/keys", d: "Create a key. The full key is returned once." },
  { m: "DELETE", path: "/api/keys/{id}", d: "Revoke a key." },
  { m: "GET", path: "/api/audit", d: "Your recent audit events." },
];

const ERRORS: { code: string; name: string; d: string }[] = [
  { code: "400", name: "Bad request", d: "Missing images, too many images, or an image that cannot be decoded." },
  { code: "401", name: "Unauthorized", d: "Missing, invalid or expired token or API key." },
  { code: "403", name: "Forbidden", d: "Authenticated but not allowed, or a signed file link has expired." },
  { code: "404", name: "Not found", d: "The case, patient or key does not exist or is not yours." },
  { code: "413", name: "Payload too large", d: "An image is over 15 MB or its dimensions are too large." },
  { code: "415", name: "Unsupported media type", d: "Only JPEG, PNG, WebP or HEIC images are accepted." },
  { code: "422", name: "Unprocessable", d: "Validation failed, e.g. not an intraoral image, unknown site, bad risk_factors JSON." },
  { code: "423", name: "Locked", d: "Account temporarily locked after repeated failed sign-ins." },
  { code: "429", name: "Too many requests", d: "Rate limit exceeded. Wait for the number of seconds in Retry-After." },
  { code: "503", name: "Service unavailable", d: "The model is still loading. Retry after the Retry-After header." },
];

const METHOD_TONE: Record<string, string> = {
  GET: "text-low bg-low/10",
  POST: "text-brand bg-brand/10",
  PUT: "text-moderate bg-moderate/10",
  PATCH: "text-moderate bg-moderate/10",
  DELETE: "text-high bg-high/10",
};

function DocSection({ id, title, children }: { id: string; title: string; children: ReactNode }) {
  return (
    <section id={id} className="scroll-mt-24 border-t border-line py-12 first:border-t-0 first:pt-0">
      <h2 className="text-2xl font-semibold tracking-tight">{title}</h2>
      <div className="mt-5 space-y-5 text-[15px] leading-relaxed text-muted [&_strong]:font-medium [&_strong]:text-ink">{children}</div>
    </section>
  );
}

function Code({ children }: { children: ReactNode }) {
  return <code className="rounded-md border border-line bg-surface-2 px-1.5 py-0.5 font-mono text-[12.5px] text-ink">{children}</code>;
}

function useActiveSection(ids: string[]) {
  const [active, setActive] = useState(ids[0]);
  useEffect(() => {
    const obs = new IntersectionObserver(
      (entries) => {
        const visible = entries.filter((e) => e.isIntersecting).sort((a, b) => a.boundingClientRect.top - b.boundingClientRect.top);
        if (visible[0]) setActive(visible[0].target.id);
      },
      { rootMargin: "-80px 0px -65% 0px" },
    );
    ids.forEach((id) => { const el = document.getElementById(id); if (el) obs.observe(el); });
    return () => obs.disconnect();
  }, [ids]);
  return active;
}

type Health = { status: string; version: string; model_ready: boolean };

function StatusPanel() {
  const [health, setHealth] = useState<Health | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [latency, setLatency] = useState<number | null>(null);
  const [loading, setLoading] = useState(true);
  const [checkedAt, setCheckedAt] = useState<Date | null>(null);

  const check = useCallback(async () => {
    setLoading(true);
    const t0 = performance.now();
    try {
      const h = await api.health();
      setHealth(h);
      setError(null);
      setLatency(Math.round(performance.now() - t0));
    } catch (e) {
      setHealth(null);
      setLatency(null);
      setError((e as Error).message);
    } finally {
      setCheckedAt(new Date());
      setLoading(false);
    }
  }, []);

  useEffect(() => { check(); }, [check]);

  const apiUp = !!health && health.status === "ok";
  const rows: { icon: ReactNode; label: string; state: "up" | "warn" | "down" | "loading"; text: string }[] = [
    { icon: <Server className="h-4 w-4" />, label: "API", state: loading && !checkedAt ? "loading" : apiUp ? "up" : health ? "warn" : "down",
      text: loading && !checkedAt ? "Checking..." : apiUp ? "Operational" : health ? `Status: ${health.status}` : "Unreachable" },
    { icon: <Cpu className="h-4 w-4" />, label: "Inference model", state: loading && !checkedAt ? "loading" : health?.model_ready ? "up" : health ? "warn" : "down",
      text: loading && !checkedAt ? "Checking..." : health?.model_ready ? "Loaded and serving" : health ? "Loading weights" : "Unknown" },
  ];
  const dot = { up: "bg-low", warn: "bg-moderate", down: "bg-high", loading: "bg-muted animate-pulse" };

  return (
    <div className="card overflow-hidden">
      <ul className="divide-y divide-line" aria-live="polite">
        {rows.map((r) => (
          <li key={r.label} className="flex items-center gap-3 px-5 py-4">
            <span className="text-muted">{r.icon}</span>
            <span className="flex-1 text-sm font-medium text-ink">{r.label}</span>
            <span className="flex items-center gap-2 text-sm text-ink/80"><span className={cn("h-2 w-2 rounded-full", dot[r.state])} />{r.text}</span>
          </li>
        ))}
      </ul>
      <div className="flex flex-wrap items-center justify-between gap-3 border-t border-line bg-surface-2/50 px-5 py-3 text-xs text-muted">
        <span className="flex flex-wrap gap-x-4 gap-y-1">
          {health && <span>Version <span className="font-mono text-ink">{health.version}</span></span>}
          {latency !== null && <span>Round trip <span className="font-mono text-ink">{latency} ms</span></span>}
          {checkedAt && <span>Checked {checkedAt.toLocaleTimeString()}</span>}
          {error && <span className="text-high">{error}</span>}
        </span>
        <Button size="sm" variant="secondary" onClick={check} loading={loading}>{!loading && <RefreshCw className="h-3.5 w-3.5" />}Refresh</Button>
      </div>
    </div>
  );
}

export default function Docs() {
  useSeo("API documentation", "OralAID REST API: authentication, quickstart for analysing intraoral images, response schema, endpoints, errors, rate limits and live status.");
  const active = useActiveSection(TOC_IDS);

  return (
    <div className="container py-14 sm:py-20">
      <header className="max-w-3xl">
        <p className="eyebrow mb-4 text-brand">Developers</p>
        <h1 className="display text-5xl sm:text-6xl">API <em className="italic text-brand">documentation</em></h1>
        <p className="mt-5 text-base leading-relaxed text-muted sm:text-lg">
          Submit intraoral images, read risk tiers and explanations, and sync clinician feedback from your own systems.
        </p>
      </header>

      {/* Mobile TOC */}
      <nav aria-label="On this page" className="mt-8 -mx-5 overflow-x-auto px-5 lg:hidden">
        <ul className="flex gap-2">
          {TOC.map((t) => (
            <li key={t.id}><a href={`#${t.id}`} className="block whitespace-nowrap rounded-full border border-line px-3 py-1 text-xs text-muted hover:text-ink">{t.label}</a></li>
          ))}
        </ul>
      </nav>

      <div className="mt-12 grid gap-12 lg:grid-cols-[200px_minmax(0,1fr)] lg:gap-16">
        <aside className="hidden lg:block">
          <nav aria-label="On this page" className="sticky top-24">
            <p className="eyebrow mb-3">On this page</p>
            <ul className="space-y-0.5 border-l border-line">
              {TOC.map((t) => (
                <li key={t.id}>
                  <a href={`#${t.id}`} aria-current={active === t.id ? "location" : undefined}
                    className={cn("-ml-px block border-l py-1.5 pl-4 text-sm transition",
                      active === t.id ? "border-brand font-medium text-ink" : "border-transparent text-muted hover:text-ink")}>
                    {t.label}
                  </a>
                </li>
              ))}
            </ul>
          </nav>
        </aside>

        <article className="min-w-0 max-w-3xl">
          <DocSection id="overview" title="Overview">
            <p>
              The OralAID API is a JSON REST API served under <Code>/api</Code>. Images are uploaded as <Code>multipart/form-data</Code>;
              everything else is JSON. All timestamps are UTC in ISO 8601.
            </p>
            <div className="grid gap-3 sm:grid-cols-3">
              {[[BookOpen, "Base URL", `${BASE}/api`], [KeyRound, "Auth", "Bearer JWT or X-API-Key"], [Activity, "Format", "JSON, multipart for images"]].map(([Icon, k, v]) => {
                const I = Icon as typeof BookOpen;
                return (
                  <div key={k as string} className="rounded-xl border border-line bg-surface p-4">
                    <I className="h-4 w-4 text-brand" />
                    <p className="mt-3 text-xs text-muted">{k as string}</p>
                    <p className="mt-0.5 break-all font-mono text-[12.5px] text-ink">{v as string}</p>
                  </div>
                );
              })}
            </div>
            <p className="rounded-xl border border-moderate/25 bg-moderate/5 p-4 text-sm text-ink/90">
              Results are clinical decision support for screening, not a diagnosis. Display the recommendation and keep a clinician in the loop
              in any integration you build.
            </p>
          </DocSection>

          <DocSection id="authentication" title="Authentication">
            <p>
              <strong>For user sessions</strong>, sign in with <Code>POST /api/auth/login</Code> and send the returned token as{" "}
              <Code>Authorization: Bearer &lt;token&gt;</Code>. Tokens are invalidated when the user changes their password or signs out everywhere.
            </p>
            <CodeBlock title="bash · session token" code={AUTH_LOGIN} />
            <p>
              <strong>For integrations</strong>, create an API key in <Link to="/app/settings" className="text-brand underline">Settings</Link> and send it in the{" "}
              <Code>X-API-Key</Code> header. The full key is shown once; store it as a secret. Keys act on behalf of the user who created them and can be revoked at any time.
            </p>
            <CodeBlock title="bash · API key" code={AUTH_KEY} />
          </DocSection>

          <DocSection id="quickstart" title="Quickstart: analyse a case">
            <p>
              Send one to six images as repeated <Code>files</Code> fields to <Code>POST /api/cases</Code>. All other fields are optional.
            </p>
            <CodeTabs tabs={[
              { label: "curl", code: CURL, title: "bash" },
              { label: "Python", code: PYTHON, title: "requests" },
              { label: "JavaScript", code: JS, title: "fetch" },
            ]} />
            <div className="overflow-x-auto rounded-xl border border-line">
              <table className="w-full min-w-[520px] text-left text-sm">
                <caption className="sr-only">Form fields for POST /api/cases</caption>
                <thead className="bg-surface-2/60 text-xs text-muted">
                  <tr><th scope="col" className="px-4 py-2.5 font-medium">Field</th><th scope="col" className="px-4 py-2.5 font-medium">Type</th><th scope="col" className="px-4 py-2.5 font-medium">Description</th></tr>
                </thead>
                <tbody className="divide-y divide-line text-ink/90">
                  {[
                    ["files[]", "file, 1 to 6", "Intraoral photos. JPEG, PNG, WebP or HEIC, 15 MB each."],
                    ["patient_reference", "string", "Your patient ID. Cases with the same reference join one timeline."],
                    ["patient_age", "integer", "0 to 120."],
                    ["patient_sex", "enum", "female | male | other"],
                    ["site", "string", "Anatomical site, e.g. Tongue (lateral), Buccal mucosa, Gingiva."],
                    ["risk_factors", "JSON string", "List of strings, e.g. [\"tobacco\",\"alcohol\"]."],
                    ["notes", "string", "Free text, up to 4,000 characters."],
                    ["with_lime", "boolean", "Compute LIME explanations now (slower). Default false."],
                  ].map(([f, t, d]) => (
                    <tr key={f}><td className="px-4 py-2.5 font-mono text-[12.5px]">{f}</td><td className="px-4 py-2.5 text-xs text-muted">{t}</td><td className="px-4 py-2.5">{d}</td></tr>
                  ))}
                </tbody>
              </table>
            </div>
          </DocSection>

          <DocSection id="response" title="Response schema">
            <p>
              A successful call returns <Code>201 Created</Code> with the case. The case-level <Code>probability</Code> and <Code>tier</Code> come from the most
              suspicious image. Tiers: <strong>low</strong> below 0.30, <strong>moderate</strong> from 0.30, <strong>high</strong> at 0.55 and above.
            </p>
            <CodeBlock title="201 Created · application/json" code={RESPONSE} />
            <p>
              <strong>Image URLs are signed and expire after one hour.</strong> Fetch them promptly or re-request the case with{" "}
              <Code>GET /api/cases/{"{id}"}</Code> to get fresh links. Prefix relative paths with the base URL.
            </p>
          </DocSection>

          <DocSection id="endpoints" title="Endpoints">
            <div className="overflow-hidden rounded-xl border border-line">
              <ul className="divide-y divide-line">
                {ENDPOINTS.map((e) => (
                  <li key={e.m + e.path} className="flex flex-col gap-1.5 px-4 py-3 sm:flex-row sm:items-center sm:gap-4">
                    <span className="flex min-w-0 items-center gap-3 sm:w-[300px] sm:shrink-0">
                      <span className={cn("w-14 shrink-0 rounded-md py-0.5 text-center font-mono text-[11px] font-semibold", METHOD_TONE[e.m])}>{e.m}</span>
                      <span className="truncate font-mono text-[12.5px] text-ink">{e.path}</span>
                    </span>
                    <span className="text-sm text-muted">{e.d}</span>
                  </li>
                ))}
              </ul>
            </div>
          </DocSection>

          <DocSection id="errors" title="Errors">
            <p>Errors return a JSON body with a human-readable <Code>detail</Code> string, and every response carries an <Code>X-Request-ID</Code> header for support.</p>
            <CodeBlock title="429 Too Many Requests" code={`Retry-After: 12\n\n{ "detail": "Too many requests. Please slow down." }`} />
            <div className="overflow-hidden rounded-xl border border-line">
              <dl className="divide-y divide-line">
                {ERRORS.map((e) => (
                  <div key={e.code} className="grid gap-1 px-4 py-3 sm:grid-cols-[150px_1fr] sm:gap-4">
                    <dt className="flex items-center gap-2"><Badge tone={e.code.startsWith("5") ? "high" : e.code === "429" || e.code === "423" ? "moderate" : "neutral"} className="font-mono">{e.code}</Badge><span className="text-sm font-medium text-ink">{e.name}</span></dt>
                    <dd className="text-sm">{e.d}</dd>
                  </div>
                ))}
              </dl>
            </div>
          </DocSection>

          <DocSection id="rate-limits" title="Rate limits">
            <div className="grid gap-3 sm:grid-cols-2">
              <div className="rounded-xl border border-line bg-surface p-5"><p className="display text-4xl text-ink">30<span className="text-lg text-muted"> / min</span></p><p className="mt-1 text-sm">Analyses per user (<Code>POST /api/cases</Code>)</p></div>
              <div className="rounded-xl border border-line bg-surface p-5"><p className="display text-4xl text-ink">120<span className="text-lg text-muted"> / min</span></p><p className="mt-1 text-sm">Requests per API key, all endpoints</p></div>
            </div>
            <p>
              LIME is limited to 10 requests per minute per user. When you exceed a limit you'll get <Code>429</Code> with a <Code>Retry-After</Code> header
              in seconds. Back off for that long, then retry. Plan limits on monthly API calls apply separately.
            </p>
          </DocSection>

          <DocSection id="status" title="API status">
            <p>Live check against <Code>GET /api/health</Code> from your browser.</p>
            <StatusPanel />
          </DocSection>
        </article>
      </div>
    </div>
  );
}
