// Typed client for the OralAID API. In production Netlify proxies /api and /files
// to the backend, so requests are same-origin. Set VITE_API_URL to call a backend
// on another origin directly instead.
export const API_BASE = (import.meta.env.VITE_API_URL as string | undefined)?.replace(/\/$/, "") ?? "";

export type Tier = "low" | "moderate" | "high";

export interface User {
  id: number;
  email: string;
  full_name: string;
  organization?: string | null;
  specialty?: string | null;
  role: "clinician" | "admin";
  plan: string;
  created_at: string;
}

export interface Quality {
  score: number;
  sharpness: number;
  brightness: number;
  resolution: string;
  issues: string[];
}

export interface ImageResult {
  index: number;
  probability: number;
  uncertainty: number;
  confidence: number;
  tier: Tier;
  label: string;
  finding: string;
  color: string;
  recommendation: string;
  threshold: number;
  region_area_ratio: number;
  attention_area_ratio: number;
  quality: Quality;
  files: { original: string; gradcam: string; heatmap: string; segmentation: string; lime?: string };
  lime_top_regions?: { segment: number; weight: number }[];
  inference_ms: number;
}

export interface Feedback {
  verdict: "agree" | "disagree" | "unsure";
  clinical_impression?: "normal" | "benign" | "opmd" | "malignant" | null;
  histopathology?: string | null;
  explanation_rating?: number | null;
  comment?: string | null;
  created_at?: string;
}

export interface PatientRef {
  id: number;
  name?: string | null;
  reference?: string | null;
  age?: number | null;
  sex?: string | null;
  risk_factors?: string[];
}

export interface CaseSummary {
  id: string;
  created_at: string;
  site?: string | null;
  status: "open" | "referred" | "closed";
  probability: number;
  tier: Tier;
  threshold: number;
  high_risk_threshold: number;
  label: string;
  finding: string;
  color: string;
  thumbnail: string | null;
  image_count: number;
  patient: PatientRef | null;
  feedback: Feedback | null;
}

export interface CaseDetail extends CaseSummary {
  notes?: string | null;
  recommendation: string;
  images: ImageResult[];
}

export interface Dashboard {
  totals: {
    cases: number; patients: number; high: number; moderate: number; low: number;
    last_period: number; awaiting_review: number; referred: number;
  };
  agreement_rate: number | null;
  reviewed: number;
  explanation_rating: number | null;
  avg_inference_ms: number | null;
  avg_quality: number | null;
  threshold: number;
  high_risk_threshold: number;
  trend: { date: string; low: number; moderate: number; high: number }[];
  histogram: { bin: string; count: number }[];
  sites: { site: string; count: number }[];
  recent: CaseSummary[];
  queue: CaseSummary[];
}

export interface ModelCard {
  id: string;
  name: string;
  task: string;
  status: "production" | "retired" | "beta";
  architecture: string;
  input: string;
  classes: string[];
  params_m?: number;
  metrics: Record<string, number | string>;
  notes: string;
  runtime?: string | null;
}

export interface PatientRow extends PatientRef {
  case_count: number;
  latest_tier: Tier | null;
  latest_at: string | null;
}

export interface ApiKey {
  id: number; name: string; prefix: string; revoked: boolean; created_at: string; last_used_at: string | null; key?: string;
}

export class ApiError extends Error {
  constructor(public status: number, message: string) {
    super(message);
  }
}

const TOKEN_KEY = "oralaid-token";

export const tokenStore = {
  get: () => {
    try { return localStorage.getItem(TOKEN_KEY); } catch { return null; }
  },
  set: (t: string | null) => {
    try { t ? localStorage.setItem(TOKEN_KEY, t) : localStorage.removeItem(TOKEN_KEY); } catch { /* private mode */ }
  },
};

type Listener = () => void;
const unauthorizedListeners = new Set<Listener>();
export const onUnauthorized = (fn: Listener) => {
  unauthorizedListeners.add(fn);
  return () => unauthorizedListeners.delete(fn);
};

export function fileUrl(path?: string | null) {
  if (!path) return "";
  return path.startsWith("http") ? path : `${API_BASE}${path}`;
}

async function request<T>(path: string, init: RequestInit = {}): Promise<T> {
  const headers = new Headers(init.headers);
  const token = tokenStore.get();
  if (token) headers.set("Authorization", `Bearer ${token}`);
  if (init.body && !(init.body instanceof FormData)) headers.set("Content-Type", "application/json");

  let res: Response;
  try {
    res = await fetch(`${API_BASE}${path}`, { ...init, headers });
  } catch {
    throw new ApiError(0, "Can't reach the OralAID service. Check your connection and try again.");
  }
  if (res.status === 401 && token) unauthorizedListeners.forEach((fn) => fn());
  if (!res.ok) {
    let message = res.statusText || "Request failed";
    try {
      const body = await res.json();
      if (typeof body.detail === "string") message = body.detail;
      else if (Array.isArray(body.detail)) message = body.detail.map((d: { msg: string }) => d.msg.replace(/^Value error, /, "")).join(". ");
    } catch { /* not JSON */ }
    throw new ApiError(res.status, message);
  }
  if (res.headers.get("content-type")?.includes("application/json")) return res.json() as Promise<T>;
  return (await res.blob()) as unknown as T;
}

const json = (body: unknown) => JSON.stringify(body);

export interface CaseFilters { tier?: string; status?: string; q?: string; review?: boolean }

const toQuery = (params: object) =>
  new URLSearchParams(
    Object.entries(params).filter(([, v]) => v !== undefined && v !== "" && v !== false).map(([k, v]) => [k, String(v)]),
  );

export const api = {
  health: () => request<{ status: string; version: string; model_ready: boolean }>("/api/health"),
  models: () => request<{ threshold: number; high_risk_threshold: number; models: ModelCard[] }>("/api/models"),
  lead: (body: { kind: string; email: string; name?: string; organization?: string; message?: string; meta?: object; website?: string }) =>
    request<{ ok: boolean }>("/api/leads", { method: "POST", body: json(body) }),
  demo: (file: Blob) => {
    const fd = new FormData();
    fd.append("file", file, "image.jpg");
    return request<ImageResult & { id: string }>("/api/demo/analyze", { method: "POST", body: fd });
  },

  register: (body: { email: string; password: string; full_name: string; organization?: string; specialty?: string }) =>
    request<{ token: string; user: User }>("/api/auth/register", { method: "POST", body: json(body) }),
  login: (body: { email: string; password: string }) =>
    request<{ token: string; user: User }>("/api/auth/login", { method: "POST", body: json(body) }),
  me: () => request<User>("/api/auth/me"),
  updateMe: (body: Partial<Pick<User, "full_name" | "organization" | "specialty">>) =>
    request<User>("/api/auth/me", { method: "PATCH", body: json(body) }),
  changePassword: (body: { current_password: string; new_password: string }) =>
    request<{ token: string; user: User }>("/api/auth/password", { method: "POST", body: json(body) }),
  logoutAll: () => request<{ ok: boolean }>("/api/auth/logout-all", { method: "POST" }),

  createCase: (fd: FormData) => request<CaseDetail>("/api/cases", { method: "POST", body: fd }),
  cases: (params: CaseFilters & { limit?: number; offset?: number } = {}) =>
    request<{ total: number; cases: CaseSummary[] }>(`/api/cases?${toQuery(params)}`),
  case: (id: string) => request<CaseDetail>(`/api/cases/${encodeURIComponent(id)}`),
  updateCase: (id: string, body: { status?: string; notes?: string }) =>
    request<CaseDetail>(`/api/cases/${encodeURIComponent(id)}`, { method: "PATCH", body: json(body) }),
  lime: (id: string) => request<CaseDetail>(`/api/cases/${encodeURIComponent(id)}/lime`, { method: "POST" }),
  feedback: (id: string, body: Feedback) =>
    request<CaseDetail>(`/api/cases/${encodeURIComponent(id)}/feedback`, { method: "PUT", body: json(body) }),
  deleteCase: (id: string) => request<{ ok: boolean }>(`/api/cases/${encodeURIComponent(id)}`, { method: "DELETE" }),
  exportCsv: (filters: CaseFilters = {}) => request<Blob>(`/api/cases/export.csv?${toQuery(filters)}`),

  patients: (q?: string) => request<{ patients: PatientRow[] }>(`/api/patients${q ? `?q=${encodeURIComponent(q)}` : ""}`),
  timeline: (id: number) =>
    request<{ patient: PatientRef; cases: CaseSummary[]; latest_change: number | null }>(`/api/patients/${id}/timeline`),

  dashboard: (days = 30) => request<Dashboard>(`/api/dashboard?days=${days}`),
  audit: () => request<{ events: { action: string; target: string | null; ip: string; created_at: string }[] }>("/api/audit"),

  keys: () => request<{ keys: ApiKey[] }>("/api/keys"),
  createKey: (name: string) => request<ApiKey>("/api/keys", { method: "POST", body: json({ name }) }),
  revokeKey: (id: number) => request<{ ok: boolean }>(`/api/keys/${id}`, { method: "DELETE" }),
};
