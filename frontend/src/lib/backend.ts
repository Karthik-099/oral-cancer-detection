import { useEffect, useState } from "react";
import { API_BASE } from "./api";

// The site works without a backend (on-device screening). Accounts, case
// history and the clinical app appear only when an API answers /api/health.
type Status = "checking" | "online" | "offline";
let cached: Promise<Status> | null = null;

export function checkBackend(): Promise<Status> {
  if (!cached) {
    cached = fetch(`${API_BASE}/api/health`, { headers: { Accept: "application/json" } })
      .then(async (r) => {
        if (!r.ok || !r.headers.get("content-type")?.includes("json")) return "offline" as const;
        const body = await r.json();
        return body?.status === "ok" ? ("online" as const) : ("offline" as const);
      })
      .catch(() => "offline" as const);
  }
  return cached;
}

export function useBackend() {
  const [status, setStatus] = useState<Status>("checking");
  useEffect(() => { checkBackend().then(setStatus); }, []);
  return status;
}
