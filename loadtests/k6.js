// k6 load test with SLO thresholds.
//   k6 run -e BASE_URL=http://localhost:8000 loadtests/k6.js
//   k6 run -e BASE_URL=https://staging.example.com -e PROFILE=stress loadtests/k6.js
import http from "k6/http";
import { check, sleep } from "k6";
import { uuidv4 } from "https://jslib.k6.io/k6-utils/1.4.0/index.js";

const BASE = __ENV.BASE_URL || "http://localhost:8000";

const PROFILES = {
  smoke: { vus: 2, duration: "30s" },
  load: { stages: [{ duration: "1m", target: 20 }, { duration: "3m", target: 20 }, { duration: "1m", target: 0 }] },
  stress: { stages: [{ duration: "2m", target: 50 }, { duration: "3m", target: 100 }, { duration: "2m", target: 0 }] },
  spike: { stages: [{ duration: "20s", target: 5 }, { duration: "10s", target: 150 }, { duration: "1m", target: 150 }, { duration: "20s", target: 0 }] },
};

export const options = {
  ...PROFILES[__ENV.PROFILE || "smoke"],
  thresholds: {
    http_req_failed: ["rate<0.01"],
    "http_req_duration{kind:read}": ["p(95)<500"],
    "http_req_duration{kind:health}": ["p(99)<300"],
    checks: ["rate>0.99"],
  },
};

export function setup() {
  const res = http.post(`${BASE}/api/auth/register`, JSON.stringify({
    email: `k6-${uuidv4().slice(0, 10)}@example.com`, password: "LoadTest2026x", full_name: "k6",
  }), { headers: { "Content-Type": "application/json" } });
  return { token: res.json("token") };
}

export default function ({ token }) {
  const auth = { headers: { Authorization: `Bearer ${token}` }, tags: { kind: "read" } };
  check(http.get(`${BASE}/api/health`, { tags: { kind: "health" } }), { "health 200": (r) => r.status === 200 });
  check(http.get(`${BASE}/api/dashboard`, auth), { "dashboard 200": (r) => r.status === 200 });
  check(http.get(`${BASE}/api/cases?limit=25`, auth), { "cases 200": (r) => r.status === 200 });
  check(http.get(`${BASE}/api/cases`, { responseCallback: http.expectedStatuses(401) }), { "unauthenticated is 401": (r) => r.status === 401 });
  sleep(1);
}
