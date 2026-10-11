#!/usr/bin/env bash
# Quick Apache Bench baseline against read endpoints.
#   ./loadtests/ab.sh http://localhost:8000 [requests] [concurrency]
set -euo pipefail
BASE="${1:-http://localhost:8000}"
N="${2:-2000}"
C="${3:-50}"
command -v ab >/dev/null || { echo "install apache2-utils (ab)"; exit 1; }

TOKEN=$(curl -fsS -X POST "$BASE/api/auth/register" -H 'Content-Type: application/json' \
  -d "{\"email\":\"ab-$RANDOM$RANDOM@example.com\",\"password\":\"LoadTest2026x\",\"full_name\":\"ab\"}" \
  | python3 -c 'import sys,json; print(json.load(sys.stdin)["token"])')

echo "== /api/health  ($N requests, concurrency $C)"
ab -q -n "$N" -c "$C" "$BASE/api/health" | grep -E "Requests per second|Time per request|Failed requests|95%|99%"
echo "== /api/cases (authenticated)"
ab -q -n "$N" -c "$C" -H "Authorization: Bearer $TOKEN" "$BASE/api/cases?limit=25" | grep -E "Requests per second|Time per request|Failed requests|95%|99%"
echo "== /api/dashboard (authenticated)"
ab -q -n "$N" -c "$C" -H "Authorization: Bearer $TOKEN" "$BASE/api/dashboard" | grep -E "Requests per second|Time per request|Failed requests|95%|99%"
