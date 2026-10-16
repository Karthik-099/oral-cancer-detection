#!/usr/bin/env bash
# Pull the latest code and (re)start the OralAID stack. Used by hand and by
# the GitHub Actions deploy job (over SSH).
set -euo pipefail
APP_DIR="${APP_DIR:-$HOME/oral-cancer-detection}"
BRANCH="${BRANCH:-prod}"
cd "$APP_DIR"

git fetch --quiet origin "$BRANCH"
git checkout --quiet "$BRANCH"
git reset --quiet --hard "origin/$BRANCH"   # the VM never carries local code changes

[ -f deploy/.env ] || { echo "deploy/.env missing; run deploy/oracle/setup.sh first"; exit 1; }
ls models/*.pth >/dev/null 2>&1 || grep -q '^ORALAID_CLASSIFIER_URL=https' deploy/.env || \
  echo "WARNING: no model in ./models and no ORALAID_CLASSIFIER_URL; analyses will return 503"

docker compose -f deploy/docker-compose.yml --env-file deploy/.env up -d --build --remove-orphans
docker image prune -f >/dev/null

echo "Waiting for the API..."
for i in $(seq 1 60); do
  if curl -fsS http://127.0.0.1:8000/api/health 2>/dev/null | grep -q '"status":"ok"'; then
    curl -fsS http://127.0.0.1:8000/api/health; echo; echo "OralAID is up."; exit 0
  fi
  sleep 5
done
echo "API did not become healthy in time. Recent logs:"
docker compose -f deploy/docker-compose.yml --env-file deploy/.env logs --tail 80 oralaid
exit 1
