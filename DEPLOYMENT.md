# Deploying OralAID

OralAID has two parts:

| Part | What it is | Where it runs |
|---|---|---|
| Web app | React SPA in `frontend/` | Netlify (global CDN) |
| API + model | FastAPI + PyTorch in `oralaid_app/`, Postgres | Any Docker host: a VPS, Render, Railway, Fly, or your own server |

Netlify only serves static files, so the API needs a Docker host. Netlify proxies `/api/*` and `/files/*` to it, so the browser only ever talks to your domain.

```
browser ──> your-domain (Netlify CDN) ──/api, /files──> API container ──> Postgres (internal Docker network)
                    └── static assets, cached at the edge
```

## 1. API server (Docker + Postgres)

On the server (2 vCPU / 4 GB RAM is enough for a clinic):

```bash
git clone <repo> && cd oral-cancer-detection
mkdir -p models && cp /path/to/final_efficientnet_b4_oral_cancer.pth models/   # never committed
cp deploy/.env.example deploy/.env
# Fill in deploy/.env. Generate secrets with:
python3 -c "import secrets; print(secrets.token_urlsafe(48))"
docker compose -f deploy/docker-compose.yml --env-file deploy/.env up -d --build
curl localhost:8000/api/health
```

- Postgres has **no published ports**. It sits on an `internal: true` Docker network that only the API container can reach.
- It's tuned small (512 MB RAM cap). It stores case metadata only, about 1–2 KB per case, so 100k cases stay well under 1 GB.
- Images live in the `oralaid-data` volume. A nightly `pg_dump` runs into the `oralaid-backups` volume and keeps 7 days.
- Put HTTPS in front with Caddy, nginx or your provider's load balancer, e.g. `api.yourdomain.com -> :8000`.

Instead of copying the weights, you can set `ORALAID_CLASSIFIER_URL` (an HTTPS signed URL) and `ORALAID_CLASSIFIER_SHA256`. The download is refused if the checksum doesn't match.

**Render alternative:** `render.yaml` is a blueprint (API + managed Postgres) that deploys from the `prod` branch.

## 2. Web app on Netlify

1. In Netlify, choose "Add new site", then "Import from Git" and pick this repo. `netlify.toml` sets everything: base `frontend`, build `npm ci && npm run build`, publish `dist`.
2. Under Site settings, set the production branch to **`prod`**.
3. Add these environment variables:
   - `ORALAID_API_ORIGIN` = `https://api.yourdomain.com`
   - `SITE_URL` = `https://yourdomain.com`
4. Under Domain management, add your domain. Netlify provisions HTTPS automatically.
5. On the API server, set `ORALAID_CORS_ORIGINS=https://yourdomain.com` in `deploy/.env`.

Every pull request gets a deploy preview. Pushes to `prod` go live.

### What Netlify serves

- `/assets/*` is cached for one year (`immutable`, hashed file names). `index.html` always revalidates.
- `/api/*` and `/files/*` are proxied and never cached. Clinical images are served through short-lived signed URLs.
- Security headers: HSTS, a strict CSP, `X-Frame-Options: DENY`, and a Permissions-Policy that allows only the camera, for capture.
- SEO: per-route titles and descriptions, canonical URLs, Open Graph tags, JSON-LD, `robots.txt` (blocks `/app`, `/api`, `/files`), and a `sitemap.xml` generated at build time. After launch, submit `https://yourdomain.com/sitemap.xml` in Google Search Console.

## 3. CI/CD (GitHub Actions)

| Workflow | Trigger | What it does |
|---|---|---|
| `ci.yml` | every push / PR | ruff lint, pytest (API, security, isolation tests), TypeScript, vitest, production build + bundle budget, Docker build, container smoke test, k6 smoke with SLO thresholds |
| `security.yml` | push / PR / weekly | gitleaks secret scan, bandit SAST, pip-audit, npm audit, CodeQL (Python + TS), Trivy image scan, OWASP ZAP API scan (DAST) |
| `deploy.yml` | push to `prod` | runs CI + security, then deploys the API (Render hook) and the web app (Netlify CLI), then post-deploy health, header and SEO checks plus a k6 smoke run |
| `load-test.yml` | manual | k6 (smoke / load / stress / spike) and Locust mixed browse + analyse against a staging URL you choose |

Secrets for `deploy.yml`: `NETLIFY_AUTH_TOKEN`, `NETLIFY_SITE_ID`, optionally `RENDER_DEPLOY_HOOK_URL` and `PRODUCTION_URL`. Variables: `ORALAID_API_ORIGIN`, `SITE_URL`. If you use Netlify's Git integration instead, you can delete the `web` job.

## 4. Load testing locally

```bash
cd oralaid_app && ORALAID_RATE_LIMIT_SCALE=1000 uvicorn app_fastapi:app --port 8000 &
k6 run -e BASE_URL=http://localhost:8000 -e PROFILE=load loadtests/k6.js
locust -f loadtests/locustfile.py --host http://localhost:8000
./loadtests/ab.sh http://localhost:8000 2000 50
```

`ORALAID_RATE_LIMIT_SCALE` raises the per-IP limits so one load-generator IP isn't throttled. Use it only on staging.

## 5. Environment variables (API)

| Variable | Required | Purpose |
|---|---|---|
| `ORALAID_SECRET_KEY` | yes | signs sessions and image URLs |
| `ORALAID_CORS_ORIGINS` | yes in production | comma-separated site origins |
| `DATABASE_URL` | no | Postgres URL. Defaults to SQLite in `ORALAID_DATA_DIR` |
| `ORALAID_MODEL_DIR` | no | folder with `.pth` weights |
| `ORALAID_CLASSIFIER_URL` / `_SHA256` | no | download weights on start, checksum-verified |
| `ORALAID_THRESHOLD` / `ORALAID_HIGH_THRESHOLD` | no | risk tier cut-offs (default 0.30 / 0.55) |
| `ORALAID_ADMIN_EMAIL` / `_PASSWORD` | no | bootstrap an admin account |
| `ORALAID_PUBLIC_DOCS` | no | `1` exposes `/docs` and the OpenAPI schema in production |
| `ORALAID_SERVE_FRONTEND` | no | `0` disables serving the SPA from the API (use when Netlify hosts it) |

## 6. Models

`python -m inference.evaluate --patients 400` (from `oralaid_app/`) re-runs the evaluation. Update `inference/registry.py` with the new numbers after retraining. Weights, patient images and CSVs are gitignored and must never be committed.
