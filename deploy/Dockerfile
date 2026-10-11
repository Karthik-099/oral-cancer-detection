# OralAID: one image that serves the API and the built web app.
# Used for the hosted backend, private-VPC and on-premise installs.
#
#   docker build -f deploy/Dockerfile -t oralaid .
#   docker run -p 8000:8000 -v $PWD/models:/models:ro -e ORALAID_SECRET_KEY=... oralaid

# ── 1. web app ─────────────────────────────────────────────────────────────
FROM node:22-alpine AS web
WORKDIR /web
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY frontend/ ./
RUN npm run build

# ── 2. python deps ─────────────────────────────────────────────────────────
FROM python:3.11-slim AS deps
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
COPY oralaid_app/requirements-api.txt /tmp/
RUN pip install --prefix=/install -r /tmp/requirements-api.txt \
      --extra-index-url https://download.pytorch.org/whl/cpu

# ── 3. runtime ─────────────────────────────────────────────────────────────
FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    ORALAID_ENV=production \
    ORALAID_DATA_DIR=/data \
    ORALAID_MODEL_DIR=/models \
    ORALAID_FRONTEND_DIST=/app/frontend/dist \
    OMP_NUM_THREADS=4 \
    HOME=/tmp \
    XDG_CACHE_HOME=/tmp/.cache \
    PORT=8000

RUN apt-get update && apt-get install -y --no-install-recommends libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --system --uid 10001 --home /app oralaid \
    && mkdir -p /data /models && chown oralaid /data /models

COPY --from=deps /install /usr/local
WORKDIR /app/oralaid_app
COPY --chown=oralaid oralaid_app/app_fastapi.py oralaid_app/security.py ./
COPY --chown=oralaid oralaid_app/inference ./inference
COPY --chown=oralaid oralaid_app/models/validator.py ./models/
COPY --from=web --chown=oralaid /web/dist /app/frontend/dist

USER oralaid
EXPOSE 8000
VOLUME ["/data"]

HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
  CMD python -c "import urllib.request,os; urllib.request.urlopen(f'http://127.0.0.1:{os.environ[\"PORT\"]}/api/health', timeout=4)"

# One worker per container keeps the model in memory once; scale with replicas.
CMD ["sh", "-c", "uvicorn app_fastapi:app --host 0.0.0.0 --port ${PORT} --proxy-headers --forwarded-allow-ips='*' --no-server-header"]
