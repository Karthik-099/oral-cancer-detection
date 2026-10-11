FROM python:3.10-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    TORCH_HOME=/app/.torch_cache \
    OMP_NUM_THREADS=4

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 libsm6 libxext6 libxrender-dev \
    libpq-dev gcc curl \
    && rm -rf /var/lib/apt/lists/*

# ── Builder ──────────────────────────────────────────────────────────────────
FROM base AS builder
WORKDIR /build
COPY requirements.txt .
RUN pip install --prefix=/install -r requirements.txt \
    && pip install --prefix=/install psycopg2-binary==2.9.9

# ── Runtime ──────────────────────────────────────────────────────────────────
FROM base AS runtime
WORKDIR /app

COPY --from=builder /install /usr/local
COPY . .

RUN mkdir -p uploads static/results static/annotations logs models .torch_cache \
    && addgroup --system oralaid \
    && adduser --system --ingroup oralaid oralaid \
    && chown -R oralaid:oralaid /app

USER oralaid

EXPOSE 5000

HEALTHCHECK --interval=30s --timeout=10s --start-period=90s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:5000/metrics')"

# 4 workers, 2 threads each — optimal for CPU inference
CMD ["gunicorn", \
     "--bind", "0.0.0.0:5000", \
     "--workers", "4", \
     "--threads", "2", \
     "--worker-class", "gthread", \
     "--timeout", "120", \
     "--keep-alive", "5", \
     "--access-logfile", "-", \
     "--error-logfile", "-", \
     "app:app"]
