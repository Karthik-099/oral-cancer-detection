# Backend-only image for a Hugging Face Docker Space (the website is on Netlify).
# The deploy workflow copies this file, README.md and the API source into the Space.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    ORALAID_ENV=production \
    ORALAID_SERVE_FRONTEND=0 \
    ORALAID_DATA_DIR=/tmp/oralaid \
    ORALAID_MODEL_DIR=/tmp/oralaid/models \
    ORALAID_TORCH_THREADS=2 \
    HOME=/tmp \
    PORT=7860

RUN apt-get update && apt-get install -y --no-install-recommends libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd -m -u 1000 user

COPY requirements-api.txt /tmp/requirements-api.txt
RUN pip install -r /tmp/requirements-api.txt --extra-index-url https://download.pytorch.org/whl/cpu

WORKDIR /app
COPY --chown=user app_fastapi.py security.py ./
COPY --chown=user inference ./inference
COPY --chown=user models ./models

USER user
EXPOSE 7860
CMD ["sh", "-c", "uvicorn app_fastapi:app --host 0.0.0.0 --port ${PORT} --proxy-headers --forwarded-allow-ips='*' --no-server-header"]
