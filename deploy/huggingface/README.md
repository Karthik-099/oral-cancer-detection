---
title: OralAID API
colorFrom: green
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
---

OralAID backend (FastAPI + OralNet-B4). Deployed automatically from the `prod`
branch by `.github/workflows/deploy.yml`. Configure secrets under Settings:
ORALAID_SECRET_KEY, ORALAID_CORS_ORIGINS, DATABASE_URL, ORALAID_CLASSIFIER_URL,
ORALAID_CLASSIFIER_SHA256, HF_TOKEN.
