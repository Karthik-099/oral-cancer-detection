"""
Realistic load profile for the OralAID API.

    locust -f loadtests/locustfile.py --host http://localhost:8000
    locust -f loadtests/locustfile.py --host https://staging.example.com \
        --headless -u 50 -r 5 -t 5m --csv results/load

Most traffic is browsing (dashboard, case lists); a smaller share uploads
images, which is the expensive path. Each simulated clinician registers once.
"""
import io
import os
import random
import uuid

import numpy as np
from locust import HttpUser, between, events, task
from PIL import Image


def _oral_jpeg():
    arr = np.full((480, 640, 3), (200, 110, 120), np.uint8)
    arr[:120] = (235, 230, 220)
    arr = np.clip(arr.astype(int) + np.random.randint(-25, 25, arr.shape), 0, 255).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="JPEG", quality=85)
    return buf.getvalue()


IMAGE = _oral_jpeg()
ANALYZE_WEIGHT = int(os.environ.get("ORALAID_LOAD_ANALYZE_WEIGHT", "1"))


class Clinician(HttpUser):
    wait_time = between(1, 4)

    def on_start(self):
        email = f"load-{uuid.uuid4().hex[:10]}@example.com"
        r = self.client.post("/api/auth/register", json={
            "email": email, "password": "LoadTest2026x", "full_name": "Load Test"}, name="/api/auth/register")
        self.headers = {"Authorization": f"Bearer {r.json().get('token', '')}"} if r.ok else {}
        self.case_ids = []

    @task(6)
    def dashboard(self):
        self.client.get("/api/dashboard", headers=self.headers, name="/api/dashboard")

    @task(4)
    def list_cases(self):
        self.client.get(f"/api/cases?limit=25&tier={random.choice(['', 'high', 'moderate', 'low'])}",
                        headers=self.headers, name="/api/cases")

    @task(2)
    def view_case(self):
        if self.case_ids:
            self.client.get(f"/api/cases/{random.choice(self.case_ids)}", headers=self.headers, name="/api/cases/[id]")

    @task(ANALYZE_WEIGHT)
    def analyze(self):
        with self.client.post("/api/cases", headers=self.headers, name="/api/cases [analyze]",
                              files={"files": ("load.jpg", IMAGE, "image/jpeg")},
                              data={"site": "Buccal mucosa"}, catch_response=True) as r:
            if r.status_code == 201:
                self.case_ids.append(r.json()["id"])
                r.success()
            elif r.status_code in (429, 503):
                r.success()  # back-pressure is expected behaviour under load, not an error
            else:
                r.failure(f"{r.status_code}: {r.text[:120]}")

    @task(3)
    def health(self):
        self.client.get("/api/health", name="/api/health")


@events.quitting.add_listener
def _gate(environment, **_):
    """Fail the run (non-zero exit) when SLOs are missed, so CI can gate on it."""
    stats = environment.stats.total
    if stats.fail_ratio > 0.01:
        environment.process_exit_code = 1
    elif stats.get_response_time_percentile(0.95) > float(os.environ.get("ORALAID_P95_MS", "2000")):
        environment.process_exit_code = 1
