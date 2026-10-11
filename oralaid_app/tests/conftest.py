import io
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

_tmp = tempfile.mkdtemp(prefix="oralaid-test-")
os.environ.setdefault("ORALAID_DATA_DIR", _tmp)
os.environ.setdefault("ORALAID_SECRET_KEY", "test-secret-key-not-for-production-use-0123456789")
os.environ.setdefault("ORALAID_SKIP_MODEL", "1")
os.environ.setdefault("ORALAID_SERVE_FRONTEND", "0")


class FakeEngine:
    """Deterministic stand-in so API tests don't need the 224 MB checkpoint."""

    ready = True
    status = {"oralnet-b4": "fake"}

    def __init__(self, probability=0.72):
        self.probability = probability

    def analyze(self, image, out_dir, case_id, with_lime=False):
        from inference.engine import RISK_TIERS, InferenceEngine

        files = {}
        for kind in ("original", "gradcam", "heatmap", "segmentation") + (("lime",) if with_lime else ()):
            name = f"{case_id}_{kind}.jpg"
            image.resize((64, 64)).save(Path(out_dir) / name)
            files[kind] = name
        tier = InferenceEngine.risk_tier(self.probability)
        return {"probability": self.probability, "uncertainty": 0.02, "confidence": self.probability,
                "tier": tier, **RISK_TIERS[tier], "threshold": 0.3, "region_area_ratio": 0.1,
                "attention_area_ratio": 0.1, "quality": InferenceEngine.quality(image), "files": files,
                "models": {"classifier": "oralnet-b4"}, "inference_ms": 5}

    def explain_lime(self, image, out_dir, case_id):
        name = f"{case_id}_lime.jpg"
        image.resize((64, 64)).save(Path(out_dir) / name)
        return {"file": name, "top_regions": [{"segment": 1, "weight": 0.1}]}


def make_oral_jpeg(size=(640, 480)):
    """Pink mucosa with white 'teeth' band: passes the oral-image heuristic."""
    arr = np.zeros((size[1], size[0], 3), np.uint8)
    arr[:] = (200, 110, 120)
    arr[: size[1] // 4] = (235, 230, 220)
    rng = np.random.default_rng(0)
    arr = np.clip(arr.astype(int) + rng.integers(-25, 25, arr.shape), 0, 255).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="JPEG")
    return buf.getvalue()


@pytest.fixture()
def client():
    from fastapi.testclient import TestClient

    import app_fastapi
    import security

    security.reset_rate_limits()
    app_fastapi.app.dependency_overrides[app_fastapi.require_engine] = lambda: FakeEngine()
    with TestClient(app_fastapi.app) as c:
        yield c
    app_fastapi.app.dependency_overrides.clear()


@pytest.fixture()
def auth(client):
    import uuid

    email = f"dr-{uuid.uuid4().hex[:8]}@clinic.example.com"
    r = client.post("/api/auth/register", json={"email": email, "password": "Screening2026", "full_name": "Test Clinician"})
    assert r.status_code == 201, r.text
    return {"Authorization": f"Bearer {r.json()['token']}"}


@pytest.fixture()
def oral_jpeg():
    return make_oral_jpeg()
