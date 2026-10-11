"""Runs the real model when weights are available (skipped in lightweight CI)."""
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from inference.registry import find_weights

pytestmark = pytest.mark.skipif(find_weights("oralnet-b4") is None, reason="model weights not available")


@pytest.fixture(scope="module")
def engine():
    from inference.engine import InferenceEngine

    return InferenceEngine(device="cpu")


@pytest.fixture()
def image():
    rng = np.random.default_rng(3)
    arr = np.full((400, 400, 3), (190, 100, 110), np.uint8)
    arr[150:250, 150:250] = (240, 235, 225)
    arr = np.clip(arr.astype(int) + rng.integers(-20, 20, arr.shape), 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def test_probability_is_valid_and_deterministic(engine, image):
    p1, s1 = engine.predict_proba(image)
    p2, _ = engine.predict_proba(image)
    assert 0 <= p1 <= 1 and s1 >= 0
    assert abs(p1 - p2) < 1e-6


def test_gradcam_shape_and_range(engine, image):
    cam = engine.gradcam(image)
    assert cam.shape == (380, 380)
    assert cam.min() >= 0 and cam.max() <= 1


def test_analyze_writes_outputs(engine, image, tmp_path):
    r = engine.analyze(image, str(tmp_path), "t")
    for name in r["files"].values():
        assert (Path(tmp_path) / name).exists()
    assert r["tier"] in ("low", "moderate", "high")


def test_lime_returns_weight_per_superpixel(engine, image):
    segments, weights = engine.lime(image, num_samples=32)
    assert len(weights) == segments.max() + 1
