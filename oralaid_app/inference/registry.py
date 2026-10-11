"""
Model registry: where checkpoints live and what we know about them.

Metrics come from `python -m inference.evaluate` on data/original +
data/metadata.csv (specialist diagnosis). Re-run after retraining and update.
"""
import logging
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

# Searched in order. ORALAID_MODEL_DIR wins so deployments can mount weights anywhere.
MODEL_DIRS = [
    Path(os.environ["ORALAID_MODEL_DIR"]) if os.environ.get("ORALAID_MODEL_DIR") else None,
    REPO_ROOT / "models",
    REPO_ROOT / "oralaid_app" / "models",
    REPO_ROOT / "src" / "data_processing" / "tool_withpolygon" / "modularize_tool" / "models",
]

MODELS = {
    "oralnet-b4": {
        "id": "oralnet-b4",
        "name": "OralNet-B4",
        "task": "classification",
        "status": "production",
        "architecture": "EfficientNet-B4 backbone, 3-layer MLP head (BatchNorm + dropout)",
        "files": ["final_efficientnet_b4_oral_cancer.pth", "trained_efficientnet_b4_oral_cancer.pth"],
        "env_url": "ORALAID_CLASSIFIER_URL",
        "sha256": "6177f31b15f4d83b118485ee8683b5cb3914f5d26bb061a62b81fa2da96ff3a0",
        "input": "300×300 and 380×380 RGB (multi-scale TTA with flips)",
        "classes": ["Non-suspicious", "Suspicious"],
        "params_m": 18.4,
        "metrics": {
            "val_accuracy": 0.768,
            "auc": 0.826,
            "sensitivity": 0.705,
            "specificity": 0.802,
            "ppv": 0.768,
            "npv": 0.744,
            "high_risk_ppv": 0.90,
            "eval_set": "val_accuracy: held-out split at training time. Other metrics: internal evaluation on "
                        "400 specialist-labelled images at threshold 0.30, which may overlap training data. "
                        "Independent external validation is pending.",
        },
        "notes": "Binary screening model (suspicious vs non-suspicious). Not a diagnostic device.",
    },
    "cam-grabcut": {
        "id": "cam-grabcut",
        "name": "Region Localizer",
        "task": "segmentation",
        "status": "production",
        "architecture": "Grad-CAM seeds → GrabCut graph-cut refinement → peak-filtered components",
        "files": [],
        "input": "512×512 RGB + OralNet-B4 activation map",
        "classes": ["Background", "Region of interest"],
        "metrics": {},
        "notes": "Weakly supervised: outlines the tissue that drove the prediction. Not a pixel-trained lesion mask.",
    },
    "lesionseg-r34": {
        "id": "lesionseg-r34",
        "name": "U-Net R34 (legacy)",
        "task": "segmentation",
        "status": "retired",
        "architecture": "U-Net decoder, ResNet-34 encoder",
        "files": ["unet_segmenter.pth"],
        "input": "512×512 RGB",
        "classes": ["Background", "Lesion"],
        "params_m": 24.4,
        "metrics": {},
        "notes": "Trained on whole-image pseudo-masks, so it marks the entire photo. Needs pixel-level annotations to retrain.",
    },
    "explain-gradcam": {
        "id": "explain-gradcam",
        "name": "Grad-CAM",
        "task": "explainability",
        "status": "production",
        "architecture": "Gradient-weighted class activation on the final EfficientNet stage",
        "files": [],
        "input": "Shares OralNet-B4",
        "classes": [],
        "metrics": {},
        "notes": "Shows where the classifier looked when making its call.",
    },
    "explain-lime": {
        "id": "explain-lime",
        "name": "LIME superpixels",
        "task": "explainability",
        "status": "production",
        "architecture": "SLIC superpixels + kernel-weighted ridge surrogate",
        "files": [],
        "input": "Shares OralNet-B4",
        "classes": [],
        "metrics": {},
        "notes": "Ranks image regions by how much they push the prediction toward 'Suspicious'.",
    },
    "mobilenet-v2-legacy": {
        "id": "mobilenet-v2-legacy",
        "name": "MobileNetV2 (legacy)",
        "task": "classification",
        "status": "retired",
        "architecture": "MobileNetV2, linear head",
        "files": ["mobilenet_classifier.pth"],
        "input": "224×224 RGB",
        "classes": ["Non-suspicious", "Suspicious"],
        "params_m": 2.2,
        "metrics": {"auc": 0.41, "eval_set": "same held-out set"},
        "notes": "Failed validation (below chance). Kept for audit only; never served.",
    },
}


def find_weights(model_id):
    """Return the first existing checkpoint path for a model, or None."""
    spec = MODELS[model_id]
    for d in MODEL_DIRS:
        if d is None:
            continue
        for f in spec["files"]:
            p = d / f
            # Git LFS pointer files are tiny text files; skip them.
            if p.exists() and p.stat().st_size > 1_000_000:
                return p
    return None


def download_weights(model_id):
    """
    Fetch weights from the HTTPS URL in the model's env var. If the matching
    *_SHA256 variable is set (recommended), the file must match it exactly.
    """
    import hashlib
    import tempfile
    import urllib.request

    spec = MODELS[model_id]
    env = spec.get("env_url", "")
    url = os.environ.get(env, "")
    if not url:
        return None
    if not url.startswith("https://"):
        raise ValueError(f"{env} must be an https:// URL")
    expected = os.environ.get(env.replace("_URL", "_SHA256"), spec.get("sha256", "")).lower()

    target_dir = MODEL_DIRS[0] or (REPO_ROOT / "models")
    if not os.access(target_dir if target_dir.exists() else target_dir.parent, os.W_OK):
        target_dir = Path(os.environ.get("ORALAID_DATA_DIR", tempfile.gettempdir())) / "models"
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / spec["files"][0]

    log = logging.getLogger("oralaid.inference")
    log.info("downloading %s weights", model_id)
    tmp = target.with_suffix(".part")
    digest = hashlib.sha256()
    with urllib.request.urlopen(url, timeout=60) as r, open(tmp, "wb") as f:  # noqa: S310  # nosec B310
        for chunk in iter(lambda: r.read(1 << 20), b""):
            digest.update(chunk)
            f.write(chunk)
    if expected and digest.hexdigest() != expected:
        tmp.unlink(missing_ok=True)
        raise ValueError(f"{model_id} checksum mismatch; refusing to load downloaded weights")
    if not expected:
        log.warning("%s downloaded without checksum verification; set %s", model_id, env.replace("_URL", "_SHA256"))
    tmp.replace(target)
    return target
