"""
OralAID inference engine.

One place that loads the validated checkpoints and produces everything the
product shows for a case: suspicion probability, risk tier, Grad-CAM, LIME,
lesion localisation and image-quality checks.
"""
import logging
import os
import threading
import time
from functools import lru_cache

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image, ImageOps
from torchvision.models import efficientnet_b4

from .registry import MODELS, download_weights, find_weights

log = logging.getLogger("oralaid.inference")

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Probability of "Suspicious" above which we flag a case. Chosen on the held-out
# set to favour sensitivity, because a missed lesion costs more than a referral.
SUSPICION_THRESHOLD = float(os.environ.get("ORALAID_THRESHOLD", "0.30"))
HIGH_RISK_THRESHOLD = float(os.environ.get("ORALAID_HIGH_THRESHOLD", "0.55"))

BLUR_THRESHOLD = 8.0
SOFT_THRESHOLD = 12.0

RISK_TIERS = {
    "low": {
        "label": "Low risk",
        "finding": "No suspicious features detected",
        "color": "#10b981",
        "recommendation": "Routine recall. Re-screen at the next scheduled visit or sooner if symptoms change.",
    },
    "moderate": {
        "label": "Moderate risk",
        "finding": "Possible suspicious features",
        "color": "#f59e0b",
        "recommendation": "Clinical re-examination within 2–3 weeks. Consider specialist review if the area persists.",
    },
    "high": {
        "label": "High risk",
        "finding": "Suspicious oral lesion features",
        "color": "#ef4444",
        "recommendation": "Refer to an oral medicine / maxillofacial specialist for evaluation and possible biopsy.",
    },
}


class OralNetB4(nn.Module):
    """Matches the layer layout of final_efficientnet_b4_oral_cancer.pth exactly."""

    def __init__(self, num_classes=2, dropout_rate=0.4):
        super().__init__()
        self.backbone = efficientnet_b4(weights=None)
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate * 0.6),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate * 0.4),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.backbone(x)


def _to_tensor(image, size):
    t = T.Compose([T.Resize((size, size)), T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)])
    return t(image).unsqueeze(0)


def load_image(path_or_file):
    image = Image.open(path_or_file)
    image = ImageOps.exif_transpose(image)
    return image.convert("RGB")


class InferenceEngine:
    def __init__(self, device=None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        torch.set_num_threads(max(1, int(os.environ.get("ORALAID_TORCH_THREADS", os.cpu_count() or 2))))
        self._lock = threading.Lock()
        self.classifier = None
        self.status = {}
        self._load_classifier()

    # ── loading ────────────────────────────────────────────────────────────

    def _load_classifier(self):
        path = find_weights("oralnet-b4") or download_weights("oralnet-b4")
        if not path:
            self.status["oralnet-b4"] = "missing weights"
            log.error("OralNet-B4 weights not found. Set ORALAID_MODEL_DIR or ORALAID_CLASSIFIER_URL.")
            return
        ckpt = torch.load(path, map_location=self.device, weights_only=True)  # no pickle code execution
        state = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
        model = OralNetB4(num_classes=2)
        model.load_state_dict(state, strict=True)  # strict: a silent partial load is how predictions go wrong
        self.classifier = model.to(self.device).eval()
        self.status["oralnet-b4"] = f"loaded from {path.name}"
        log.info("OralNet-B4 loaded from %s on %s", path, self.device)

    @property
    def ready(self):
        return self.classifier is not None

    # ── classification ─────────────────────────────────────────────────────

    @torch.no_grad()
    def predict_proba(self, image):
        """Multi-scale + flip TTA. Returns P(suspicious) and the per-view spread."""
        views = []
        for size in (300, 380):
            x = _to_tensor(image, size).to(self.device)
            batch = torch.cat([x, x.flip(3), x.flip(2)])
            views.append(torch.softmax(self.classifier(batch), dim=1)[:, 1])
        probs = torch.cat(views).cpu().numpy()
        return float(probs.mean()), float(probs.std())

    @staticmethod
    def risk_tier(p):
        if p >= HIGH_RISK_THRESHOLD:
            return "high"
        if p >= SUSPICION_THRESHOLD:
            return "moderate"
        return "low"

    # ── explainability ─────────────────────────────────────────────────────

    def gradcam(self, image, size=380):
        """Grad-CAM on the last EfficientNet stage, for the 'Suspicious' logit."""
        target = self.classifier.backbone.features[-1]
        store = {}
        h1 = target.register_forward_hook(lambda m, i, o: store.__setitem__("act", o))
        h2 = target.register_full_backward_hook(lambda m, gi, go: store.__setitem__("grad", go[0]))
        try:
            x = _to_tensor(image, size).to(self.device).requires_grad_(True)
            self.classifier.zero_grad()
            logits = self.classifier(x)
            # Explain the suspicious-vs-normal margin so the map highlights evidence for suspicion.
            (logits[0, 1] - logits[0, 0]).backward()
            act, grad = store["act"].detach(), store["grad"].detach()
        finally:
            h1.remove()
            h2.remove()
        weights = grad.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * act).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=(size, size), mode="bilinear", align_corners=False)[0, 0].cpu().numpy()
        cam = cv2.GaussianBlur(cam, (0, 0), sigmaX=size / 60)
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam

    def lime(self, image, num_samples=None, n_segments=60, size=300, seed=0):
        """
        LIME with SLIC superpixels and a kernel-weighted ridge surrogate.
        Returns (segments, weights) where weights[i] is superpixel i's contribution
        to P(suspicious).
        """
        from skimage.segmentation import slic
        from sklearn.linear_model import Ridge

        num_samples = num_samples or int(os.environ.get("ORALAID_LIME_SAMPLES", "192"))
        rng = np.random.default_rng(seed)
        img = np.asarray(image.resize((size, size)), dtype=np.float32) / 255.0
        segments = slic(img, n_segments=n_segments, compactness=12, sigma=1, start_label=0)
        n = int(segments.max()) + 1
        fill = cv2.GaussianBlur(img, (0, 0), sigmaX=12)  # blurred fill is less out-of-distribution than grey

        masks = rng.integers(0, 2, size=(num_samples, n))
        masks[0] = 1
        mean = torch.tensor(IMAGENET_MEAN, device=self.device).view(1, 3, 1, 1)
        std = torch.tensor(IMAGENET_STD, device=self.device).view(1, 3, 1, 1)
        img_t = torch.from_numpy(img).permute(2, 0, 1).to(self.device)
        fill_t = torch.from_numpy(fill).permute(2, 0, 1).to(self.device)
        seg_t = torch.from_numpy(segments).to(self.device)

        preds = []
        with torch.no_grad():
            for start in range(0, num_samples, 16):
                m = torch.from_numpy(masks[start:start + 16]).to(self.device)
                pix = m[:, seg_t].unsqueeze(1).float()  # (B,1,H,W)
                batch = pix * img_t + (1 - pix) * fill_t
                batch = (batch - mean) / std
                preds.append(torch.softmax(self.classifier(batch), dim=1)[:, 1].cpu().numpy())
        preds = np.concatenate(preds)

        distances = 1 - masks.sum(axis=1) / n
        kernel = np.exp(-(distances ** 2) / 0.25 ** 2)
        surrogate = Ridge(alpha=1.0).fit(masks, preds, sample_weight=kernel)
        return segments, surrogate.coef_

    @staticmethod
    def localize(image, cam, size=512):
        """
        Weakly-supervised lesion localisation: Grad-CAM seeds a GrabCut
        segmentation so the outline follows real tissue edges instead of the
        coarse 12×12 activation grid.
        """
        img = np.asarray(image.resize((size, size)))
        cam = cv2.resize(cam, (size, size))
        seed = np.full((size, size), cv2.GC_PR_BGD, np.uint8)
        seed[cam < 0.15] = cv2.GC_BGD
        seed[cam > 0.5] = cv2.GC_PR_FGD
        seed[cam > 0.85] = cv2.GC_FGD
        if not (seed == cv2.GC_FGD).any():
            return np.zeros((size, size), np.uint8)
        bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
        try:
            cv2.grabCut(cv2.cvtColor(img, cv2.COLOR_RGB2BGR), seed, None, bgd, fgd, 4, cv2.GC_INIT_WITH_MASK)
        except cv2.error:
            return (cam > 0.6).astype(np.uint8)
        mask = np.isin(seed, (cv2.GC_FGD, cv2.GC_PR_FGD)).astype(np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))
        # Keep components that contain a strong activation peak.
        n, labels = cv2.connectedComponents(mask)
        keep = np.zeros_like(mask)
        for i in range(1, n):
            region = labels == i
            if cam[region].max() > 0.75 and region.mean() > 0.002:
                keep[region] = 1
        return keep

    # ── quality ────────────────────────────────────────────────────────────

    @staticmethod
    def quality(image):
        gray = cv2.cvtColor(np.asarray(image.resize((512, 512))), cv2.COLOR_RGB2GRAY)
        sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        brightness = float(gray.mean())
        # Thresholds calibrated on real intraoral photos (Laplacian variance at 512px):
        # median ~21, 5th percentile ~9. Mucosa is smooth, so values are naturally low.
        issues = []
        if sharpness < BLUR_THRESHOLD:
            issues.append("Image looks blurry. Hold the camera steady and refocus.")
        if brightness < 50:
            issues.append("Image is underexposed. Add light or use the flash.")
        elif brightness > 215:
            issues.append("Image is overexposed. Reduce glare or flash intensity.")
        w, h = image.size
        if min(w, h) < 224:
            issues.append("Low resolution. Capture at least 640px on the short side.")
        score = 100
        score -= 35 if sharpness < BLUR_THRESHOLD else (10 if sharpness < SOFT_THRESHOLD else 0)
        score -= 25 if (brightness < 50 or brightness > 215) else 0
        score -= 20 if min(w, h) < 224 else 0
        return {
            "score": max(score, 0),
            "sharpness": round(sharpness, 1),
            "brightness": round(brightness, 1),
            "resolution": f"{w}×{h}",
            "issues": issues,
        }

    # ── full pipeline ──────────────────────────────────────────────────────

    def analyze(self, image, out_dir, case_id, with_lime=False):
        if not self.ready:
            raise RuntimeError("Classifier weights are not loaded")
        t0 = time.time()
        with self._lock:
            p, spread = self.predict_proba(image)
            cam = self.gradcam(image)
            lime = self.lime(image) if with_lime else None
        mask = self.localize(image, cam)
        tier = self.risk_tier(p)

        os.makedirs(out_dir, exist_ok=True)
        base = np.asarray(image.resize((512, 512)))
        files = {"original": _save(base, out_dir, f"{case_id}_original.jpg")}
        files["gradcam"] = _save(_overlay_heatmap(base, cv2.resize(cam, (512, 512))), out_dir, f"{case_id}_gradcam.jpg")
        files["heatmap"] = _save(_colorize(cv2.resize(cam, (512, 512))), out_dir, f"{case_id}_heatmap.png")

        files["segmentation"] = _save(_overlay_mask(base, mask), out_dir, f"{case_id}_segmentation.jpg")
        lesion_ratio = float(mask.mean())
        if lime is not None:
            files["lime"] = _save(_overlay_lime(base, *lime), out_dir, f"{case_id}_lime.jpg")

        # Where the classifier attends: the share of the image in the top Grad-CAM band.
        attention_ratio = float((cam > 0.6).mean())

        return {
            "probability": round(p, 4),
            "uncertainty": round(spread, 4),
            "confidence": round(max(p, 1 - p), 4),
            "tier": tier,
            **RISK_TIERS[tier],
            "threshold": SUSPICION_THRESHOLD,
            "region_area_ratio": round(lesion_ratio, 4),
            "attention_area_ratio": round(attention_ratio, 4),
            "quality": self.quality(image),
            "files": files,
            "models": {"classifier": "oralnet-b4", "localizer": "cam-grabcut"},
            "inference_ms": int((time.time() - t0) * 1000),
        }

    def explain_lime(self, image, out_dir, case_id):
        with self._lock:
            segments, weights = self.lime(image)
        base = np.asarray(image.resize((512, 512)))
        name = _save(_overlay_lime(base, segments, weights), out_dir, f"{case_id}_lime.jpg")
        top = np.argsort(weights)[::-1][:5]
        return {"file": name, "top_regions": [{"segment": int(i), "weight": round(float(weights[i]), 4)} for i in top]}

    def info(self):
        return {
            "device": str(self.device),
            "ready": self.ready,
            "status": self.status,
            "threshold": SUSPICION_THRESHOLD,
            "high_risk_threshold": HIGH_RISK_THRESHOLD,
            "models": list(MODELS.values()),
        }


# ── rendering helpers ──────────────────────────────────────────────────────

def _save(rgb, out_dir, name):
    path = os.path.join(out_dir, name)
    bgr = cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, bgr, [cv2.IMWRITE_JPEG_QUALITY, 90] if name.endswith(".jpg") else [])
    return name


def _colorize(cam):
    heat = cv2.applyColorMap((cam * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    return cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)


def _overlay_heatmap(base, cam):
    heat = _colorize(cam).astype(np.float32)
    alpha = (0.15 + 0.45 * cam)[..., None]  # weak regions stay mostly visible
    return (base * (1 - alpha) + heat * alpha).astype(np.uint8)


def _overlay_mask(base, mask):
    mask = cv2.resize(mask, (base.shape[1], base.shape[0]), interpolation=cv2.INTER_NEAREST)
    out = base.copy()
    tint = np.array([239, 68, 68], dtype=np.float32)
    out[mask > 0] = (out[mask > 0] * 0.55 + tint * 0.45).astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, contours, -1, (255, 255, 255), 2)
    return out


def _overlay_lime(base, segments, weights):
    from skimage.segmentation import find_boundaries

    seg = cv2.resize(segments.astype(np.int32), (base.shape[1], base.shape[0]), interpolation=cv2.INTER_NEAREST)
    out = base.astype(np.float32).copy()
    scale = np.abs(weights).max() + 1e-8
    k = max(1, int(len(weights) * 0.2))
    order = np.argsort(weights)
    pos = [i for i in order[::-1][:k] if weights[i] > 0]  # push toward "Suspicious"
    neg = [i for i in order[:k] if weights[i] < 0]  # push toward "Non-suspicious"
    for i in pos:
        a = 0.25 + 0.35 * weights[i] / scale
        out[seg == i] = out[seg == i] * (1 - a) + np.array([239, 68, 68]) * a
    for i in neg:
        a = 0.2 + 0.3 * -weights[i] / scale
        out[seg == i] = out[seg == i] * (1 - a) + np.array([16, 185, 129]) * a
    out[find_boundaries(seg, mode="thin")] = (255, 255, 255)
    return out.astype(np.uint8)


@lru_cache(maxsize=1)
def get_engine():
    return InferenceEngine()
