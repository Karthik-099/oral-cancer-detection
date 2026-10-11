"""
Oral Image Validator
Rejects non-oral cavity images before running the main classifier
Uses a lightweight heuristic + color/texture analysis
"""
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
from PIL import Image


_validator_model = None
_device = None

ORAL_TRANSFORM = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def _get_validator(device):
    global _validator_model, _device
    if _validator_model is None:
        _device = device
        base = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1)
        # Binary head: oral vs non-oral
        base.classifier = nn.Sequential(
            nn.Linear(576, 128), nn.Hardswish(), nn.Dropout(0.2), nn.Linear(128, 2)
        )
        _validator_model = base.to(device).eval()
    return _validator_model


def _analyze_image(image: Image.Image) -> dict:
    """
    Oral cavity images tend to have:
    - High red/pink channel values (mucosa, gums)
    - Moderate saturation
    - Presence of teeth (white/off-white regions)
    Returns extracted heuristic signals and an aggregate score.
    """
    img = np.array(image.resize((128, 128))).astype(np.float32)
    r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]

    # Pink/red mucosa ratio
    pink_mask = (r > 140) & (g < 160) & (b < 160) & (r > g) & (r > b)
    pink_ratio = pink_mask.mean()

    # White teeth regions
    white_mask = (r > 180) & (g > 180) & (b > 180)
    white_ratio = white_mask.mean()

    # Dark background (mouth interior)
    dark_mask = (r < 60) & (g < 60) & (b < 60)
    dark_ratio = dark_mask.mean()

    # Outdoor/non-oral scene cues
    green_mask = (g > (r * 1.1)) & (g > (b * 1.1)) & (g > 70)
    green_ratio = green_mask.mean()
    blue_mask = (b > (r * 1.1)) & (b > (g * 1.1)) & (b > 70)
    blue_ratio = blue_mask.mean()
    scene_ratio = max(green_ratio, blue_ratio)

    # Flat-color graphics/logos tend to have a few dominant colors
    quantized = (img.astype(np.uint8) // 32).reshape(-1, 3)
    _, counts = np.unique(quantized, axis=0, return_counts=True)
    top3_ratio = np.sort(counts)[-3:].sum() / counts.sum() if counts.size else 0.0

    score = (pink_ratio * 2.0 + white_ratio * 1.5 + dark_ratio * 0.5)
    return {
        'score': min(score, 1.0),
        'pink_ratio': float(pink_ratio),
        'white_ratio': float(white_ratio),
        'dark_ratio': float(dark_ratio),
        'scene_ratio': float(scene_ratio),
        'flat_color_ratio': float(top3_ratio),
    }


def is_oral_image(image: Image.Image, device, threshold: float = 0.15) -> tuple[bool, str]:
    """
    Returns (is_valid, reason_message)
    Uses color heuristic as primary check (fast, no extra model needed)
    """
    try:
        features = _analyze_image(image)
        score = features['score']

        # Check image is not too small
        w, h = image.size
        if w < 50 or h < 50:
            return False, "Image resolution too low. Please upload a clear intraoral photograph."

        # Check not grayscale/document
        img_np = np.array(image.resize((64, 64))).astype(np.float32)
        channel_std = np.std(img_np, axis=(0, 1))
        if channel_std.max() < 5:
            return False, "Image appears to be blank or uniform. Please upload a real intraoral photograph."

        # Reject obvious outdoor/scene photos (e.g., people/landscape selfies)
        if features['scene_ratio'] > 0.12 and features['pink_ratio'] < 0.18:
            return False, (
                "This appears to be a non-oral scene image. "
                "Please provide only oral images (mouth, teeth, gums, or lesions)."
            )

        # Reject flat-color graphics/logos
        if (
            features['flat_color_ratio'] > 0.75 and
            features['white_ratio'] > 0.35 and
            features['dark_ratio'] < 0.15
        ):
            return False, (
                "This appears to be a logo/graphic image. "
                "Please provide only oral images (mouth, teeth, gums, or lesions)."
            )

        if score < threshold:
            return False, (
                "This image does not appear to be an oral cavity photograph. "
                "Please upload a clear intraoral image showing the mouth, teeth, gums, or oral lesions. "
                f"(Oral score: {score:.2f})"
            )

        return True, "OK"

    except Exception as e:
        return True, "OK"  # Fail open — let main classifier handle it
