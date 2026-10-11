"""
EfficientNet-B4 Classification Model
8-class oral cancer classification with metadata fusion
"""
import torch
import torch.nn as nn
from torchvision.models import efficientnet_b4, EfficientNet_B4_Weights
from functools import lru_cache


CLASS_NAMES = [
    'Normal', 'Keratosis', 'Leukoplakia',
    'Speckled Leukoplakia', 'Erythroplakia',
    'Lichen Planus', 'Squamous Cell Carcinoma', 'Other Lesion'
]

RISK_MAPPING = {
    0: {'name': 'Normal',                  'risk': 'Low-risk',    'color': '#10b981', 'category': 'Non-suspicious'},
    1: {'name': 'Keratosis',               'risk': 'Low-risk',    'color': '#f59e0b', 'category': 'Suspicious'},
    2: {'name': 'Leukoplakia',             'risk': 'Low-risk',    'color': '#f59e0b', 'category': 'Suspicious'},
    3: {'name': 'Speckled Leukoplakia',    'risk': 'High-risk',   'color': '#ef4444', 'category': 'Suspicious'},
    4: {'name': 'Erythroplakia',           'risk': 'High-risk',   'color': '#ef4444', 'category': 'Suspicious'},
    5: {'name': 'Lichen Planus',           'risk': 'Low-risk',    'color': '#f59e0b', 'category': 'Suspicious'},
    6: {'name': 'Squamous Cell Carcinoma', 'risk': 'High-risk',   'color': '#ef4444', 'category': 'Suspicious'},
    7: {'name': 'Other Lesion',            'risk': 'Medium-risk', 'color': '#f97316', 'category': 'Suspicious'},
}

BINARY_RISK_MAPPING = {
    0: {'name': 'Normal',                 'risk': 'Low-risk',    'color': '#1D9E75', 'category': 'Non-suspicious'},
    1: {'name': 'Suspicious Oral Lesion', 'risk': 'High-risk',   'color': '#E24B4A', 'category': 'Suspicious'},
}


class OralCancerClassifier(nn.Module):
    """EfficientNet-B4 with metadata fusion head"""

    def __init__(self, num_classes=8, metadata_dim=5, use_pretrained=False):
        super().__init__()
        self.num_classes = num_classes
        self.risk_mapping = RISK_MAPPING
        weights = EfficientNet_B4_Weights.IMAGENET1K_V1 if use_pretrained else None
        self.backbone = efficientnet_b4(weights=weights)
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Identity()

        self.metadata_encoder = nn.Sequential(
            nn.Linear(metadata_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
        )

        self.classifier = nn.Sequential(
            nn.Linear(in_features + 32, 512),
            nn.SiLU(),
            nn.Dropout(0.4),
            nn.Linear(512, num_classes),
        )

    def forward(self, image, metadata=None):
        img_feat = self.backbone(image)
        if metadata is not None:
            meta_feat = self.metadata_encoder(metadata)
            feat = torch.cat([img_feat, meta_feat], dim=1)
        else:
            dummy = torch.zeros(image.size(0), 32, device=image.device)
            feat = torch.cat([img_feat, dummy], dim=1)
        return self.classifier(feat)


class CheckpointEfficientNetB4(nn.Module):
    """EfficientNet-B4 head shape used by final_efficientnet_b4_oral_cancer.pth."""

    def __init__(self, num_classes=2, dropout_rate=0.4, use_pretrained=False):
        super().__init__()
        self.num_classes = num_classes
        self.risk_mapping = BINARY_RISK_MAPPING if num_classes == 2 else RISK_MAPPING
        weights = EfficientNet_B4_Weights.IMAGENET1K_V1 if use_pretrained else None
        self.backbone = efficientnet_b4(weights=weights)
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

    def forward(self, image, metadata=None):
        return self.backbone(image)


def _format_accuracy(value):
    if value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    if value <= 1:
        value *= 100
    return f"{value:.2f}%"


def _checkpoint_accuracy(checkpoint):
    if not isinstance(checkpoint, dict):
        return None
    for key in ('best_acc', 'best_val_acc', 'val_accuracy', 'accuracy'):
        if key in checkpoint:
            return checkpoint[key]
    val_accs = checkpoint.get('val_accs')
    if val_accs:
        return max(val_accs)
    return None


@lru_cache(maxsize=1)
def load_classifier(model_path, device):
    import os

    checkpoint_exists = bool(model_path and os.path.exists(model_path))

    if checkpoint_exists:
        checkpoint = torch.load(model_path, map_location=device)
        state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
        if 'metadata_encoder.0.weight' in state_dict:
            model = OralCancerClassifier(use_pretrained=False)
            model.risk_mapping = RISK_MAPPING
            model.num_classes = 8
        elif 'backbone.classifier.9.weight' in state_dict:
            num_classes = int(state_dict['backbone.classifier.9.weight'].shape[0])
            model = CheckpointEfficientNetB4(num_classes=num_classes, use_pretrained=False)
        else:
            model = OralCancerClassifier(use_pretrained=False)
            model.risk_mapping = RISK_MAPPING
            model.num_classes = 8

        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)

        loaded_accuracy = _format_accuracy(_checkpoint_accuracy(checkpoint))
        model.loaded_accuracy = loaded_accuracy
        print(f"✅ Classifier loaded from {model_path}")
        if loaded_accuracy:
            print(f"✅ Loaded checkpoint validation accuracy: {loaded_accuracy}")
        else:
            print("ℹ️  Loaded checkpoint has no accuracy metadata")
    else:
        try:
            model = OralCancerClassifier(use_pretrained=True)
        except Exception as exc:
            print(f"⚠️  Could not initialize ImageNet pretrained classifier ({exc}); using uninitialized weights")
            model = OralCancerClassifier(use_pretrained=False)
        model.risk_mapping = RISK_MAPPING
        model.num_classes = 8
        print("⚠️  Classifier checkpoint not found; using ImageNet pretrained weights")
        model.loaded_accuracy = None
    return model.to(device).eval()


def get_model_info(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    size_mb = total * 4 / (1024 ** 2)
    return {
        'total_parameters': total,
        'trainable_parameters': trainable,
        'model_size_mb': round(size_mb, 2),
        'architecture': 'EfficientNet-B4 Checkpoint Head' if isinstance(model, CheckpointEfficientNetB4) else 'EfficientNet-B4 + Metadata Fusion',
        'num_classes': getattr(model, 'num_classes', 8),
        'input_size': '380x380',
        'loaded_accuracy': getattr(model, 'loaded_accuracy', None),
    }
