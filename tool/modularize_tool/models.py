import torch
import torch.nn as nn
from torchvision.models import efficientnet_b4, EfficientNet_B4_Weights
from config import CONFIG

class EnhancedEfficientNetB4(nn.Module):
    def __init__(self, num_classes=CONFIG['NUM_CLASSES'], dropout_rate=0.4):
        super(EnhancedEfficientNetB4, self).__init__()
        self.backbone = efficientnet_b4(weights=EfficientNet_B4_Weights.IMAGENET1K_V1)
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
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        logits = self.backbone(x)
        return logits