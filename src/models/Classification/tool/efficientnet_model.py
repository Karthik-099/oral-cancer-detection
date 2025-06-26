import torch
import torch.nn as nn
from torchvision.models import efficientnet_b4, EfficientNet_B4_Weights
import os
import logging

# Configuration
CONFIG = {
    'MODEL_PATH': '/home/user/Desktop/oral_cancer/gdc_oral_cancer_dataset/oral-cancer-detection/models/trained_efficientnet_b4_oral_cancer.pth',
    'DEVICE': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    'NUM_CLASSES': 2,
}

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

def load_model():
    try:
        if not os.path.exists(CONFIG['MODEL_PATH']):
            logging.error("Model file not found!")
            raise FileNotFoundError("Model file not found!")
        model = EnhancedEfficientNetB4(num_classes=CONFIG['NUM_CLASSES'])
        checkpoint = torch.load(CONFIG['MODEL_PATH'], map_location=CONFIG['DEVICE'])
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        model.to(CONFIG['DEVICE'])
        model.eval()
        logging.info("Model loaded successfully!")
        return model
    except Exception as e:
        logging.error(f"Error loading model: {str(e)}")
        raise Exception(f"Failed to load model: {str(e)}")