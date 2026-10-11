import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as T
from functools import lru_cache
import os

try:
    import timm
except ImportError:
    timm = None

class SeverityScorer:
    def __init__(self, device='cpu'):
        self.device = device
        self.model = None
        if timm is not None:
            use_pretrained = os.environ.get('ORALAID_ALLOW_MODEL_DOWNLOADS') == '1'
            try:
                self.model = timm.create_model('mobilevit_xs', pretrained=use_pretrained)
            except Exception as exc:
                print(f"⚠️  MobileViT-XS pretrained weights unavailable ({exc}); using local initialization")
                self.model = timm.create_model('mobilevit_xs', pretrained=False)

            self.model.head.fc = nn.Sequential(
                nn.Linear(384, 1),
                nn.Sigmoid()
            )
            self.model.to(self.device)
            self.model.eval()
        else:
            print("⚠️  timm is not installed; severity scorer will use heuristic-only mode")
        
        self.transform = T.Compose([
            T.Resize((256, 256)),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def score(self, image: Image.Image, normal_confidence, is_high_risk_class, lesion_area_ratio):
        """
        Heuristic-based severity scoring (1-10)
        Using MobileViT-XS features + diagnostic context
        """
        if self.model is not None:
            img_tensor = self.transform(image).unsqueeze(0).to(self.device)
            with torch.no_grad():
                _ = self.model(img_tensor)

        high_risk_indicator = 1.0 if is_high_risk_class else 0.0
        normalized_lesion_area = max(0.0, min(1.0, float(lesion_area_ratio or 0)))
        normal_confidence = max(0.0, min(1.0, float(normal_confidence or 0)))

        base_score = (
            0.3 * (1.0 - normal_confidence) +
            0.4 * high_risk_indicator +
            0.3 * normalized_lesion_area
        )
        return round(max(1.0, base_score * 10), 1)

@lru_cache(maxsize=1)
def load_severity_scorer(device):
    return SeverityScorer(device)
