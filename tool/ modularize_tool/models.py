import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import efficientnet_b4, EfficientNet_B4_Weights
from config import CONFIG
import logging

class EnhancedEfficientNetB4(nn.Module):
    def __init__(self, num_classes=2, dropout_rate=0.4):
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

class GradCAM:
    def __init__(self, model, target_layer=None):
        self.model = model.eval()
        if target_layer is None:
            target_layer = self.model.backbone.features[-2] if hasattr(self.model.backbone, 'features') else list(self.model.modules())[-2]
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        try:
            self.target_layer.register_forward_hook(self.save_activations)
            self.target_layer.register_full_backward_hook(self.save_gradients)
        except AttributeError:
            logging.error("Invalid target layer for GradCAM")
            raise ValueError("Target layer is not valid for the provided model")

    def save_activations(self, module, input, output):
        self.activations = output.detach()

    def save_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_image, roi_mask, target_class=None):
        input_image = input_image.requires_grad_(True)
        output = self.model(input_image)
        confidence = torch.softmax(output, dim=1)[0, 1].item()
        if target_class is None:
            target_class = 1 if confidence < 0.9 else output.argmax(dim=1).item()
        final_confidence = torch.softmax(output, dim=1)[0, target_class].item()
        self.model.zero_grad()
        output[:, target_class].backward()
        weights = torch.mean(self.gradients, dim=[2, 3], keepdim=True)
        weights = F.relu(weights)
        raw_heatmap = torch.sum(weights * F.relu(self.activations), dim=1).squeeze()
        raw_heatmap = F.relu(raw_heatmap)
        if not CONFIG['BYPASS_ROI_MASK']:
            roi_mask_tensor = torch.tensor(roi_mask, dtype=torch.float32, device=CONFIG['DEVICE'])
            mask_tensor = F.interpolate(
                roi_mask_tensor.unsqueeze(0).unsqueeze(0),
                size=raw_heatmap.shape[-2:], mode='bilinear', align_corners=False
            ).squeeze()
            raw_heatmap = raw_heatmap * mask_tensor
        heatmap = raw_heatmap / (raw_heatmap.max() + 1e-8)
        heatmap_np = heatmap.cpu().numpy()
        logging.info(f"Grad-CAM heatmap max: {heatmap_np.max():.3f}, min: {heatmap_np.min():.3f}")
        return heatmap_np, target_class, output, final_confidence