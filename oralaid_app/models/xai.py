"""
XAI Module — Real LIME + GradCAM
"""
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from PIL import Image
import base64
import io
import os


# ─── GradCAM ────────────────────────────────────────────────────────────────

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self._register_hooks()

    def _register_hooks(self):
        def fwd_hook(_, __, output):
            self.activations = output.detach()

        def bwd_hook(_, __, grad_output):
            self.gradients = grad_output[0].detach()

        self.target_layer.register_forward_hook(fwd_hook)
        self.target_layer.register_full_backward_hook(bwd_hook)

    def generate(self, input_tensor, class_idx=None):
        self.model.eval()
        output = self.model(input_tensor)
        if class_idx is None:
            class_idx = output.argmax(dim=1).item()

        self.model.zero_grad()
        output[0, class_idx].backward()

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=input_tensor.shape[2:], mode='bilinear', align_corners=False)
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam


def apply_gradcam(model, image_tensor, target_layer, original_image, save_path=None):
    gc = GradCAM(model, target_layer)
    cam = gc.generate(image_tensor)

    # Resize original image to match tensor for overlay
    img_np = np.array(original_image.resize((image_tensor.shape[3], image_tensor.shape[2])))
    
    # Generate heatmap
    heatmap = cv2.applyColorMap((cam * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    
    # Overlay heatmap on original image (alpha 0.4 for heatmap)
    overlay = cv2.addWeighted(img_np, 0.6, heatmap, 0.4, 0)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        Image.fromarray(overlay.astype(np.uint8)).save(save_path)
        return save_path, cam
        
    return _to_base64(overlay), cam


# ─── LIME ────────────────────────────────────────────────────────────────────

def apply_lime(model, image, device, save_path=None, num_samples=500, num_segments=50):
    """Simplified LIME using superpixel perturbation"""
    from skimage.segmentation import slic
    from skimage.color import label2rgb

    img_np = np.array(image.resize((224, 224))).astype(np.float32) / 255.0
    segments = slic(img_np, n_segments=num_segments, compactness=10, sigma=1, start_label=0)
    n_segments = segments.max() + 1

    import torchvision.transforms as T
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    # Get original prediction
    orig_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        orig_pred = torch.softmax(model(orig_tensor), dim=1).cpu().numpy()[0]
    target_class = orig_pred.argmax()

    # Perturb
    weights = np.zeros(n_segments)
    for _ in range(num_samples):
        mask = np.random.randint(0, 2, n_segments)
        perturbed = img_np.copy()
        for seg_id in range(n_segments):
            if mask[seg_id] == 0:
                perturbed[segments == seg_id] = 0.5

        pil_perturbed = Image.fromarray((perturbed * 255).astype(np.uint8))
        t = transform(pil_perturbed).unsqueeze(0).to(device)
        with torch.no_grad():
            pred = torch.softmax(model(t), dim=1).cpu().numpy()[0]

        for seg_id in range(n_segments):
            if mask[seg_id] == 1:
                weights[seg_id] += pred[target_class] - orig_pred[target_class]

    # Normalize weights for visualization
    if weights.max() != weights.min():
        weights = (weights - weights.min()) / (weights.max() - weights.min() + 1e-8)
    else:
        weights = np.zeros_like(weights)

    # Visualize top segments (contributing positively to target class)
    top_mask = np.zeros_like(segments, dtype=bool)
    # Highlight top 30% of superpixels
    num_top = max(1, int(n_segments * 0.3))
    top_segs = np.argsort(weights)[-num_top:]
    for s in top_segs:
        top_mask[segments == s] = True

    img_vis = (img_np * 255).astype(np.uint8)
    overlay = img_vis.copy()
    
    # Apply green tint to important regions
    overlay[top_mask] = (overlay[top_mask] * 0.5 + np.array([0, 200, 100]) * 0.5).astype(np.uint8)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        Image.fromarray(overlay.astype(np.uint8)).save(save_path)
        return save_path

    return _to_base64(overlay)


# ─── Segmentation Overlay ───────────────────────────────────────────────────

def apply_segmentation(original_image, mask, save_path=None):
    """Overlay binary mask as semi-transparent red on original image"""
    img_np = np.array(original_image.resize((256, 256)))
    
    # Ensure mask is 2D and matches image size
    if mask.ndim == 3:
        mask = mask[0]
    mask_resized = cv2.resize(mask, (256, 256), interpolation=cv2.INTER_NEAREST)
    
    # Create red mask overlay
    red_mask = np.zeros_like(img_np)
    red_mask[mask_resized > 0.5] = [255, 0, 0]  # Red
    
    # Blend images
    overlay = cv2.addWeighted(img_np, 0.7, red_mask, 0.3, 0)
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        Image.fromarray(overlay.astype(np.uint8)).save(save_path)
        return save_path
        
    return _to_base64(overlay)


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _to_base64(img_np):
    pil = Image.fromarray(img_np.astype(np.uint8))
    buf = io.BytesIO()
    pil.save(buf, format='PNG')
    return base64.b64encode(buf.getvalue()).decode()
