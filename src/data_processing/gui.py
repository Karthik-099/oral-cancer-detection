import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.models import efficientnet_b4, EfficientNet_B4_Weights
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
from lime import lime_image
from skimage.segmentation import slic
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import os
import json
from datetime import datetime
import cv2
import logging
import time
import pandas as pd
import warnings
import scipy.ndimage as ndimage

warnings.filterwarnings('ignore')

# Setup base directory
BASE_DIR = '/home/karthik/oral-cancer-detection'

# Create results directories before logging
RESULTS_DIR = os.path.join(BASE_DIR, 'results_fl')
os.makedirs(os.path.join(RESULTS_DIR, 'debug'), exist_ok=True)
os.makedirs(os.path.join(RESULTS_DIR, 'lime_gradcam_results'), exist_ok=True)
os.makedirs(os.path.join(RESULTS_DIR, 'combined_results'), exist_ok=True)

# Setup logging
logging.basicConfig(
    filename=os.path.join(RESULTS_DIR, 'debug.log'),
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

CONFIG = {
    'MODEL_PATH': os.path.join(BASE_DIR, 'models', 'trained_efficientnet_b4_oral_cancer.pth'),
    'DATA_DIR': os.path.join(BASE_DIR, 'data', 'processed'),
    'CSV_PATH': os.path.join(BASE_DIR, 'data', 'processed_labels.csv'),
    'RESULTS_DIR': RESULTS_DIR,
    'IMAGE_SIZE': 300,
    'DEVICE': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    'HEATMAP_THRESHOLD': 0.3,
    'MIN_MUCOSAL_AREA': 400,
    'BYPASS_TOOL_MASK': True,
    'BYPASS_ROI_MASK': False,
    'SHOW_ROI_BBOX': False,  # New config to control ROI bounding box display
    'NUM_CLASSES': 2,
}

CANCER_TYPES = {
    1: 'Homogenous Oral Leukoplakia',
    2: 'Smokeless Tobacco Keratosis',
    3: 'Squamous Cell Carcinoma',
    4: 'Proliferative Verrucous Oral Leukoplakia',
    5: 'Erythroplakia',
    6: 'Oral Lichen Planus',
    7: 'Speckled Oral Leukoplakia',
    8: 'Oral Cavity Normal',
    9: 'Benign',
    10: 'Other'
}

def normalize_path(image_path):
    norm_path = image_path.lower().replace('\\', '/').strip('/')
    data_dir_prefix = CONFIG['DATA_DIR'].lower().replace('\\', '/').strip('/')
    if norm_path.startswith(data_dir_prefix):
        norm_path = norm_path[len(data_dir_prefix):].lstrip('/')
    parts = norm_path.split('/')
    if len(parts) > 2:
        norm_path = '/'.join(parts[-2:])
    norm_path = os.path.splitext(norm_path)[0] + '.jpeg'
    logging.info(f"Normalized path: {image_path} -> {norm_path}")
    return norm_path

def detect_intraoral_roi(image):
    start = time.time()
    image_np = np.array(image)
    image_hsv = cv2.cvtColor(image_np, cv2.COLOR_RGB2HSV)
    timestamp = time.time()
    debug_dir = os.path.join(CONFIG['RESULTS_DIR'], 'debug')

    # Mucosa: Pink/red hues, exclude tongue
    lower_mucosa = np.array([0, 30, 20])
    upper_mucosa = np.array([25, 200, 255])
    lower_mucosa2 = np.array([155, 30, 20])
    upper_mucosa2 = np.array([180, 200, 255])
    mucosa_mask = cv2.bitwise_or(
        cv2.inRange(image_hsv, lower_mucosa, upper_mucosa),
        cv2.inRange(image_hsv, lower_mucosa2, upper_mucosa2)
    )
    cv2.imwrite(os.path.join(debug_dir, f'mucosa_mask_initial_{timestamp}.png'), mucosa_mask)

    # Exclude skin (fingers)
    lower_skin = np.array([0, 15, 80])
    upper_skin = np.array([25, 255, 255])
    skin_mask = cv2.inRange(image_hsv, lower_skin, upper_skin)
    skin_mask = cv2.dilate(skin_mask, np.ones((7, 7), np.uint8), iterations=2)
    mucosa_mask = cv2.bitwise_and(mucosa_mask, cv2.bitwise_not(skin_mask))
    cv2.imwrite(os.path.join(debug_dir, f'mucosa_mask_post_skin_{timestamp}.png'), mucosa_mask)

    # Exclude teeth
    _, _, v_channel = cv2.split(image_hsv)
    brightness_mask = v_channel < 180
    mucosa_mask = cv2.bitwise_and(mucosa_mask, mucosa_mask, mask=brightness_mask.astype(np.uint8))
    cv2.imwrite(os.path.join(debug_dir, f'mucosa_mask_post_teeth_{timestamp}.png'), mucosa_mask)

    # Texture filter
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    edge_magnitude = cv2.Laplacian(gray, cv2.CV_64F, ksize=3)
    edge_magnitude = cv2.normalize(np.abs(edge_magnitude), None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    texture_mask = edge_magnitude > 15
    mucosa_mask_pre_texture = mucosa_mask.copy()
    mucosa_mask = cv2.bitwise_and(mucosa_mask, mucosa_mask, mask=texture_mask.astype(np.uint8))
    cv2.imwrite(os.path.join(debug_dir, f'mucosa_mask_post_texture_{timestamp}.png'), mucosa_mask)
    if np.sum(mucosa_mask > 0) == 0:
        logging.warning(f"Texture filter zeroed out mucosa mask at {timestamp}, reverting")
        mucosa_mask = mucosa_mask_pre_texture

    # Tool mask
    tool_mask = np.zeros_like(mucosa_mask)
    if not CONFIG['BYPASS_TOOL_MASK']:
        tool_mask = cv2.bitwise_or(
            cv2.inRange(image_hsv, np.array([0, 70, 0]), np.array([180, 255, 130])),
            cv2.inRange(image_hsv, np.array([0, 0, 140]), np.array([180, 60, 255]))
        )
        tool_mask = cv2.dilate(tool_mask, np.ones((5, 5), np.uint8), iterations=2)
        mucosa_mask = cv2.bitwise_and(mucosa_mask, cv2.bitwise_not(tool_mask))
    cv2.imwrite(os.path.join(debug_dir, f'tool_mask_{timestamp}.png'), tool_mask)

    # Fallback if mucosa mask is empty
    if np.sum(mucosa_mask > 0) == 0:
        logging.warning(f"Empty mucosa mask at {timestamp}, using fallback mask")
        mucosa_mask = np.ones_like(mucosa_mask, dtype=np.uint8) * 255

    mucosa_coverage = np.sum(mucosa_mask > 0) / mucosa_mask.size
    logging.info(f"Mucosa mask coverage: {mucosa_coverage:.3f}, saved as mucosa_mask_final_{timestamp}.png")

    kernel = np.ones((5, 5), np.uint8)
    mucosa_mask = cv2.dilate(mucosa_mask, kernel, iterations=2)
    mucosa_mask = cv2.erode(mucosa_mask, kernel, iterations=1)
    cv2.imwrite(os.path.join(debug_dir, f'mucosa_mask_final_{timestamp}.png'), mucosa_mask)

    contours, _ = cv2.findContours(mucosa_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    is_mouth_open = False
    roi_mask = mucosa_mask
    bbox = None
    img_center = (image_np.shape[1] / 2, image_np.shape[0] / 2)
    if contours:
        valid_contours = []
        for c in contours:
            area = cv2.contourArea(c)
            if area > CONFIG['MIN_MUCOSAL_AREA']:
                x, y, w, h = cv2.boundingRect(c)
                contour_center = (x + w / 2, y + h / 2)
                distance_to_center = np.sqrt((contour_center[0] - img_center[0])**2 + (contour_center[1] - img_center[1])**2)
                perimeter = cv2.arcLength(c, True)
                circularity = (4 * np.pi * area) / (perimeter**2) if perimeter > 0 else 0
                if distance_to_center < min(image_np.shape[:2]) / 2.5 and 0.3 < circularity < 1.0:
                    valid_contours.append((c, area, x, y, w, h))
        if valid_contours:
            _, contour_area, x, y, w, h = max(valid_contours, key=lambda x: x[1])
            is_mouth_open = True
            x, y, w, h = max(0, x), max(0, y), min(w, image_np.shape[1] - x), min(h, image_np.shape[0] - y)
            bbox = (x, y, w, h)
            logging.info(f"Detected ROI bbox: {bbox}, area: {contour_area}")
        else:
            logging.info("No valid contours found")

    if not is_mouth_open:
        crop_size = min(image_np.shape[:2]) // 3
        cy, cx = image_np.shape[0] // 2, image_np.shape[1] // 2
        x = max(0, cx - crop_size)
        y = max(0, cy - crop_size)
        w = min(crop_size * 2, image_np.shape[1] - x)
        h = min(crop_size * 2, image_np.shape[0] - y)
        bbox = (x, y, w, h)
        roi_mask = np.zeros_like(mucosa_mask)
        roi_mask[y:y+h, x:x+w] = 255
        logging.info(f"Fallback central crop bbox: {bbox}")

    roi_image = Image.fromarray(image_np).resize((CONFIG['IMAGE_SIZE'], CONFIG['IMAGE_SIZE']))
    roi_mask = cv2.resize(roi_mask, (CONFIG['IMAGE_SIZE'], CONFIG['IMAGE_SIZE']), interpolation=cv2.INTER_NEAREST)
    logging.info(f"ROI detection time: {time.time() - start:.2f}s")
    return roi_image, roi_mask, bbox, is_mouth_open, tool_mask

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
    def __init__(self, model, target_layer):
        self.model = model.eval()
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.target_layer.register_forward_hook(self.save_activations)
        self.target_layer.register_full_backward_hook(self.save_gradients)

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

class LIMEExplainer:
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.mean = np.array([0.5, 0.5, 0.5])
        self.std = np.array([0.2, 0.2, 0.2])
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.2, 0.2, 0.2])
        ])

    def detect_reflections(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(blurred, 200, 255, cv2.THRESH_BINARY)
        kernel = np.ones((3, 3), np.uint8)
        thresh = cv2.dilate(thresh, kernel, iterations=1)
        reflection_mask = thresh == 0
        debug_dir = os.path.join(CONFIG['RESULTS_DIR'], 'debug')
        timestamp = time.time()
        cv2.imwrite(os.path.join(debug_dir, f'reflection_mask_{timestamp}.png'), reflection_mask.astype(np.uint8) * 255)
        reflection_coverage = np.sum(reflection_mask) / reflection_mask.size
        logging.info(f"Reflection mask coverage: {reflection_coverage:.3f}")
        if np.sum(reflection_mask) == 0:
            logging.warning(f"Empty reflection mask at {timestamp}, using fallback")
            reflection_mask = np.ones_like(reflection_mask, dtype=bool)
        return reflection_mask

    def enhanced_segmentation(self, image, n_segments=200):
        roi_image, roi_mask, _, _, _ = detect_intraoral_roi(Image.fromarray(image.astype(np.uint8)))
        reflection_mask = self.detect_reflections(image)
        combined_mask = np.logical_and(roi_mask > 0, reflection_mask)
        debug_dir = os.path.join(CONFIG['RESULTS_DIR'], 'debug')
        timestamp = time.time()
        cv2.imwrite(os.path.join(debug_dir, f'combined_mask_{timestamp}.png'), combined_mask.astype(np.uint8) * 255)
        combined_coverage = np.sum(combined_mask) / combined_mask.size
        logging.info(f"Combined mask coverage: {combined_coverage:.3f}")
        if np.sum(combined_mask) == 0:
            logging.warning(f"Empty combined mask at {timestamp}, using fallback")
            combined_mask = np.ones_like(combined_mask, dtype=bool)
        try:
            segments = slic(
                image,
                n_segments=n_segments,
                compactness=10,
                start_label=1,
                mask=combined_mask,
                enforce_connectivity=True,
                sigma=0.5
            )
        except Exception as e:
            logging.error(f"SLIC failed: {str(e)}, using default segmentation")
            segments = slic(
                image,
                n_segments=n_segments,
                compactness=10,
                start_label=1,
                enforce_connectivity=True,
                sigma=0.5
            )
        return segments

    def predict_fn(self, images):
        self.model.eval()
        predictions = []
        with torch.no_grad():
            for img in images:
                if img.max() > 1.0:
                    img = img / 255.0
                normalized = (img - self.mean) / self.std
                tensor = torch.FloatTensor(normalized).permute(2, 0, 1).unsqueeze(0).to(self.device)
                output = self.model(tensor)
                prob = torch.nn.functional.softmax(output, dim=1)
                predictions.append(prob.cpu().numpy()[0])
        return np.array(predictions)

    def explain_image(self, image_path, n_segments=200, brightness_adjust=1.0, progress_callback=None):
        original_image = Image.open(image_path).convert('RGB')
        img_array = np.array(original_image)
        img_array = np.clip(img_array * brightness_adjust, 0, 255).astype(np.uint8)
        adjusted_image = Image.fromarray(img_array)
        resized_image = adjusted_image.resize((CONFIG['IMAGE_SIZE'], CONFIG['IMAGE_SIZE']))
        img_array = np.array(resized_image)
        roi_image, roi_mask, bbox, is_mouth_open, tool_mask = detect_intraoral_roi(original_image)
        if progress_callback:
            progress_callback(20)  # ROI detection complete

        img_tensor = self.transform(resized_image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            output = self.model(img_tensor)
            probs = torch.nn.functional.softmax(output, dim=1)
            predicted_class = torch.argmax(probs, dim=1).item()
            confidence = probs.max().item()
            non_suspicious_prob = probs[0][0].item()
            suspicious_prob = probs[0][1].item()
        if progress_callback:
            progress_callback(40)  # Model prediction complete

        try:
            explainer = lime_image.LimeImageExplainer()
            explanation = explainer.explain_instance(
                img_array,
                self.predict_fn,
                top_labels=2,
                hide_color=0,
                num_samples=1500,
                segmentation_fn=lambda x: self.enhanced_segmentation(x, n_segments)
            )
            segments = self.enhanced_segmentation(img_array, n_segments)
            if progress_callback:
                progress_callback(80)  # LIME explanation complete
            feature_importance = explanation.local_exp[predicted_class]
        except Exception as e:
            logging.error(f"LIME failed: {str(e)}, using fallback")
            segments = np.zeros((img_array.shape[0], img_array.shape[1]), dtype=np.int32)
            feature_importance = []
            explanation = type('Explanation', (), {'local_exp': {predicted_class: []}})
            if progress_callback:
                progress_callback(80)

        positive_segments = [(seg_id, imp) for seg_id, imp in feature_importance if imp > 0]
        positive_segments.sort(key=lambda x: x[1], reverse=True)
        negative_segments = [(seg_id, imp) for seg_id, imp in feature_importance if imp < 0]
        negative_segments.sort(key=lambda x: x[1])

        cancer_type = self.determine_cancer_type(suspicious_prob, predicted_class)
        risk_level = self.determine_risk_level(suspicious_prob, predicted_class)
        if progress_callback:
            progress_callback(100)  # Analysis complete

        return {
            'image': resized_image,
            'img_array': img_array,
            'roi_image': roi_image,
            'roi_mask': roi_mask,
            'bbox': bbox,
            'is_mouth_open': is_mouth_open,
            'tool_mask': tool_mask,
            'predicted_class': predicted_class,
            'confidence': confidence,
            'non_suspicious_prob': non_suspicious_prob,
            'suspicious_prob': suspicious_prob,
            'segments': segments,
            'positive_segments': positive_segments,
            'negative_segments': negative_segments,
            'all_segments': feature_importance,
            'explanation': explanation,
            'cancer_type': cancer_type,
            'risk_level': risk_level,
            'brightness_adjust': brightness_adjust
        }

    def determine_cancer_type(self, suspicious_prob, predicted_class):
        if predicted_class == 0:
            return CANCER_TYPES[8]
        else:
            if suspicious_prob > 0.9:
                return CANCER_TYPES[3]
            elif suspicious_prob > 0.8:
                return CANCER_TYPES[5]
            elif suspicious_prob > 0.7:
                return CANCER_TYPES[1]
            elif suspicious_prob > 0.6:
                return CANCER_TYPES[7]
            else:
                return CANCER_TYPES[9]

    def determine_risk_level(self, suspicious_prob, predicted_class):
        if predicted_class == 0:
            return "Low Risk"
        else:
            if suspicious_prob > 0.9:
                return "Very High Risk"
            elif suspicious_prob > 0.8:
                return "High Risk"
            elif suspicious_prob > 0.7:
                return "Moderate Risk"
            elif suspicious_prob > 0.6:
                return "Moderate-Low Risk"
            else:
                return "Low Risk"

class EnhancedLIMEGradCAMVisualization:
    def __init__(self, root):
        self.root = root
        self.root.title("LIME & Grad-CAM Visualization - Oral Cancer Detection")
        self.root.geometry("1200x800")
        self.root.configure(bg='#f0f0f0')
        self.model = None
        self.explainer = None
        self.current_data = None
        self.threshold_var = tk.IntVar(value=100)
        self.segments_var = tk.IntVar(value=200)
        self.show_numbers_var = tk.BooleanVar(value=True)
        self.show_boundaries_var = tk.BooleanVar(value=True)
        self.brightness_var = tk.DoubleVar(value=1.0)
        self.segment_status = {}
        self.doctor_comments = ""
        self.selection_rect = None
        self.start_x = None
        self.start_y = None
        self.is_analyzing = False  # Flag to prevent recursive analysis
        self.setup_ui()
        self.load_model()

    def setup_ui(self):
        style = ttk.Style()
        style.theme_use('clam')
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        self.setup_control_panel(main_frame)
        self.setup_tabbed_panels(main_frame)
        self.setup_info_panel(main_frame)
        self.setup_remarks_panel(main_frame)

    def setup_control_panel(self, parent):
        control_frame = ttk.LabelFrame(parent, text="Controls", padding=10)
        control_frame.pack(fill=tk.X, pady=(0, 10))
        file_frame = ttk.Frame(control_frame)
        file_frame.pack(fill=tk.X, pady=(0, 10))
        ttk.Button(file_frame, text="Load Image", command=self.load_image).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(file_frame, text="Analyze", command=self.analyze_image).pack(side=tk.LEFT, padx=5)
        ttk.Button(file_frame, text="Save Results", command=self.save_results).pack(side=tk.LEFT, padx=5)
        self.status_var = tk.StringVar(value="Ready - Load an image to begin")
        status_label = ttk.Label(file_frame, textvariable=self.status_var, foreground='blue')
        status_label.pack(side=tk.RIGHT)
        # Progress Bar
        progress_frame = ttk.Frame(control_frame)
        progress_frame.pack(fill=tk.X, pady=(5, 0))
        self.progress_var = tk.DoubleVar(value=0)
        self.progress_bar = ttk.Progressbar(progress_frame, variable=self.progress_var, maximum=100, length=200)
        self.progress_bar.pack(side=tk.LEFT, padx=(0, 10))
        self.progress_label = ttk.Label(progress_frame, text="0%")
        self.progress_label.pack(side=tk.LEFT)
        param_frame = ttk.Frame(control_frame)
        param_frame.pack(fill=tk.X)
        ttk.Label(param_frame, text="Segments:").grid(row=0, column=0, sticky=tk.W, padx=(0, 5))
        segments_scale = ttk.Scale(param_frame, from_=100, to=300, variable=self.segments_var,
                                  orient=tk.HORIZONTAL, length=150, command=self.debounce(self.update_segments))
        segments_scale.grid(row=0, column=1, padx=5)
        self.segments_label = ttk.Label(param_frame, text="200")
        self.segments_label.grid(row=0, column=2, padx=(5, 20))
        ttk.Label(param_frame, text="Top Contributors (%):").grid(row=0, column=3, sticky=tk.W, padx=(0, 5))
        threshold_scale = ttk.Scale(param_frame, from_=10, to=100, variable=self.threshold_var,
                                   orient=tk.HORIZONTAL, length=150, command=self.debounce(self.update_threshold))
        threshold_scale.grid(row=0, column=4, padx=5)
        self.threshold_label = ttk.Label(param_frame, text="100%")
        self.threshold_label.grid(row=0, column=5, padx=(5, 20))
        ttk.Label(param_frame, text="Brightness:").grid(row=0, column=6, sticky=tk.W, padx=(0, 5))
        brightness_scale = ttk.Scale(param_frame, from_=0.5, to=2.0, variable=self.brightness_var,
                                    orient=tk.HORIZONTAL, length=150, command=self.debounce(self.update_brightness))
        brightness_scale.grid(row=0, column=7, padx=5)
        self.brightness_label = ttk.Label(param_frame, text="1.0")
        self.brightness_label.grid(row=0, column=8, padx=(5, 20))
        ttk.Checkbutton(param_frame, text="Show Numbers", variable=self.show_numbers_var,
                       command=self.update_display_options).grid(row=0, column=9, padx=5)
        ttk.Checkbutton(param_frame, text="Show Boundaries", variable=self.show_boundaries_var,
                       command=self.update_display_options).grid(row=0, column=10, padx=5)

    def debounce(self, func):
        def wrapper(*args, **kwargs):
            if not hasattr(self, '_debounce_timer'):
                self._debounce_timer = None
            if self._debounce_timer:
                self.root.after_cancel(self._debounce_timer)
            self._debounce_timer = self.root.after(300, lambda: func(*args, **kwargs))
        return wrapper

    def setup_tabbed_panels(self, parent):
        self.notebook = ttk.Notebook(parent)
        self.notebook.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        # Original Image Tab
        panel1 = ttk.Frame(self.notebook)
        self.notebook.add(panel1, text="Original Image")
        self.canvas1 = tk.Canvas(panel1, bg='white', relief=tk.SUNKEN, bd=1)
        self.canvas1.pack(fill=tk.BOTH, expand=True)
        self.canvas1.bind('<Button-1>', self.on_canvas_press)
        self.canvas1.bind('<B1-Motion>', self.on_canvas_drag)
        self.canvas1.bind('<ButtonRelease-1>', self.on_canvas_release)
        # LIME Segmentation Tab
        panel2 = ttk.Frame(self.notebook)
        self.notebook.add(panel2, text="LIME Segmentation")
        self.fig_lime = plt.figure(figsize=(5, 5), facecolor='white')
        self.ax_lime = self.fig_lime.add_subplot(111)
        self.canvas_lime = FigureCanvasTkAgg(self.fig_lime, panel2)
        self.canvas_lime.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        # Grad-CAM Heatmap Tab
        panel3 = ttk.Frame(self.notebook)
        self.notebook.add(panel3, text="Grad-CAM Heatmap")
        self.fig_gradcam = plt.figure(figsize=(5, 5), facecolor='white')
        self.ax_gradcam = self.fig_gradcam.add_subplot(111)
        self.canvas_gradcam = FigureCanvasTkAgg(self.fig_gradcam, panel3)
        self.canvas_gradcam.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        # LIME on Grad-CAM Overlay Tab
        panel4 = ttk.Frame(self.notebook)
        self.notebook.add(panel4, text="LIME on Grad-CAM Overlay")
        self.fig_lime_gradcam = plt.figure(figsize=(5, 5), facecolor='white')
        self.ax_lime_gradcam = self.fig_lime_gradcam.add_subplot(111)
        self.canvas_lime_gradcam = FigureCanvasTkAgg(self.fig_lime_gradcam, panel4)
        self.canvas_lime_gradcam.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        # Contribution Analysis Tab
        panel5 = ttk.Frame(self.notebook)
        self.notebook.add(panel5, text="Contribution Analysis")
        self.fig_contrib = plt.figure(figsize=(5, 5), facecolor='white')
        self.ax_contrib = self.fig_contrib.add_subplot(111)
        self.canvas_contrib = FigureCanvasTkAgg(self.fig_contrib, panel5)
        self.canvas_contrib.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def setup_info_panel(self, parent):
        info_frame = ttk.LabelFrame(parent, text="Analysis Results", padding=10)
        info_frame.pack(fill=tk.X)
        info_container = ttk.Frame(info_frame)
        info_container.pack(fill=tk.X)
        left_col = ttk.Frame(info_container)
        left_col.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.prediction_var = tk.StringVar(value="Class: Not analyzed")
        self.confidence_var = tk.StringVar(value="Confidence: N/A")
        self.probabilities_var = tk.StringVar(value="Probabilities: N/A")
        ttk.Label(left_col, textvariable=self.prediction_var, font=('Arial', 10, 'bold')).pack(anchor=tk.W)
        ttk.Label(left_col, textvariable=self.confidence_var).pack(anchor=tk.W)
        ttk.Label(left_col, textvariable=self.probabilities_var).pack(anchor=tk.W)
        right_col = ttk.Frame(info_container)
        right_col.pack(side=tk.RIGHT, fill=tk.X, expand=True)
        self.cancer_type_var = tk.StringVar(value="Cancer Type: N/A")
        self.risk_level_var = tk.StringVar(value="Risk Level: N/A")
        self.segments_info_var = tk.StringVar(value="Segments: N/A")
        ttk.Label(right_col, textvariable=self.cancer_type_var, font=('Arial', 10, 'bold')).pack(anchor=tk.W)
        ttk.Label(right_col, textvariable=self.risk_level_var).pack(anchor=tk.W)
        ttk.Label(right_col, textvariable=self.segments_info_var).pack(anchor=tk.W)

    def setup_remarks_panel(self, parent):
        remarks_frame = ttk.LabelFrame(parent, text="Doctor's Remarks", padding=10)
        remarks_frame.pack(fill=tk.X, pady=(10, 0))
        segments_frame = ttk.LabelFrame(remarks_frame, text="Segment Status (Check to Mark Non-Suspicious)", padding=5)
        segments_frame.pack(fill=tk.BOTH, expand=True)
        self.segments_canvas = tk.Canvas(segments_frame, height=150, bg='#f0f0f0')
        self.segments_scrollbar = ttk.Scrollbar(segments_frame, orient=tk.VERTICAL, command=self.segments_canvas.yview)
        self.segments_frame_inner = ttk.Frame(self.segments_canvas)
        self.segments_frame_inner.bind(
            "<Configure>",
            lambda e: self.segments_canvas.configure(scrollregion=self.segments_canvas.bbox("all"))
        )
        self.segments_canvas.create_window((0, 0), window=self.segments_frame_inner, anchor="nw", width=1100)
        self.segments_canvas.configure(yscrollcommand=self.segments_scrollbar.set)
        self.segments_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        self.segments_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        def on_mouse_wheel(event):
            if self.segments_canvas.winfo_exists():
                self.segments_canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        self.segments_canvas.bind_all("<MouseWheel>", on_mouse_wheel)
        comments_frame = ttk.Frame(remarks_frame)
        comments_frame.pack(fill=tk.X, pady=(10, 0))
        ttk.Label(comments_frame, text="Doctor's Comments:").pack(anchor=tk.W)
        self.comments_text = tk.Text(comments_frame, height=4, width=80)
        self.comments_text.pack(fill=tk.X, pady=(5, 0))
        ttk.Button(comments_frame, text="Save Comments", command=self.save_comments).pack(anchor=tk.W, pady=(5, 0))

    def load_model(self):
        try:
            if not os.path.exists(CONFIG['MODEL_PATH']):
                self.status_var.set("Model file not found!")
                return
            self.model = EnhancedEfficientNetB4(num_classes=CONFIG['NUM_CLASSES'])
            checkpoint = torch.load(CONFIG['MODEL_PATH'], map_location=CONFIG['DEVICE'])
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.model.load_state_dict(checkpoint)
            self.model.to(CONFIG['DEVICE'])
            self.model.eval()
            self.explainer = LIMEExplainer(self.model, CONFIG['DEVICE'])
            self.status_var.set("Model loaded successfully!")
        except Exception as e:
            self.status_var.set(f"Error loading model: {str(e)}")
            messagebox.showerror("Error", f"Failed to load model: {str(e)}")

    def load_image(self):
        if not self.model:
            messagebox.showerror("Error", "Model not loaded!")
            return
        file_path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")]
        )
        if file_path:
            self.current_image_path = file_path
            self.image_name = os.path.splitext(os.path.basename(file_path))[0]
            self.display_original_image()
            self.status_var.set(f"Image loaded: {os.path.basename(file_path)}")

    def display_original_image(self):
        if hasattr(self, 'current_image_path'):
            image = Image.open(self.current_image_path)
            canvas_width = 400
            canvas_height = 400
            image.thumbnail((canvas_width, canvas_height), Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(image)
            self.canvas1.delete("all")
            x = canvas_width // 2
            y = canvas_height // 2
            self.canvas1.create_image(x, y, image=photo)
            self.canvas1.photo = photo

    def on_canvas_press(self, event):
        self.start_x = self.canvas1.canvasx(event.x)
        self.start_y = self.canvas1.canvasy(event.y)
        if self.selection_rect:
            self.canvas1.delete(self.selection_rect)
        self.selection_rect = self.canvas1.create_rectangle(
            self.start_x, self.start_y, self.start_x, self.start_y, outline='red', width=2
        )

    def on_canvas_drag(self, event):
        current_x = self.canvas1.canvasx(event.x)
        current_y = self.canvas1.canvasy(event.y)
        self.canvas1.coords(self.selection_rect, self.start_x, self.start_y, current_x, current_y)

    def on_canvas_release(self, event):
        if self.current_data and self.start_x and self.start_y:
            end_x = self.canvas1.canvasx(event.x)
            end_y = self.canvas1.canvasy(event.y)
            self.apply_brightness_to_region(self.start_x, self.start_y, end_x, end_y)

    def apply_brightness_to_region(self, x1, y1, x2, y2):
        if not hasattr(self, 'current_image_path'):
            return
        image = Image.open(self.current_image_path)
        img_array = np.array(image)
        canvas_width = 400
        canvas_height = 400
        img_width, img_height = image.size
        scale_x = img_width / canvas_width
        scale_y = img_height / canvas_height
        x1 = int(x1 * scale_x)
        y1 = int(y1 * scale_y)
        x2 = int(x2 * scale_x)
        y2 = int(y2 * scale_y)
        x_min, x_max = min(x1, x2), max(x1, x2)
        y_min, y_max = min(y1, y2), max(y1, y2)
        region = img_array[y_min:y_max, x_min:x_max]
        region = np.clip(region * self.brightness_var.get(), 0, 255).astype(np.uint8)
        img_array[y_min:y_max, x_min:x_max] = region
        adjusted_image = Image.fromarray(img_array)
        adjusted_image.save(self.current_image_path)
        self.analyze_image()

    def update_progress(self, value):
        self.progress_var.set(value)
        self.progress_label.config(text=f"{int(value)}%")
        self.root.update()

    def analyze_image(self):
        if self.is_analyzing:
            return
        if not hasattr(self, 'current_image_path'):
            messagebox.showerror("Error", "Please load an image first!")
            return
        self.is_analyzing = True
        self.status_var.set("Analyzing image...")
        self.progress_var.set(0)
        self.progress_label.config(text="0%")
        self.root.update()
        try:
            n_segments = self.segments_var.get()
            brightness_adjust = self.brightness_var.get()
            # LIME Analysis
            self.current_data = self.explainer.explain_image(
                self.current_image_path, n_segments, brightness_adjust, self.update_progress
            )
            # Grad-CAM Analysis
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.2, 0.2, 0.2])
            ])
            img = Image.open(self.current_image_path).convert('RGB')
            roi_image, roi_mask, _, is_mouth_open, _ = detect_intraoral_roi(img)
            input_tensor = transform(roi_image).unsqueeze(0).to(CONFIG['DEVICE'])
            grad_cam = GradCAM(self.model, self.model.backbone.features[-2])
            heatmap, gradcam_class, _, gradcam_confidence = grad_cam.generate(input_tensor, roi_mask)
            self.current_data['gradcam_heatmap'] = heatmap
            self.current_data['gradcam_class'] = gradcam_class
            self.current_data['gradcam_confidence'] = gradcam_confidence
            self.update_progress(90)
            self.update_visualizations()
            self.update_info_panel()
            self.update_remarks_panel()
            self.generate_combined_visualization()
            pred_class = self.current_data['predicted_class']
            confidence = self.current_data['confidence']
            class_name = ['Non-Suspicious', 'Suspicious'][pred_class]
            self.status_var.set(f"Analysis complete: {class_name} ({confidence:.3f})")
            self.update_progress(100)
        except Exception as e:
            logging.error(f"Analysis failed: {str(e)}")
            messagebox.showerror("Error", f"Analysis failed: {str(e)}")
            self.status_var.set("Analysis failed!")
            self.update_progress(0)
        finally:
            self.is_analyzing = False

    def generate_combined_visualization(self):
        plt.ioff()
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 12))
        # Original Image
        original_image = Image.open(self.current_image_path).convert('RGB')
        original_array = np.array(original_image)
        if CONFIG['SHOW_ROI_BBOX'] and self.current_data['bbox']:
            x, y, w, h = self.current_data['bbox']
            cv2.rectangle(original_array, (x, y), (x+w, y+h), (0, 255, 0), 2)
        ax1.imshow(original_array)
        ax1.set_title('Original Image', fontsize=10)
        ax1.axis('off')
        # LIME Heatmap
        img_array = self.current_data['img_array']
        segments = self.current_data['segments']
        positive_segments = self.current_data['positive_segments']
        num_to_show = max(1, int(len(positive_segments) * self.threshold_var.get() / 100))
        segments_to_show = positive_segments[:num_to_show]
        display_image = img_array.copy().astype(np.float32) / 255.0
        heatmap = np.zeros_like(display_image[:,:,0])
        if segments_to_show:
            importances = [imp for _, imp in segments_to_show]
            if importances:
                min_imp, max_imp = min(importances), max(importances)
                imp_range = max_imp - min_imp if max_imp != min_imp else 1
                for i, (seg_id, importance) in enumerate(segments_to_show):
                    mask = segments == seg_id
                    norm_imp = (importance - min_imp) / imp_range if imp_range > 0 else 1
                    heatmap[mask] = norm_imp
                    if self.show_numbers_var.get():
                        y_coords, x_coords = np.where(mask)
                        if len(y_coords) > 0:
                            centroid_y, centroid_x = int(np.mean(y_coords)), int(np.mean(x_coords))
                            ax2.text(centroid_x, centroid_y, f'S{i}', fontsize=8, fontweight='bold', color='white',
                                     ha='center', va='center', bbox=dict(facecolor='black', alpha=0.9))
                heatmap_colored = plt.cm.hot(heatmap)[:,:,:3]
                blended = 0.7 * display_image + 0.3 * heatmap_colored
                if self.show_boundaries_var.get():
                    boundary_image = np.zeros_like(blended)
                    for seg_id, _ in segments_to_show:
                        mask = segments == seg_id
                        edges = ndimage.binary_erosion(mask) ^ mask
                        boundary_image[edges] = [1, 1, 1]
                    boundary_mask = np.any(boundary_image > 0, axis=2)
                    blended[boundary_mask] = boundary_image[boundary_mask]
                display_image = blended
        ax2.imshow(display_image)
        lime_title = f"LIME Heatmap\nSegments Shown: {num_to_show}/{len(positive_segments)}"
        ax2.set_title(lime_title, fontsize=10)
        ax2.axis('off')
        # Grad-CAM Heatmap
        gradcam_heatmap = self.current_data['gradcam_heatmap']
        gradcam_heatmap = cv2.resize(gradcam_heatmap, (CONFIG['IMAGE_SIZE'], CONFIG['IMAGE_SIZE']))
        gradcam_heatmap = np.uint8(255 * gradcam_heatmap)
        gradcam_heatmap = cv2.applyColorMap(gradcam_heatmap, cv2.COLORMAP_JET)
        gradcam_heatmap = cv2.cvtColor(gradcam_heatmap, cv2.COLOR_BGR2RGB)
        roi_array = np.array(self.current_data['roi_image'])
        gradcam_overlay = roi_array * 0.6 + gradcam_heatmap * 0.4
        ax3.imshow(gradcam_overlay.astype(np.uint8))
        gradcam_class_name = 'Suspicious' if self.current_data['gradcam_class'] == 1 else 'Non-Suspicious'
        gradcam_title = (
            f"Grad-CAM Heatmap\nPrediction: {gradcam_class_name} "
            f"(Confidence: {self.current_data['gradcam_confidence']:.3f})"
        )
        ax3.set_title(gradcam_title, fontsize=10)
        ax3.axis('off')
        # LIME on Grad-CAM Overlay
        display_image = gradcam_overlay.astype(np.float32) / 255.0
        if segments_to_show:
            if self.show_boundaries_var.get():
                boundary_image = np.zeros_like(display_image)
                for seg_id, _ in segments_to_show:
                    mask = segments == seg_id
                    edges = ndimage.binary_erosion(mask) ^ mask
                    boundary_image[edges] = [1, 1, 1]
                boundary_mask = np.any(boundary_image > 0, axis=2)
                display_image[boundary_mask] = boundary_image[boundary_mask]
            if self.show_numbers_var.get():
                for i, (seg_id, _) in enumerate(segments_to_show):
                    mask = segments == seg_id
                    y_coords, x_coords = np.where(mask)
                    if len(y_coords) > 0:
                        centroid_y, centroid_x = int(np.mean(y_coords)), int(np.mean(x_coords))
                        ax4.text(centroid_x, centroid_y, f'S{i}', fontsize=8, fontweight='bold', color='white',
                                 ha='center', va='center', bbox=dict(facecolor='black', alpha=0.9))
        ax4.imshow(display_image)
        ax4.set_title(f"LIME Segments on Grad-CAM\nSegments Shown: {num_to_show}/{len(positive_segments)}", fontsize=10)
        ax4.axis('off')
        # Save
        results_dir = os.path.join(CONFIG['RESULTS_DIR'], 'combined_results')
        os.makedirs(results_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        combined_path = os.path.join(results_dir, f'{self.image_name}_{timestamp}_combined.png')
        plt.tight_layout()
        plt.savefig(combined_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        logging.info(f"Combined visualization saved: {combined_path}")

    def update_segments(self, value):
        segments = int(float(value))
        self.segments_label.config(text=str(segments))
        if hasattr(self, 'current_image_path') and not self.is_analyzing:
            self.analyze_image()

    def update_threshold(self, value):
        percent = int(float(value))
        self.threshold_label.config(text=f"{percent}%")
        if self.current_data:
            self.update_visualizations()

    def update_brightness(self, value):
        brightness = float(value)
        self.brightness_label.config(text=f"{brightness:.2f}")
        if hasattr(self, 'current_image_path') and not self.is_analyzing:
            self.analyze_image()

    def update_display_options(self):
        if self.current_data:
            self.update_visualizations()

    def update_info_panel(self):
        if not self.current_data:
            return
        pred_class = self.current_data['predicted_class']
        confidence = self.current_data['confidence']
        non_suspicious_prob = self.current_data['non_suspicious_prob']
        suspicious_prob = self.current_data['suspicious_prob']
        cancer_type = self.current_data['cancer_type']
        risk_level = self.current_data['risk_level']
        class_name = ['Non-Suspicious', 'Suspicious'][pred_class]
        self.prediction_var.set(f"Prediction: {class_name}")
        self.confidence_var.set(f"Confidence: {confidence:.3f}")
        self.probabilities_var.set(f"Probabilities: Non-Suspicious: {non_suspicious_prob:.3f}, Suspicious: {suspicious_prob:.3f}")
        self.cancer_type_var.set(f"Cancer Type: {cancer_type}")
        self.risk_level_var.set(f"Risk Level: {risk_level}")
        num_positive = len(self.current_data['positive_segments'])
        total_segments = len(self.current_data['all_segments'])
        self.segments_info_var.set(f"Segments: {num_positive} positive / {total_segments} total")

    def update_remarks_panel(self):
        for widget in self.segments_frame_inner.winfo_children():
            widget.destroy()
        self.segment_status.clear()
        if not self.current_data:
            return
        all_segments = self.current_data['all_segments']
        positive_segment_ids = {seg_id for seg_id, _ in self.current_data['positive_segments']}
        for idx, (seg_id, imp) in enumerate(all_segments):
            frame = ttk.Frame(self.segments_frame_inner)
            frame.pack(fill=tk.X, pady=2)
            var = tk.BooleanVar(value=False)
            self.segment_status[seg_id] = var
            is_suspicious = seg_id in positive_segment_ids
            status_text = "Suspicious" if is_suspicious else "Non-Suspicious"
            chk = ttk.Checkbutton(
                frame,
                text=f'Segment S{idx+1} ({status_text}, Imp: {imp:.3f})',
                variable=var,
                command=lambda s=seg_id: self.update_segment_status(s)
            )
            chk.pack(side=tk.LEFT, padx=5)
        self.segments_canvas.update_idletasks()
        self.segments_canvas.configure(scrollregion=self.segments_canvas.bbox("all"))

    def update_segment_status(self, segment):
        if self.current_data and segment in self.segment_status:
            self.update_visualizations()

    def save_comments(self):
        self.doctor_comments = self.comments_text.get("1.0", tk.END).strip()
        messagebox.showinfo("Success", "Comments saved!")

    def update_visualizations(self):
        if not self.current_data:
            return
        self.update_lime_heatmap()
        self.update_gradcam_heatmap()
        self.update_lime_gradcam_overlay()
        self.update_contribution_plot()

    def update_lime_heatmap(self):
        self.ax_lime.clear()
        img_array = self.current_data['img_array']
        segments = self.current_data['segments']
        positive_segments = self.current_data['positive_segments']
        num_segments = len(positive_segments)
        num_to_show = max(1, int(num_segments * self.threshold_var.get() / 100))
        segments_to_show = positive_segments[:num_to_show]
        display_image = img_array.copy().astype(np.float32) / 255.0
        heatmap = np.zeros_like(display_image[:,:,0])
        if segments_to_show:
            importances = [imp for _, imp in segments_to_show]
            if importances:
                min_imp, max_imp = min(importances), max(importances)
                imp_range = max_imp - min_imp if max_imp != min_imp else 1
                for i, (seg_id, importance) in enumerate(segments_to_show):
                    mask = segments == seg_id
                    norm_imp = (importance - min_imp) / imp_range if imp_range > 0 else 1
                    heatmap[mask] = norm_imp
                    if self.show_numbers_var.get():
                        y_coords, x_coords = np.where(mask)
                        if len(y_coords) > 0:
                            centroid_y, centroid_x = int(np.mean(y_coords)), int(np.mean(x_coords))
                            self.ax_lime.text(centroid_x, centroid_y, f'S{i}',
                                             fontsize=8, fontweight='bold', color='white',
                                             ha='center', va='center',
                                             bbox=dict(facecolor='black', alpha=0.9))
                heatmap_colored = plt.cm.hot(heatmap)[:,:,:3]
                blended = 0.7 * display_image + 0.3 * heatmap_colored
                if self.show_boundaries_var.get():
                    boundary_image = np.zeros_like(blended)
                    for seg_id, _ in segments_to_show:
                        mask = segments == seg_id
                        edges = ndimage.binary_erosion(mask) ^ mask
                        boundary_image[edges] = [1, 1, 1]
                    boundary_mask = np.any(boundary_image > 0, axis=2)
                    blended[boundary_mask] = boundary_image[boundary_mask]
                display_image = blended
        self.ax_lime.imshow(display_image)
        pred_class = self.current_data['predicted_class']
        class_name = ['Non-Suspicious', 'Suspicious'][pred_class]
        confidence = self.current_data['confidence']
        cancer_type = self.current_data['cancer_type']
        risk_level = self.current_data['risk_level']
        title_text = f'LIME Heatmap: {class_name} ({confidence:.3f})\n{cancer_type} | {risk_level}\n({num_to_show}/{num_segments} segments)'
        self.ax_lime.set_title(title_text, fontsize=10, fontweight='bold', pad=15)
        self.ax_lime.axis('off')
        self.canvas_lime.draw()

    def update_gradcam_heatmap(self):
        self.ax_gradcam.clear()
        heatmap = self.current_data['gradcam_heatmap']
        heatmap = cv2.resize(heatmap, (CONFIG['IMAGE_SIZE'], CONFIG['IMAGE_SIZE']))
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        roi_array = np.array(self.current_data['roi_image'])
        overlay = roi_array * 0.6 + heatmap * 0.4
        self.ax_gradcam.imshow(overlay.astype(np.uint8))
        gradcam_class_name = 'Suspicious' if self.current_data['gradcam_class'] == 1 else 'Non-Suspicious'
        title_text = (
            f'Grad-CAM Heatmap\nPrediction: {gradcam_class_name} '
            f"(Confidence: {self.current_data['gradcam_confidence']:.3f})\n"
            f'Mouth Status: {"Open" if self.current_data["is_mouth_open"] else "Closed or Fallback"}'
        )
        self.ax_gradcam.set_title(title_text, fontsize=10, fontweight='bold', pad=15)
        self.ax_gradcam.axis('off')
        self.canvas_gradcam.draw()

    def update_lime_gradcam_overlay(self):
        self.ax_lime_gradcam.clear()
        heatmap = self.current_data['gradcam_heatmap']
        heatmap = cv2.resize(heatmap, (CONFIG['IMAGE_SIZE'], CONFIG['IMAGE_SIZE']))
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        roi_array = np.array(self.current_data['roi_image'])
        gradcam_overlay = roi_array * 0.6 + heatmap * 0.4
        display_image = gradcam_overlay.astype(np.float32) / 255.0
        segments = self.current_data['segments']
        positive_segments = self.current_data['positive_segments']
        num_segments = len(positive_segments)
        num_to_show = max(1, int(num_segments * self.threshold_var.get() / 100))
        segments_to_show = positive_segments[:num_to_show]
        if segments_to_show:
            if self.show_boundaries_var.get():
                boundary_image = np.zeros_like(display_image)
                for seg_id, _ in segments_to_show:
                    mask = segments == seg_id
                    edges = ndimage.binary_erosion(mask) ^ mask
                    boundary_image[edges] = [1, 1, 1]
                boundary_mask = np.any(boundary_image > 0, axis=2)
                display_image[boundary_mask] = boundary_image[boundary_mask]
            if self.show_numbers_var.get():
                for i, (seg_id, _) in enumerate(segments_to_show):
                    mask = segments == seg_id
                    y_coords, x_coords = np.where(mask)
                    if len(y_coords) > 0:
                        centroid_y, centroid_x = int(np.mean(y_coords)), int(np.mean(x_coords))
                        self.ax_lime_gradcam.text(
                            centroid_x, centroid_y, f'S{i}',
                            fontsize=8, fontweight='bold', color='white',
                            ha='center', va='center',
                            bbox=dict(facecolor='black', alpha=0.9)
                        )
        self.ax_lime_gradcam.imshow(display_image)
        pred_class = self.current_data['predicted_class']
        class_name = ['Non-Suspicious', 'Suspicious'][pred_class]
        confidence = self.current_data['confidence']
        title_text = f'LIME Segments on Grad-CAM: {class_name} ({confidence:.3f})\nSegments Shown: {num_to_show}/{num_segments}'
        self.ax_lime_gradcam.set_title(title_text, fontsize=10, fontweight='bold', pad=15)
        self.ax_lime_gradcam.axis('off')
        self.canvas_lime_gradcam.draw()

    def update_contribution_plot(self):
        self.ax_contrib.clear()
        positive_segments = self.current_data['positive_segments']
        num_to_show = max(1, int(len(positive_segments) * self.threshold_var.get() / 100))
        if not positive_segments:
            self.ax_contrib.text(0.5, 0.5, 'No positive contributors found',
                                ha='center', va='center', fontsize=12,
                                bbox=dict(facecolor='lightgray'))
            self.canvas_contrib.draw()
            return
        segments_to_plot = positive_segments[:num_to_show]
        segment_labels = [f'S{i}' for i in range(len(segments_to_plot))]
        importances = [imp for _, imp in segments_to_plot]
        colors = ['#FF3333' if i < num_to_show//3 else '#FF9933' if i < 2*num_to_show//3 else '#FFCC33'
                  for i in range(len(segments_to_plot))]
        bars = self.ax_contrib.bar(segment_labels, importances, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
        self.ax_contrib.set_ylabel('Contribution Score', fontsize=11, fontweight='bold')
        self.ax_contrib.set_xlabel('Segments', fontsize=11, fontweight='bold')
        pred_class = self.current_data['predicted_class']
        class_name = ['Non-Suspicious', 'Suspicious'][pred_class]
        confidence = self.current_data['confidence']
        cancer_type = self.current_data['cancer_type']
        risk_level = self.current_data['risk_level']
        title_text = f'Contributing Segments\n{class_name} ({confidence:.3f})\n{cancer_type}\nRisk: {risk_level}'
        self.ax_contrib.set_title(title_text, fontsize=10, fontweight='bold', pad=20)
        for i, (bar, importance) in enumerate(zip(bars, importances)):
            height = bar.get_height()
            self.ax_contrib.text(bar.get_x() + bar.get_width()/2, height + max(importances) * 0.01,
                                f'{importance:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
        self.ax_contrib.spines['top'].set_visible(False)
        self.ax_contrib.spines['right'].set_visible(False)
        self.ax_contrib.grid(axis='y', alpha=0.3, linestyle='--')
        plt.figure(self.fig_contrib.number)
        plt.tight_layout()
        self.canvas_contrib.draw()

    def save_results(self):
        if not self.current_data:
            messagebox.showerror("Error", "No analysis to save!")
            return
        try:
            results_dir = os.path.join(CONFIG['RESULTS_DIR'], 'lime_gradcam_results')
            os.makedirs(results_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_filename = f"{self.image_name}_{timestamp}"
            # Save visualization images
            self.fig_lime.savefig(os.path.join(results_dir, f'{base_filename}_lime.png'), dpi=300, bbox_inches='tight')
            self.fig_gradcam.savefig(os.path.join(results_dir, f'{base_filename}_gradcam.png'), dpi=300, bbox_inches='tight')
            self.fig_lime_gradcam.savefig(os.path.join(results_dir, f'{base_filename}_lime_gradcam_overlay.png'), dpi=300, bbox_inches='tight')
            self.fig_contrib.savefig(os.path.join(results_dir, f'{base_filename}_contributions.png'), dpi=300, bbox_inches='tight')
            # Prepare results dictionary
            results = {
                'image_name': self.image_name,
                'timestamp': timestamp,
                'prediction': {
                    'class': ['Non-Suspicious', 'Suspicious'][self.current_data['predicted_class']],
                    'confidence': float(self.current_data['confidence']),
                    'non_suspicious_prob': float(self.current_data['non_suspicious_prob']),
                    'suspicious_prob': float(self.current_data['suspicious_prob']),
                    'cancer_type': self.current_data['cancer_type'],
                    'risk_level': self.current_data['risk_level'],
                    'gradcam_class': ['Non-Suspicious', 'Suspicious'][self.current_data['gradcam_class']],
                    'gradcam_confidence': float(self.current_data['gradcam_confidence'])
                },
                'segments': {
                    'total': len(self.current_data['all_segments']),
                    'positive': [{
                        'id': f'S{i+1}',
                        'segment_id': seg_id,
                        'importance': float(imp),
                        'doctor_status': self.segment_status.get(seg_id, tk.BooleanVar(value=False)).get()
                    } for i, (seg_id, imp) in enumerate(self.current_data['all_segments'])]
                },
                'doctor_comments': self.doctor_comments,
                'brightness_adjust': float(self.brightness_var.get()),
                'num_segments': self.segments_var.get(),
                'is_mouth_open': self.current_data['is_mouth_open']
            }
            # Save JSON
            json_path = os.path.join(results_dir, f'{base_filename}_results.json')
            with open(json_path, 'w') as f:
                json.dump(results, f, indent=2)
            # Save report
            report_path = os.path.join(results_dir, f'{base_filename}_report.txt')
            with open(report_path, 'w') as f:
                f.write("LIME & Grad-CAM Analysis Report\n")
                f.write("=" * 50 + "\n\n")
                f.write(f"Image: {self.image_name}\n")
                f.write(f"Prediction: {results['prediction']['class']}\n")
                f.write(f"Confidence: {results['prediction']['confidence']:.4f}\n")
                f.write(f"Grad-CAM Prediction: {results['prediction']['gradcam_class']}\n")
                f.write(f"Grad-CAM Confidence: {results['prediction']['gradcam_confidence']:.4f}\n")
                f.write(f"Cancer Type: {results['prediction']['cancer_type']}\n")
                f.write(f"Risk Level: {results['prediction']['risk_level']}\n")
                f.write(f"Non-Suspicious Probability: {results['prediction']['non_suspicious_prob']:.4f}\n")
                f.write(f"Suspicious Probability: {results['prediction']['suspicious_prob']:.4f}\n")
                f.write(f"Brightness Adjustment: {results['brightness_adjust']:.2f}\n")
                f.write(f"Mouth Status: {'Open' if results['is_mouth_open'] else 'Closed or Fallback'}\n\n")
                f.write("Doctor's Comments:\n")
                f.write(f"{self.doctor_comments if self.doctor_comments else 'No comments provided'}\n\n")
                f.write("Top Contributing Segments:\n")
                f.write("-" * 30 + "\n")
                for seg in results['segments']['positive']:
                    f.write(f"{seg['id']}: Segment {seg['segment_id']}, Importance: {seg['importance']:.4f}, "
                           f"Doctor Status: {'Non-Suspicious' if seg['doctor_status'] else 'Suspicious'}\n")
            messagebox.showinfo("Success", f"Results saved to {results_dir}/")
            self.status_var.set(f"Results saved to {results_dir}/")
        except Exception as e:
            logging.error(f"Failed to save results: {str(e)}")
            messagebox.showerror("Error", f"Failed to save results: {str(e)}")

if __name__ == "__main__":
    root = tk.Tk()
    app = EnhancedLIMEGradCAMVisualization(root)
    root.mainloop()