import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import efficientnet_b4, EfficientNet_B4_Weights
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
from lime import lime_image
from skimage.segmentation import slic, mark_boundaries
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import os
import json
from datetime import datetime
import cv2
import warnings
import scipy.ndimage as ndimage
warnings.filterwarnings('ignore')

CONFIG = {
    'MODEL_PATH': '/home/karthik/oral-cancer-detection/models/trained_efficientnet_b4_oral_cancer.pth',
    'IMAGE_SIZE': 300,
    'DEVICE': torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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
        return self.backbone(x)

class LIMEExplainer:
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.mean = np.array([0.485, 0.456, 0.406])
        self.std = np.array([0.229, 0.224, 0.225])
        self.transform = transforms.Compose([
            transforms.Resize((CONFIG['IMAGE_SIZE'], CONFIG['IMAGE_SIZE'])),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.cached_explanation = None
        self.cached_img_array = None
    
    def detect_mouth_region(self, image):
        img_array = np.array(image)
        hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
        lower_red1 = np.array([0, 40, 40])
        upper_red1 = np.array([15, 255, 255])
        lower_red2 = np.array([160, 40, 40])
        upper_red2 = np.array([180, 255, 255])
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask = mask1 + mask2
        kernel = np.ones((3,3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        return mask > 0
    
    def detect_reflections(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(blurred, 220, 255, cv2.THRESH_BINARY)
        kernel = np.ones((3,3), np.uint8)
        thresh = cv2.dilate(thresh, kernel, iterations=1)
        return thresh == 0
    
    def enhanced_segmentation(self, image, n_segments=200):
        mouth_mask = self.detect_mouth_region(Image.fromarray(image.astype(np.uint8)))
        reflection_mask = self.detect_reflections(image)
        combined_mask = np.logical_and(mouth_mask, reflection_mask)
        
        segments = slic(
            image,
            n_segments=n_segments,
            compactness=10,
            start_label=1,
            mask=combined_mask,
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
    
    def explain_image(self, image_path, n_segments=200, brightness_adjust=1.0, use_cache=True):
        original_image = Image.open(image_path).convert('RGB')
        
        img_array = np.array(original_image)
        img_array = np.clip(img_array * brightness_adjust, 0, 255).astype(np.uint8)
        adjusted_image = Image.fromarray(img_array)
        
        resized_image = adjusted_image.resize((CONFIG['IMAGE_SIZE'], CONFIG['IMAGE_SIZE']))
        img_array = np.array(resized_image)
        
        img_tensor = self.transform(resized_image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            output = self.model(img_tensor)
            probs = torch.nn.functional.softmax(output, dim=1)
            predicted_class = torch.argmax(probs, dim=1).item()
            confidence = probs.max().item()
            non_suspicious_prob = probs[0][0].item()
            suspicious_prob = probs[0][1].item()
        
        # Check if we can use cached explanation
        if use_cache and self.cached_explanation is not None and np.array_equal(self.cached_img_array, img_array):
            explanation = self.cached_explanation
        else:
            explainer = lime_image.LimeImageExplainer()
            explanation = explainer.explain_instance(
                img_array,
                self.predict_fn,
                top_labels=2,
                hide_color=0,
                num_samples=1500,
                segmentation_fn=lambda x: self.enhanced_segmentation(x, n_segments)
            )
            self.cached_explanation = explanation
            self.cached_img_array = img_array.copy()
        
        segments = self.enhanced_segmentation(img_array, n_segments)
        feature_importance = explanation.local_exp[predicted_class]
        
        positive_segments = [(seg_id, imp) for seg_id, imp in feature_importance if imp > 0]
        positive_segments.sort(key=lambda x: x[1], reverse=True)
        
        negative_segments = [(seg_id, imp) for seg_id, imp in feature_importance if imp < 0]
        negative_segments.sort(key=lambda x: x[1])
        
        cancer_type = self.determine_cancer_type(suspicious_prob, predicted_class)
        risk_level = self.determine_risk_level(suspicious_prob, predicted_class)
        
        return {
            'image': resized_image,
            'img_array': img_array,
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

class EnhancedLIMEVisualization:
    def __init__(self, root):
        self.root = root
        self.root.title("Enhanced LIME Visualization - Oral Cancer Detection")
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
        self.contribution_fig = None
        self.contribution_ax = None
        self.contribution_canvas = None
        
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
        
        param_frame = ttk.Frame(control_frame)
        param_frame.pack(fill=tk.X)
        
        ttk.Label(param_frame, text="Segments:").grid(row=0, column=0, sticky=tk.W, padx=(0, 5))
        segments_scale = ttk.Scale(param_frame, from_=100, to=300, variable=self.segments_var,
                                  orient=tk.HORIZONTAL, length=150, command=self.update_segments)
        segments_scale.grid(row=0, column=1, padx=5)
        self.segments_label = ttk.Label(param_frame, text="200")
        self.segments_label.grid(row=0, column=2, padx=(5, 20))
        
        ttk.Label(param_frame, text="Top Contributors (%):").grid(row=0, column=3, sticky=tk.W, padx=(0, 5))
        threshold_scale = ttk.Scale(param_frame, from_=10, to=100, variable=self.threshold_var,
                                   orient=tk.HORIZONTAL, length=150, command=self.update_threshold)
        threshold_scale.grid(row=0, column=4, padx=5)
        self.threshold_label = ttk.Label(param_frame, text="100%")
        self.threshold_label.grid(row=0, column=5, padx=(5, 20))
        
        ttk.Label(param_frame, text="Brightness:").grid(row=0, column=6, sticky=tk.W, padx=(0, 5))
        brightness_scale = ttk.Scale(param_frame, from_=0.5, to=2.0, variable=self.brightness_var,
                                    orient=tk.HORIZONTAL, length=150, command=self.update_brightness)
        brightness_scale.grid(row=0, column=7, padx=5)
        self.brightness_label = ttk.Label(param_frame, text="1.0")
        self.brightness_label.grid(row=0, column=8, padx=(5, 20))
        
        ttk.Checkbutton(param_frame, text="Show Numbers", variable=self.show_numbers_var,
                       command=self.update_display_options).grid(row=0, column=9, padx=5)
        ttk.Checkbutton(param_frame, text="Show Boundaries", variable=self.show_boundaries_var,
                       command=self.update_display_options).grid(row=0, column=10, padx=5)
    
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
        
        # LIME Heatmap Tab
        panel2 = ttk.Frame(self.notebook)
        self.notebook.add(panel2, text="LIME Heatmap")
        self.fig2 = plt.figure(figsize=(5, 5), facecolor='white')
        self.ax2 = self.fig2.add_subplot(111)
        self.canvas2 = FigureCanvasTkAgg(self.fig2, panel2)
        self.canvas2.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Contribution Analysis Tab
        panel3 = ttk.Frame(self.notebook)
        self.notebook.add(panel3, text="Contribution Analysis")
        self.contribution_fig = plt.figure(figsize=(5, 5), facecolor='white')
        self.contribution_ax = self.contribution_fig.add_subplot(111)
        self.contribution_canvas = FigureCanvasTkAgg(self.contribution_fig, panel3)
        self.contribution_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
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
        
        segments_frame = ttk.Frame(remarks_frame)
        segments_frame.pack(fill=tk.X)
        
        ttk.Label(segments_frame, text="Segment Status (Check for Non-Suspicious):").pack(side=tk.LEFT)
        self.segments_canvas = tk.Canvas(segments_frame, height=150)
        self.segments_frame_inner = ttk.Frame(self.segments_canvas)
        self.segments_scrollbar = ttk.Scrollbar(segments_frame, orient=tk.VERTICAL, command=self.segments_canvas.yview)
        self.segments_canvas.configure(yscrollcommand=self.segments_scrollbar.set)
        
        self.segments_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.segments_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.segments_canvas.create_window((0,0), window=self.segments_frame_inner, anchor='nw')
        
        def on_frame_configure(event):
            self.segments_canvas.configure(scrollregion=self.segments_canvas.bbox("all"))
        self.segments_frame_inner.bind('<Configure>', on_frame_configure)
        
        comments_frame = ttk.Frame(remarks_frame)
        comments_frame.pack(fill=tk.X, pady=(10, 0))
        
        ttk.Label(comments_frame, text="Doctor's Comments:").pack(anchor=tk.W)
        self.comments_text = tk.Text(comments_frame, height=4, width=80)
        self.comments_text.pack(fill=tk.X, pady=(5, 5))
        ttk.Button(comments_frame, text="Save Comments", command=self.save_comments).pack(anchor=tk.W)
    
    def load_model(self):
        try:
            if not os.path.exists(CONFIG['MODEL_PATH']):
                self.status_var.set("Model file not found!")
                return
            self.model = EnhancedEfficientNetB4(num_classes=2)
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
            self.explainer.cached_explanation = None  # Clear cache on new image
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
        self.explainer.cached_explanation = None  # Clear cache on image change
        self.analyze_image()
    
    def analyze_image(self):
        if not hasattr(self, 'current_image_path'):
            messagebox.showerror("Error", "Please load an image first!")
            return
        
        self.status_var.set("Analyzing image...")
        self.root.update()
        
        try:
            n_segments = self.segments_var.get()
            brightness_adjust = self.brightness_var.get()
            self.current_data = self.explainer.explain_image(
                self.current_image_path, n_segments, brightness_adjust, use_cache=False
            )
            self.update_visualizations()
            self.update_info_panel()
            self.update_remarks_panel()
            
            pred_class = self.current_data['predicted_class']
            confidence = self.current_data['confidence']
            class_name = ['Non-Suspicious', 'Suspicious'][pred_class]
            self.status_var.set(f"Analysis complete: {class_name} ({confidence:.3f})")
        except Exception as e:
            messagebox.showerror("Error", f"Analysis failed: {str(e)}")
            self.status_var.set("Analysis failed!")
    
    def update_segments(self, value):
        segments = int(float(value))
        self.segments_label.config(text=str(segments))
        if self.current_data:
            self.update_segment_visualizations(segments)
    
    def update_segment_visualizations(self, n_segments):
        try:
            self.status_var.set("Updating segment visualizations...")
            self.root.update()
            
            # Re-run explanation with cached data
            self.current_data = self.explainer.explain_image(
                self.current_image_path,
                n_segments=n_segments,
                brightness_adjust=self.brightness_var.get(),
                use_cache=True
            )
            self.update_visualizations()
            self.update_remarks_panel()
            self.status_var.set("Segment visualizations updated")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to update segments: {str(e)}")
            self.status_var.set("Segment update failed")
    
    def update_threshold(self, value):
        percent = int(float(value))
        self.threshold_label.config(text=f"{percent}%")
        if self.current_data:
            self.update_visualizations()
    
    def update_brightness(self, value):
        brightness = float(value)
        self.brightness_label.config(text=f"{brightness:.2f}")
        if hasattr(self, 'current_image_path'):
            self.explainer.cached_explanation = None  # Clear cache on brightness change
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
        
        if not self.current_data:
            return
        
        positive_segments = self.current_data['positive_segments']
        for idx, (seg_id, imp) in enumerate(positive_segments):
            frame = ttk.Frame(self.segments_frame_inner)
            frame.pack(fill=tk.X, pady=2)
            
            var = tk.BooleanVar(value=self.segment_status.get(f'S{idx}', False))
            self.segment_status[f'S{idx}'] = var
            
            chk = ttk.Checkbutton(frame, text=f'S{idx} (Imp: {imp:.3f})', variable=var,
                                 command=lambda s=f'S{idx}': self.update_segment_status(s))
            chk.pack(side=tk.LEFT)
    
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
        self.update_contribution_plot()
    
    def update_lime_heatmap(self):
        self.ax2.clear()
        
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
                            self.ax2.text(centroid_x, centroid_y, f'S{i}',
                                        fontsize=8, fontweight='bold', color='white',
                                        ha='center', va='center',
                                        bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.9))
                
                from matplotlib import cm
                heatmap_colored = cm.hot(heatmap)
                heatmap_colored = heatmap_colored[:,:,:3]
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
        
        self.ax2.imshow(display_image)
        
        pred_class = self.current_data['predicted_class']
        class_name = ['Non-Suspicious', 'Suspicious'][pred_class]
        confidence = self.current_data['confidence']
        cancer_type = self.current_data['cancer_type']
        risk_level = self.current_data['risk_level']
        
        title_text = f'LIME Heatmap: {class_name} ({confidence:.3f})\n{cancer_type} | {risk_level}\n({num_to_show}/{num_segments} segments shown)'
        self.ax2.set_title(title_text, fontsize=10, fontweight='bold', pad=15)
        self.ax2.axis('off')
        self.canvas2.draw()
    
    def update_contribution_plot(self):
        self.contribution_ax.clear()
        
        positive_segments = self.current_data['positive_segments']
        num_to_show = max(1, int(len(positive_segments) * self.threshold_var.get() / 100))
        
        if not positive_segments:
            self.contribution_ax.text(0.5, 0.5, 'No positive contributors found',
                                     ha='center', va='center', fontsize=12,
                                     bbox=dict(boxstyle='round', facecolor='lightgray'))
            self.contribution_canvas.draw()
            return
        
        segments_to_plot = positive_segments[:num_to_show]
        segment_labels = [f'S{i}' for i in range(len(segments_to_plot))]
        importances = [imp for _, imp in segments_to_plot]
        
        colors = ['#FF3333' if i < num_to_show//3 else '#FF9933' if i < 2*num_to_show//3 else '#FFCC33'
                  for i in range(len(segments_to_plot))]
        
        bars = self.contribution_ax.bar(segment_labels, importances, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
        
        self.contribution_ax.set_ylabel('Contribution Score', fontsize=11, fontweight='bold')
        self.contribution_ax.set_xlabel('Segments', fontsize=11, fontweight='bold')
        
        pred_class = self.current_data['predicted_class']
        class_name = ['Non-Suspicious', 'Suspicious'][pred_class]
        confidence = self.current_data['confidence']
        cancer_type = self.current_data['cancer_type']
        risk_level = self.current_data['risk_level']
        
        title_text = f'Contributing Segments\n{class_name} ({confidence:.3f})\n{cancer_type}\nRisk: {risk_level}'
        self.contribution_ax.set_title(title_text, fontsize=10, fontweight='bold', pad=20)
        
        for i, (bar, importance) in enumerate(zip(bars, importances)):
            height = bar.get_height()
            self.contribution_ax.text(bar.get_x() + bar.get_width()/2, height + max(importances) * 0.01,
                                     f'{importance:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        self.contribution_ax.spines['top'].set_visible(False)
        self.contribution_ax.spines['right'].set_visible(False)
        self.contribution_ax.spines['left'].set_color('#CCCCCC')
        self.contribution_ax.spines['bottom'].set_color('#CCCCCC')
        self.contribution_ax.tick_params(colors='#666666')
        self.contribution_ax.grid(axis='y', alpha=0.3, linestyle='--')
        
        plt.figure(self.contribution_fig.number)
        plt.tight_layout()
        self.contribution_canvas.draw()
    
    def save_results(self):
        if not self.current_data:
            messagebox.showerror("Error", "No analysis to save!")
            return
        
        try:
            results_dir = "lime_results"
            os.makedirs(results_dir, exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_filename = f"{self.image_name}_{timestamp}"
            
            self.fig2.savefig(f"{results_dir}/{base_filename}_lime.png", dpi=300, bbox_inches='tight')
            self.contribution_fig.savefig(f"{results_dir}/{base_filename}_contributions.png", dpi=300, bbox_inches='tight')
            
            results = {
                'image_name': self.image_name,
                'timestamp': timestamp,
                'prediction': {
                    'class': ['Non-Suspicious', 'Suspicious'][self.current_data['predicted_class']],
                    'confidence': float(self.current_data['confidence']),
                    'non_suspicious_prob': float(self.current_data['non_suspicious_prob']),
                    'suspicious_prob': float(self.current_data['suspicious_prob']),
                    'cancer_type': self.current_data['cancer_type'],
                    'risk_level': self.current_data['risk_level']
                },
                'segments': {
                    'total': len(self.current_data['all_segments']),
                    'positive': [{
                        'id': f'S{i}',
                        'segment_id': seg_id,
                        'importance': float(imp),
                        'doctor_status': bool(self.segment_status.get(f'S{i}', tk.BooleanVar(value=False)).get())
                    } for i, (seg_id, imp) in enumerate(self.current_data['positive_segments'])]
                },
                'doctor_comments': self.doctor_comments,
                'brightness_adjust': float(self.brightness_var.get()),
                'num_segments': self.segments_var.get()
            }
            
            json_path = f"{results_dir}/{base_filename}_results.json"
            with open(json_path, 'w') as f:
                json.dump(results, f, indent=2)
            
            report_path = f"{results_dir}/{base_filename}_report.txt"
            with open(report_path, 'w') as f:
                f.write("LIME Analysis Report\n")
                f.write("=" * 50 + "\n\n")
                f.write(f"Image: {self.image_name}\n")
                f.write(f"Prediction: {results['prediction']['class']}\n")
                f.write(f"Confidence: {results['prediction']['confidence']:.4f}\n")
                f.write(f"Cancer Type: {results['prediction']['cancer_type']}\n")
                f.write(f"Risk Level: {results['prediction']['risk_level']}\n")
                f.write(f"Non-Suspicious Probability: {results['prediction']['non_suspicious_prob']:.4f}\n")
                f.write(f"Suspicious Probability: {results['prediction']['suspicious_prob']:.4f}\n")
                f.write(f"Brightness Adjustment: {results['brightness_adjust']:.2f}\n\n")
                f.write("Doctor's Comments:\n")
                f.write(f"{self.doctor_comments}\n\n")
                f.write("Top Contributing Segments:\n")
                f.write("-" * 30 + "\n")
                for seg in results['segments']['positive']:
                    f.write(f"{seg['id']}: Segment {seg['segment_id']}, Importance: {seg['importance']:.4f}, "
                           f"Doctor Status: {'Non-Suspicious' if seg['doctor_status'] else 'Suspicious'}\n")
            
            messagebox.showinfo("Success", f"Results saved to {results_dir}/")
            self.status_var.set(f"Results saved to {results_dir}/")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save results: {str(e)}")

if __name__ == "__main__":
    root = tk.Tk()
    app = EnhancedLIMEVisualization(root)
    root.mainloop()