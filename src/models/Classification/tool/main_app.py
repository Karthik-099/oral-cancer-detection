import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import cv2
import os
import json
from datetime import datetime
import logging
import scipy.ndimage as ndimage
from PIL import Image, ImageTk
import torchvision.transforms as transforms
import warnings
import torch
from efficientnet_model import load_model, EnhancedEfficientNetB4
from gradcam import GradCAM
from lime_explainer import LIMEExplainer, detect_intraoral_roi

warnings.filterwarnings('ignore')

# Setup base directory
BASE_DIR = '/home/user/Desktop/oral_cancer/gdc_oral_cancer_dataset/oral-cancer-detection'

# Create results directories before logging
RESULTS_DIR = os.path.join(BASE_DIR, 'src/models/Classification/lime_results')
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
    'RESULTS_DIR': RESULTS_DIR,
    'IMAGE_SIZE': 300,
    'DEVICE': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    'HEATMAP_THRESHOLD': 0.3,
}

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
        segments_frame = ttk.Frame(remarks_frame)
        segments_frame.pack(fill=tk.X)
        ttk.Label(segments_frame, text="Segment Status (Check for Non-Suspicious):").pack(side=tk.LEFT)
        self.segments_canvas = tk.Canvas(segments_frame, height=100)
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
        ttk.Label(comments_frame, text="Comments:").pack(anchor=tk.W)
        self.comments_text = tk.Text(comments_frame, height=4)
        self.comments_text.pack(fill=tk.X)
        ttk.Button(comments_frame, text="Save Comments", command=self.save_comments).pack(anchor=tk.W, pady=(5, 0))

    def load_model(self):
        try:
            self.model = load_model()
            self.explainer = LIMEExplainer(self.model)
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

    def analyze_image(self):
        if not hasattr(self, 'current_image_path'):
            messagebox.showerror("Error", "Please load an image first!")
            return
        self.status_var.set("Analyzing image...")
        self.root.update()
        try:
            n_segments = self.segments_var.get()
            brightness_adjust = self.brightness_var.get()
            # LIME Analysis
            self.current_data = self.explainer.explain_image(
                self.current_image_path, n_segments, brightness_adjust
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
            self.update_visualizations()
            self.update_info_panel()
            self.update_remarks_panel()
            self.generate_combined_visualization()
            pred_class = self.current_data['predicted_class']
            confidence = self.current_data['confidence']
            class_name = ['Non-Suspicious', 'Suspicious'][pred_class]
            self.status_var.set(f"Analysis complete: {class_name} ({confidence:.3f})")
        except Exception as e:
            logging.error(f"Analysis failed: {str(e)}")
            messagebox.showerror("Error", f"Analysis failed: {str(e)}")
            self.status_var.set("Analysis failed!")

    def generate_combined_visualization(self):
        plt.ioff()
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 12))
        # Original Image
        original_image = Image.open(self.current_image_path).convert('RGB')
        original_array = np.array(original_image)
        if self.current_data['bbox']:
            x, y, w, h = self.current_data['bbox']
            cv2.rectangle(original_array, (x, y), (x+w, y+h), (0, 255, 0), 2)
        ax1.imshow(original_array)
        ax1.set_title('Original Image with ROI', fontsize=10)
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
        self.segments_label.configure(text=str(segments))
        if hasattr(self, 'current_image_path'):
            self.analyze_image()

    def update_threshold(self, value):
        percent = int(float(value))
        self.threshold_label.configure(text=f"{percent}%")
        if self.current_data:
            self.update_visualizations()

    def update_brightness(self, value):
        brightness = float(value)
        self.brightness_label.configure(text=f"{brightness:.2f}")
        if hasattr(self, 'current_image_path'):
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
        self.segment_status.clear()  # Clear previous segment status
        for idx, (seg_id, imp) in enumerate(positive_segments[:10]):
            frame = ttk.Frame(self.segments_frame_inner)
            frame.pack(fill=tk.X, pady=2)
            var = tk.BooleanVar(value=False)
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
            f'(Confidence: {self.current_data['gradcam_confidence']:.3f})\n'
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
                for i, (seg_id, _) in enumerate(segments_to_show[:10]):
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
            self.fig_lime.savefig(os.path.join(results_dir, f'{base_filename}_lime.png'), dpi=300, bbox_inches='tight')
            self.fig_gradcam.savefig(os.path.join(results_dir, f'{base_filename}_gradcam.png'), dpi=300, bbox_inches='tight')
            self.fig_lime_gradcam.savefig(os.path.join(results_dir, f'{base_filename}_lime_gradcam_overlay.png'), dpi=300, bbox_inches='tight')
            self.fig_contrib.savefig(os.path.join(results_dir, f'{base_filename}_contributions.png'), dpi=300, bbox_inches='tight')
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
                        'id': f'S{i}',
                        'segment_id': seg_id,
                        'importance': float(imp),
                        'doctor_status': self.segment_status.get(f'S{i}', tk.BooleanVar(value=False)).get()
                    } for i, (seg_id, imp) in enumerate(self.current_data['positive_segments'])]
                },
                'doctor_comments': self.doctor_comments,
                'brightness_adjust': float(self.brightness_var.get()),
                'num_segments': self.segments_var.get(),
                'is_mouth_open': self.current_data['is_mouth_open']
            }
            json_path = os.path.join(results_dir, f'{base_filename}_results.json')
            with open(json_path, 'w') as f:
                json.dump(results, f, indent=2)
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
                f.write(f"{self.doctor_comments}\n\n")
                f.write("Top Contributing Segments:\n")
                f.write("-" * 30 + "\n")
                for seg in results['segments']['positive'][:10]:
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