import torch
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, colorchooser
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
from PIL import Image, ImageTk, ImageDraw
import cv2
import os
import json
from datetime import datetime
import logging
import scipy.ndimage as ndimage
from config import CONFIG
from models import EnhancedEfficientNetB4
from gradcam import GradCAM
from lime_explainer import LIMEExplainer
from utils import detect_intraoral_roi, normalize_path
import torchvision.transforms as transforms

class EnhancedLIMEGradCAMVisualization:
    def __init__(self, root):
        self.root = root
        self.root.title("LIME & GradCAM Visualization - Oral Cancer Detection")
        self.root.geometry("1200x800")
        self.root.configure(bg='#f0f0f0')
        self.model = None
        self.explainer = None
        self.current_data = None
        self.threshold_var = tk.IntVar(value=100)
        self.segments_var = tk.IntVar(value=200)
        self.show_numbers_var = tk.BooleanVar(value=True)
        self.show_boundaries_var = tk.BooleanVar(value=True)
        self.hotspot_segments_var = tk.IntVar(value=20)
        self.segment_status = {}
        self.doctor_comments = ""
        self.paint_mode = tk.BooleanVar(value=False)
        self.brush_color = '#FF0000'
        self.brush_size = tk.IntVar(value=5)
        self.drawing = False
        self.last_x = None
        self.last_y = None
        self.draw_image = None
        self.draw_photo = None
        self.draw_paths = []
        self.hotspots = []
        self.hotspot_counter = 0
        self.manual_segment = []
        self.setup_ui()
        self.load_model()

    def setup_ui(self):
        style = ttk.Style()
        style.theme_use('clam')
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.setup_control_panel(main_frame)
        self.setup_tabbed_panels(main_frame)
        self.setup_info_panel(main_frame)
        self.setup_remarks_panel(main_frame)
        main_frame.rowconfigure(1, weight=1)

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
        segments_scale = ttk.Scale(param_frame, from_=100, to_=300, variable=self.segments_var,
                                  orient=tk.HORIZONTAL, length=150, command=self.update_segments)
        segments_scale.grid(row=0, column=1, padx=5)
        self.segments_entry = ttk.Entry(param_frame, width=5, textvariable=self.segments_var)
        self.segments_entry.grid(row=0, column=2, padx=(5, 20))
        self.segments_entry.bind('<Return>', self.update_segments_entry)
        
        ttk.Label(param_frame, text="Top Contributors (%):").grid(row=0, column=3, sticky=tk.W, padx=(0, 5))
        threshold_scale = ttk.Scale(param_frame, from_=10, to=100, variable=self.threshold_var,
                                   orient=tk.HORIZONTAL, length=150, command=self.update_threshold)
        threshold_scale.grid(row=0, column=4, padx=5)
        self.threshold_entry = ttk.Entry(param_frame, width=5, textvariable=self.threshold_var)
        self.threshold_entry.grid(row=0, column=5, padx=(5, 20))
        self.threshold_entry.bind('<Return>', self.update_threshold_entry)
        
        ttk.Checkbutton(param_frame, text="Show Numbers", variable=self.show_numbers_var,
                       command=self.update_visualizations).grid(row=0, column=6, padx=5)
        ttk.Checkbutton(param_frame, text="Show Boundaries", variable=self.show_boundaries_var,
                        command=self.update_visualizations).grid(row=0, column=7, padx=5)

        paint_frame = ttk.Frame(control_frame)
        paint_frame.pack(fill=tk.X, pady=(10, 0))
        ttk.Checkbutton(paint_frame, text="Paint Mode", variable=self.paint_mode,
                        command=self.toggle_paint_mode).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(paint_frame, text="Choose Color", command=self.choose_color).pack(side=tk.LEFT, padx=5)
        ttk.Button(paint_frame, text="Clear Painting", command=self.clear_painting).pack(side=tk.LEFT, padx=5)
        ttk.Label(paint_frame, text="Brush Size:").pack(side=tk.LEFT, padx=(10, 5))
        brush_scale = ttk.Scale(paint_frame, from_=1, to=20, variable=self.brush_size,
                                orient=tk.HORIZONTAL, length=100, command=lambda x: self.brush_size.set(int(float(x))))
        brush_scale.pack(side=tk.LEFT, padx=5)
        self.brush_size_entry = ttk.Entry(paint_frame, width=5, textvariable=self.brush_size)
        self.brush_size_entry.pack(side=tk.LEFT, padx=5)
        self.brush_size_entry.bind('<Return>', self.update_brush_size_entry)

    def setup_tabbed_panels(self, parent):
        self.notebook = ttk.Notebook(parent)
        self.notebook.pack(fill=tk.BOTH, expand=True, pady=(0, 20))
        panel1 = ttk.Frame(self.notebook)
        self.notebook.add(panel1, text="Original Image")
        self.canvas1 = tk.Canvas(panel1, bg='white', relief=tk.SUNKEN, bd=1)
        self.canvas1.pack(fill=tk.BOTH, expand=True)
        panel2 = ttk.Frame(self.notebook)
        self.notebook.add(panel2, text="LIME Segmentation")
        self.fig_lime = plt.figure(figsize=(4, 4), facecolor='white')
        self.ax_lime = self.fig_lime.add_subplot(111)
        self.canvas_lime = FigureCanvasTkAgg(self.fig_lime, panel2)
        self.canvas_lime.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        panel3 = ttk.Frame(self.notebook)
        self.notebook.add(panel3, text="Grad-CAM Heatmap")
        self.fig_gradcam = plt.figure(figsize=(4, 4), facecolor='white')
        self.ax_gradcam = self.fig_gradcam.add_subplot(111)
        self.canvas_gradcam = FigureCanvasTkAgg(self.fig_gradcam, panel3)
        self.canvas_gradcam.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        panel4 = ttk.Frame(self.notebook)
        self.notebook.add(panel4, text="LIME on Grad-CAM Overlay")
        self.fig_lime_gradcam = plt.figure(figsize=(5, 5), facecolor='white')
        self.ax_lime_gradcam = self.fig_lime_gradcam.add_subplot(111)
        self.canvas_lime_gradcam = FigureCanvasTkAgg(self.fig_lime_gradcam, panel4)
        self.canvas_lime_gradcam.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        hotspot_frame = ttk.Frame(panel4)
        hotspot_frame.pack(fill=tk.X, pady=(5, 0))
        ttk.Label(hotspot_frame, text="Hotspot Segments:").pack(side=tk.LEFT, padx=(0, 5))
        hotspot_scale = ttk.Scale(
            hotspot_frame, from_=1, to=10, variable=self.hotspot_segments_var,
            orient=tk.HORIZONTAL, length=200, command=self.update_hotspot_segments
        )
        hotspot_scale.pack(side=tk.LEFT, padx=(5, 5))
        self.hotspot_segments_entry = ttk.Entry(hotspot_frame, width=5, textvariable=self.hotspot_segments_var)
        self.hotspot_segments_entry.pack(side=tk.LEFT, padx=5)
        self.hotspot_segments_entry.bind('<Return>', self.update_hotspot_segments_entry)
        self.hotspot_segments_label = ttk.Label(hotspot_frame, text=str(self.hotspot_segments_var.get()))
        self.hotspot_segments_label.pack(side=tk.LEFT, padx=5)
        panel5 = ttk.Frame(self.notebook)
        self.notebook.add(panel5, text="Contribution Analysis")
        self.fig_contrib = plt.figure(figsize=(4, 4), facecolor='white')
        self.ax_contrib = self.fig_contrib.add_subplot(111)
        self.canvas_contrib = FigureCanvasTkAgg(self.fig_contrib, panel5)
        self.canvas_contrib.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        panel6 = ttk.Frame(self.notebook)
        self.notebook.add(panel6, text="Doctor Feedback")
        feedback_container = ttk.Frame(panel6)
        feedback_container.pack(fill=tk.BOTH, expand=True)
        left_frame = ttk.Frame(feedback_container)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        self.canvas_feedback_original = tk.Canvas(left_frame, bg='white', relief=tk.SUNKEN, bd=1)
        self.canvas_feedback_original.pack(fill=tk.BOTH, expand=True)
        right_frame = ttk.Frame(feedback_container)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 0))
        self.fig_feedback_lime = plt.figure(figsize=(4, 4), facecolor='white')
        self.ax_feedback_lime = self.fig_feedback_lime.add_subplot(111)
        self.canvas_feedback_lime = FigureCanvasTkAgg(self.fig_feedback_lime, right_frame)
        self.canvas_feedback_lime.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        segment_frame = ttk.LabelFrame(panel6, text="Segment Risk Assessment", padding=10)
        segment_frame.pack(fill=tk.X, pady=(10, 0))
        self.segment_feedback_canvas = tk.Canvas(segment_frame, height=100)
        self.segment_feedback_frame = ttk.Frame(self.segment_feedback_canvas)
        self.segment_feedback_scrollbar = ttk.Scrollbar(segment_frame, orient=tk.VERTICAL, command=self.segment_feedback_canvas.yview)
        self.segment_feedback_canvas.configure(yscrollcommand=self.segment_feedback_scrollbar.set)
        self.segment_feedback_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.segment_feedback_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.segment_feedback_canvas.create_window((0, 0), window=self.segment_feedback_frame, anchor='nw')
        def on_segment_frame_configure(event):
            self.segment_feedback_canvas.configure(scrollregion=self.segment_feedback_canvas.bbox("all"))
        self.segment_feedback_frame.bind('<Configure>', on_segment_frame_configure)

    def setup_info_panel(self, parent):
        info_frame = ttk.LabelFrame(parent, text="Analysis Results", padding=5)
        info_frame.pack(fill=tk.X, pady=(5, 5))
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
        comments_frame = ttk.Frame(remarks_frame)
        comments_frame.pack(fill=tk.X, pady=(10, 0))
        ttk.Label(comments_frame, text="Comments:").pack(anchor=tk.W)
        self.comments_text = tk.Text(comments_frame, height=4)
        self.comments_text.pack(fill=tk.X)
        ttk.Button(comments_frame, text="Save Comments", command=self.save_comments).pack(anchor=tk.W, pady=(5, 0))

    def load_model(self):
        try:
            if not os.path.exists(CONFIG['MODEL_PATH']):
                self.status_var.set("Model file not found!")
                return
            self.model = EnhancedEfficientNetB4(num_classes=CONFIG['NUM_CLASSES'])
            checkpoint = torch.load(CONFIG['MODEL_PATH'], map_location=CONFIG['DEVICE'], weights_only=False)
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.model.load_state_dict(checkpoint)
            self.model = self.model.to(CONFIG['DEVICE'])
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
            self.draw_paths = []
            self.hotspots = []
            self.hotspot_counter = 0
            self.manual_segment = []
            self.segment_status.clear()
            self.draw_image = None
            self.current_data = None  # Clear previous analysis data
            self.display_original_image()
            self.display_feedback_original_image()
            self.update_feedback_panel()
            self.status_var.set(f"Image loaded: {os.path.basename(file_path)}")
            # Clear LIME heatmap in feedback panel
            self.ax_feedback_lime.clear()
            self.ax_feedback_lime.axis('off')
            self.canvas_feedback_lime.draw()
            logging.info("Cleared LIME heatmap in feedback panel on new image load")

    def display_original_image(self):
        if hasattr(self, 'current_image_path'):
            image = Image.open(self.current_image_path).convert('RGB')
            canvas_width = 300
            canvas_height = 300
            image.thumbnail((canvas_width, canvas_height), Image.LANCZOS)
            self.draw_image = image.copy()
            draw = ImageDraw.Draw(self.draw_image)
            img_width, img_height = image.size
            scale_x = img_width / canvas_width
            scale_y = img_height / canvas_height
            for hotspot in self.hotspots:
                x, y = hotspot['center']
                radius = hotspot['radius']
                label = hotspot['label']
                color = hotspot['color']
                x_img, y_img = x * scale_x, y * scale_y
                radius_img = radius * min(scale_x, scale_y)
                draw.ellipse(
                    [x_img - radius_img, y_img - radius_img, x_img + radius_img, y_img + radius_img],
                    outline=color, width=2
                )
                draw.text(
                    (x_img, y_img), label, fill='white',
                    font=None, anchor='mm'
                )
            for path in self.draw_paths:
                points = [(x * scale_x, y * scale_y) for x, y in path['points']]
                color = path['color']
                size = path['size'] * min(scale_x, scale_y)
                for i in range(len(points) - 1):
                    draw.line([points[i], points[i+1]], fill=color, width=int(size))
            photo = ImageTk.PhotoImage(self.draw_image)
            self.canvas1.delete("all")
            x = canvas_width // 2
            y = canvas_height // 2
            self.canvas1.create_image(x, y, image=photo)
            self.canvas1.photo = photo

    def display_feedback_original_image(self):
        if hasattr(self, 'current_image_path'):
            image = Image.open(self.current_image_path).convert('RGB')
            canvas_width = 300
            canvas_height = 300
            image.thumbnail((canvas_width, canvas_height), Image.LANCZOS)
            draw_image = image.copy()
            draw = ImageDraw.Draw(draw_image)
            img_width, img_height = image.size
            scale_x = img_width / canvas_width
            scale_y = img_height / canvas_height
            for hotspot in self.hotspots:
                x, y = hotspot['center']
                radius = hotspot['radius']
                label = hotspot['label']
                color = hotspot['color']
                x_img, y_img = x * scale_x, y * scale_y
                radius_img = radius * min(scale_x, scale_y)
                draw.ellipse(
                    [x_img - radius_img, y_img - radius_img, x_img + radius_img, y_img + radius_img],
                    outline=color, width=2
                )
                draw.text(
                    (x_img, y_img), label, fill='white',
                    font=None, anchor='mm'
                )
            for path in self.draw_paths:
                points = [(x * scale_x, y * scale_y) for x, y in path['points']]
                color = path['color']
                size = path['size'] * min(scale_x, scale_y)
                for i in range(len(points) - 1):
                    draw.line([points[i], points[i+1]], fill=color, width=int(size))
            photo = ImageTk.PhotoImage(draw_image)
            self.canvas_feedback_original.delete("all")
            x = canvas_width // 2
            y = canvas_height // 2
            self.canvas_feedback_original.create_image(x, y, image=photo)
            self.canvas_feedback_original.photo = photo

    def analyze_image(self):
        if not hasattr(self, 'current_image_path'):
            messagebox.showerror("Error", "Please load an image first!")
            return
        self.status_var.set("Analyzing image...")
        self.root.update()
        try:
            n_segments = self.segments_var.get()
            logging.info(f"Calling LIMEExplainer.explain_image with {n_segments} segments")
            self.current_data = self.explainer.explain_image(self.current_image_path, n_segments)
            logging.info(f"LIME explanation completed. Positive segments: {len(self.current_data['positive_segments'])}, "
                        f"Segments shape: {self.current_data['segments'].shape if self.current_data['segments'] is not None else 'None'}, "
                        f"Predicted class: {self.current_data['predicted_class']}, "
                        f"Probabilities: Non-Suspicious={self.current_data['non_suspicious_prob']:.3f}, "
                        f"Suspicious={self.current_data['suspicious_prob']:.3f}")
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.2, 0.2, 0.2])
            ])
            img = Image.open(self.current_image_path).convert('RGB')
            roi_image, roi_mask, _, is_mouth_open, _ = detect_intraoral_roi(img)
            input_tensor = transform(roi_image).unsqueeze(0).to(CONFIG['DEVICE'])
            grad_cam = GradCAM(self.model)
            logging.info("Calling GradCAM.generate")
            heatmap, gradcam_class, _, gradcam_confidence = grad_cam.generate(input_tensor, roi_mask)
            logging.info(f"Grad-CAM generation completed. Class: {gradcam_class}, Confidence: {gradcam_confidence:.3f}")
            self.current_data['gradcam_heatmap'] = heatmap
            self.current_data['gradcam_class'] = gradcam_class
            self.current_data['gradcam_confidence'] = gradcam_confidence
            self.current_data['hotspot_segments'] = self.compute_hotspot_segments()
            logging.info(f"Computed {len(self.current_data['hotspot_segments'])} hotspot segments")
            max_segments = min(10, len(self.current_data['hotspot_segments']))
            self.hotspot_segments_var.set(max_segments)
            self.hotspot_segments_label.config(text=str(max_segments))
            self.hotspot_segments_entry.delete(0, tk.END)
            self.hotspot_segments_entry.insert(0, str(max_segments))
            self.update_visualizations()
            self.update_info_panel()
            self.update_remarks_panel()
            self.update_feedback_panel()
            self.generate_combined_visualization()
            pred_class = self.current_data['predicted_class']
            confidence = self.current_data['confidence']
            class_name = ['Non-Suspicious', 'Suspicious'][pred_class]
            self.status_var.set(f"Analysis complete: {class_name} ({confidence:.3f})")
        except Exception as e:
            logging.error(f"Analysis failed: {str(e)}")
            messagebox.showerror("Error", f"Analysis failed: {str(e)}")
            self.status_var.set("Analysis failed!")

    def compute_hotspot_segments(self):
        if not self.current_data:
            return []
        heatmap = self.current_data['gradcam_heatmap'].copy()
        heatmap = cv2.resize(heatmap, (CONFIG['IMAGE_SIZE'], CONFIG['IMAGE_SIZE']))
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
        roi_mask = self.current_data['roi_mask']
        roi_mask_resized = cv2.resize(roi_mask, (CONFIG['IMAGE_SIZE'], CONFIG['IMAGE_SIZE']), interpolation=cv2.INTER_NEAREST)
        hotspot_mask = (heatmap >= CONFIG['HOTSPOT_THRESHOLD']) & (roi_mask_resized > 0)
        segments = self.current_data['segments']
        positive_segments = self.current_data['positive_segments']
        hotspot_segments = []
        for seg_id, importance in positive_segments:
            seg_mask = segments == seg_id
            seg_area = np.sum(seg_mask)
            if seg_area == 0:
                continue
            overlap = np.sum(seg_mask & hotspot_mask) / seg_area
            if overlap > 0.5:
                hotspot_segments.append((seg_id, importance, overlap))
        hotspot_segments.sort(key=lambda x: (x[1], x[2]), reverse=True)
        logging.info(f"Computed {len(hotspot_segments)} hotspot segments within ROI")
        return [(seg_id, imp) for seg_id, imp, _ in hotspot_segments]

    def update_hotspot_segments(self, value):
        try:
            num_segments = int(float(value))
            self.hotspot_segments_var.set(num_segments)
            self.hotspot_segments_label.config(text=str(num_segments))
            self.hotspot_segments_entry.delete(0, tk.END)
            self.hotspot_segments_entry.insert(0, str(num_segments))
            logging.info(f"Hotspot segments updated to: {num_segments}")
            if self.current_data:
                self.update_lime_gradcam_overlay()
                self.update_feedback_lime_heatmap()
        except ValueError:
            logging.error("Invalid hotspot segments value")
            self.hotspot_segments_entry.delete(0, tk.END)
            self.hotspot_segments_entry.insert(0, str(self.hotspot_segments_var.get()))

    def update_hotspot_segments_entry(self, event):
        try:
            value = int(self.hotspot_segments_entry.get())
            if 1 <= value <= 10:
                self.hotspot_segments_var.set(value)
                self.hotspot_segments_label.config(text=str(value))
                logging.info(f"Hotspot segments set via entry to: {value}")
                if self.current_data:
                    self.update_lime_gradcam_overlay()
                    self.update_feedback_lime_heatmap()
            else:
                messagebox.showerror("Error", "Hotspot segments must be between 1 and 10")
                self.hotspot_segments_entry.delete(0, tk.END)
                self.hotspot_segments_entry.insert(0, str(self.hotspot_segments_var.get()))
        except ValueError:
            messagebox.showerror("Error", "Please enter a valid number")
            self.hotspot_segments_entry.delete(0, tk.END)
            self.hotspot_segments_entry.insert(0, str(self.hotspot_segments_var.get()))

    def update_segments(self, value):
        try:
            segments = int(float(value))
            self.segments_var.set(segments)
            self.segments_entry.delete(0, tk.END)
            self.segments_entry.insert(0, str(segments))
            if hasattr(self, 'current_image_path'):
                self.analyze_image()
        except ValueError:
            logging.error("Invalid segments value")
            self.segments_entry.delete(0, tk.END)
            self.segments_entry.insert(0, str(self.segments_var.get()))

    def update_segments_entry(self, event):
        try:
            value = int(self.segments_entry.get())
            if 100 <= value <= 300:
                self.segments_var.set(value)
                if hasattr(self, 'current_image_path'):
                    self.analyze_image()
            else:
                messagebox.showerror("Error", "Segments must be between 100 and 300")
                self.segments_entry.delete(0, tk.END)
                self.segments_entry.insert(0, str(self.segments_var.get()))
        except ValueError:
            messagebox.showerror("Error", "Please enter a valid number")
            self.segments_entry.delete(0, tk.END)
            self.segments_entry.insert(0, str(self.segments_var.get()))

    def update_threshold(self, value):
        try:
            percent = int(float(value))
            self.threshold_var.set(percent)
            self.threshold_entry.delete(0, tk.END)
            self.threshold_entry.insert(0, str(percent))
            if self.current_data:
                self.update_visualizations()
                self.update_remarks_panel()
                self.update_feedback_panel()
        except ValueError:
            logging.error("Invalid threshold value")
            self.threshold_entry.delete(0, tk.END)
            self.threshold_entry.insert(0, str(self.threshold_var.get()))

    def update_threshold_entry(self, event):
        try:
            value = int(self.threshold_entry.get())
            if 10 <= value <= 100:
                self.threshold_var.set(value)
                if self.current_data:
                    self.update_visualizations()
                    self.update_remarks_panel()
                    self.update_feedback_panel()
            else:
                messagebox.showerror("Error", "Top Contributors must be between 10 and 100")
                self.threshold_entry.delete(0, tk.END)
                self.threshold_entry.insert(0, str(self.threshold_var.get()))
        except ValueError:
            messagebox.showerror("Error", "Please enter a valid number")
            self.threshold_entry.delete(0, tk.END)
            self.threshold_entry.insert(0, str(self.threshold_var.get()))

    def toggle_paint_mode(self):
        if self.paint_mode.get():
            self.canvas1.bind("<Button-1>", self.start_drawing)
            self.canvas1.bind("<B1-Motion>", self.draw)
            self.canvas1.bind("<ButtonRelease-1>", self.stop_drawing)
            self.canvas_feedback_original.bind("<Button-1>", self.start_drawing)
            self.canvas_feedback_original.bind("<B1-Motion>", self.draw)
            self.canvas_feedback_original.bind("<ButtonRelease-1>", self.stop_drawing)
            self.status_var.set("Paint mode enabled: Click to create hotspots or drag to draw")
        else:
            self.canvas1.unbind("<Button-1>")
            self.canvas1.unbind("<B1-Motion>")
            self.canvas1.unbind("<ButtonRelease-1>")
            self.canvas_feedback_original.unbind("<Button-1>")
            self.canvas_feedback_original.unbind("<B1-Motion>")
            self.canvas_feedback_original.unbind("<ButtonRelease-1>")
            self.status_var.set("Paint mode disabled")
        self.display_original_image()
        self.display_feedback_original_image()

    def choose_color(self):
        color_code = colorchooser.askcolor(title="Choose Brush Color")[1]
        if color_code:
            self.brush_color = color_code
            self.status_var.set(f"Brush color set to {self.brush_color}")

    def clear_painting(self):
        self.draw_paths = []
        self.hotspots = []
        self.hotspot_counter = 0
        self.manual_segment = []
        self.segment_status.clear()
        self.display_original_image()
        self.display_feedback_original_image()
        self.update_feedback_panel()
        self.status_var.set("Painting and hotspots cleared")

    def update_brush_size_entry(self, event):
        try:
            value = int(self.brush_size_entry.get())
            if 1 <= value <= 20:
                self.brush_size.set(value)
            else:
                messagebox.showerror("Error", "Brush size must be between 1 and 20")
                self.brush_size_entry.delete(0, tk.END)
                self.brush_size_entry.insert(0, str(self.brush_size.get()))
        except ValueError:
            messagebox.showerror("Error", "Please enter a valid number")
            self.brush_size_entry.delete(0, tk.END)
            self.brush_size_entry.insert(0, str(self.brush_size.get()))

    def start_drawing(self, event):
        if self.paint_mode.get():
            self.drawing = True
            self.last_x, self.last_y = event.x, event.y
            hotspot_radius = self.brush_size.get() * 2
            hotspot_label = f"m{self.hotspot_counter}"
            self.hotspots.append({
                'center': (self.last_x, self.last_y),
                'radius': hotspot_radius,
                'label': hotspot_label,
                'color': self.brush_color
            })
            self.manual_segment.append({
                'label': hotspot_label,
                'segment_status': '',
                'center': (self.last_x, self.last_y),
                'radius': hotspot_radius,
                'color': self.brush_color
            })
            self.hotspot_counter += 1
            self.draw_paths.append({'points': [(self.last_x, self.last_y)], 'color': self.brush_color, 'size': self.brush_size.get()})
            self.display_original_image()
            self.display_feedback_original_image()
            self.update_feedback_panel()
            self.status_var.set(f"Hotspot {hotspot_label} created at ({self.last_x}, {self.last_y})")

    def draw(self, event):
        if self.drawing and self.paint_mode.get():
            current_path = self.draw_paths[-1]
            current_path['points'].append((event.x, event.y))
            self.canvas1.create_line(
                self.last_x, self.last_y, event.x, event.y,
                fill=self.brush_color, width=self.brush_size.get(), capstyle=tk.ROUND, smooth=True
            )
            self.canvas_feedback_original.create_line(
                self.last_x, self.last_y, event.x, event.y,
                fill=self.brush_color, width=self.brush_size.get(), capstyle=tk.ROUND, smooth=True
            )
            self.last_x, self.last_y = event.x, event.y
            self.display_original_image()
            self.display_feedback_original_image()

    def stop_drawing(self, event):
        self.drawing = False
        self.last_x = None
        self.last_y = None

    def update_contribution_plot(self):
        self.ax_contrib.clear()
        positive_segments = self.current_data['positive_segments']
        num_to_show = max(1, int(len(positive_segments) * self.threshold_var.get() / 100))
        if not positive_segments:
            self.ax_contrib.text(0.5, 0.5, 'No positive contributors found',
                                ha='center', va='center', fontsize=10)
            self.canvas_contrib.draw()
            return
        segments_to_plot = positive_segments[:num_to_show]
        segment_labels = [f'S{i}' for i in range(len(segments_to_plot))]
        importances = [imp for _, imp in segments_to_plot]
        colors = ['red' if self.segment_status.get(f'S{i}', tk.StringVar(value='')).get() == 'Suspicious High Risk' else
                 'orange' if self.segment_status.get(f'S{i}', tk.StringVar(value='')).get() == 'Suspicious Low Risk' else
                 '#D3D3D3' for i in range(len(segments_to_plot))]
        bars = self.ax_contrib.bar(segment_labels, importances, color=colors)
        plt.ylabel('Contribution Score')
        plt.xlabel('Segments')
        pred_class = self.current_data['predicted_class']
        class_name = ['Non-Suspicious', 'Suspicious'][pred_class]
        confidence = self.current_data['confidence']
        title_text = f'Segment Contribution Analysis\nPrediction: {class_name} (Confidence: {confidence:.3f})\nTop {num_to_show}/{len(positive_segments)} Segments'
        self.ax_contrib.set_title(title_text, fontsize=10)
        self.ax_contrib.set_ylim(0, max(importances) * 1.1 if importances else 1.0)
        for bar in bars:
            height = bar.get_height()
            self.ax_contrib.text(
                bar.get_x() + bar.get_width() / 2, height,
                f'{height:.3f}', ha='center', va='bottom', fontsize=8
            )
        plt.tight_layout()
        self.canvas_contrib.draw()

    def update_feedback_panel(self):
        for widget in self.segment_feedback_frame.winfo_children():
            widget.destroy()
        
        if not self.current_data:
            logging.info("No current_data in update_feedback_panel, clearing feedback LIME heatmap")
            self.ax_feedback_lime.clear()
            self.ax_feedback_lime.axis('off')
            self.canvas_feedback_lime.draw()
            return

        logging.info("Updating feedback panel with new analysis data")
        positive_segments = self.current_data['positive_segments']
        num_to_show = max(1, int(len(positive_segments) * self.threshold_var.get() / 100))
        segments_to_show = positive_segments[:num_to_show]
        segments = self.current_data['segments']
        canvas_width, canvas_height = 300, 300
        img_width, img_height = self.current_data['img_array'].shape[1], self.current_data['img_array'].shape[0]
        scale_x = img_width / canvas_width
        scale_y = img_height / canvas_height

        hotspot_segment_map = {}
        for hotspot in self.hotspots:
            label = hotspot['label']
            x, y = hotspot['center']
            x_img, y_img = int(x * scale_x), int(y * scale_y)
            x_img = min(max(x_img, 0), img_width - 1)
            y_img = min(max(y_img, 0), img_height - 1)
            seg_id = segments[y_img, x_img] if segments is not None else -1
            hotspot_segment_map[label] = seg_id

        current_hotspot_statuses = {k: v.get() for k, v in self.segment_status.items() if k.startswith('m')}
        self.segment_status.clear()

        for idx, (seg_id, imp) in enumerate(segments_to_show):
            frame = ttk.Frame(self.segment_feedback_frame)
            frame.pack(fill=tk.X, pady=2)
            var = tk.StringVar(value='')
            self.segment_status[f'S{idx}'] = var
            ttk.Label(frame, text=f'S{idx} (Imp: {imp:.3f})').pack(side=tk.LEFT, padx=(0, 5))
            ttk.Radiobutton(frame, text="Suspicious, High Risk", value='Suspicious High Risk', variable=var,
                            command=lambda s=f'S{idx}': self.update_segment_status(s)).pack(side=tk.LEFT, padx=5)
            ttk.Radiobutton(frame, text="Suspicious, Low Risk", value='Suspicious Low Risk', variable=var,
                            command=lambda s=f'S{idx}': self.update_segment_status(s)).pack(side=tk.LEFT, padx=5)

        for hotspot in self.hotspots:
            label = hotspot['label']
            seg_id = hotspot_segment_map.get(label, -1)
            frame = ttk.Frame(self.segment_feedback_frame)
            frame.pack(fill=tk.X, pady=2)
            var = tk.StringVar(value=current_hotspot_statuses.get(label, ''))
            self.segment_status[label] = var
            seg_idx = next((i for i, (s_id, _) in enumerate(positive_segments) if s_id == seg_id), -1)
            display_text = f'{label} (Seg S{seg_idx}, Imp: {positive_segments[seg_idx][1]:.3f})' if seg_idx >= 0 else f'{label} (No Segment)'
            ttk.Label(frame, text=display_text).pack(side=tk.LEFT, padx=(0, 5))
            ttk.Radiobutton(frame, text="Suspicious, High Risk", value='Suspicious High Risk', variable=var,
                            command=lambda s=label: self.update_segment_status(s)).pack(side=tk.LEFT, padx=5)
            ttk.Radiobutton(frame, text="Suspicious Low Risk", value='Suspicious Low Risk', variable=var,
                            command=lambda s=label: self.update_segment_status(s)).pack(side=tk.LEFT, padx=5)
            for ms in self.manual_segment:
                if ms['label'] == label:
                    ms['segment_status'] = var.get()

        select_all_frame = ttk.Frame(self.segment_feedback_frame)
        select_all_frame.pack(fill=tk.X, pady=5)
        ttk.Button(select_all_frame, text="Select All High Risk", command=lambda: self.select_all_segments('Suspicious High Risk')).pack(side=tk.LEFT, padx=5)
        ttk.Button(select_all_frame, text="Select All Low Risk", command=lambda: self.select_all_segments('Suspicious Low Risk')).pack(side=tk.LEFT, padx=5)
        ttk.Button(select_all_frame, text="Deselect All", command=self.deselect_all_segments).pack(side=tk.LEFT, padx=5)

        logging.info(f"Calling update_feedback_lime_heatmap from update_feedback_panel")
        self.update_feedback_lime_heatmap()
        self.root.update_idletasks()  # Force Tkinter to process pending events

    def select_all_segments(self, risk_level):
        for var in self.segment_status.values():
            var.set(risk_level)
        for ms in self.manual_segment:
            if ms['label'] in self.segment_status:
                ms['segment_status'] = risk_level
        self.update_visualizations()
        self.update_feedback_lime_heatmap()

    def deselect_all_segments(self):
        for var in self.segment_status.values():
            var.set('')
        for ms in self.manual_segment:
            ms['segment_status'] = ''
        self.update_visualizations()
        self.update_feedback_lime_heatmap()

    def update_segment_status(self, segment):
        if self.current_data and segment in self.segment_status:
            for ms in self.manual_segment:
                if ms['label'] == segment:
                    ms['segment_status'] = self.segment_status[segment].get()
            self.update_visualizations()
            self.update_feedback_lime_heatmap()

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
                heatmap_color = plt.cm.hot(heatmap)[:,:,:3]
                blended = 0.7 * display_image + 0.3 * heatmap_color
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
        self.ax_lime.set_title(title_text, fontsize=10)
        self.ax_lime.axis('off')
        self.canvas_lime.draw()

    def update_feedback_lime_heatmap(self):
        logging.info("Starting update_feedback_lime_heatmap")
        self.ax_feedback_lime.clear()
        if not self.current_data:
            logging.info("No current_data in update_feedback_lime_heatmap, clearing canvas")
            self.ax_feedback_lime.axis('off')
            self.canvas_feedback_lime.draw()
            return

        try:
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
                                status = self.segment_status.get(f'S{i}', tk.StringVar(value='')).get()
                                color = 'red' if status == 'Suspicious High Risk' else 'orange' if status == 'Suspicious Low Risk' else 'black'
                                self.ax_feedback_lime.text(centroid_x, centroid_y, f'S{i}',
                                                          fontsize=8, fontweight='bold', color='white',
                                                          ha='center', va='center',
                                                          bbox=dict(facecolor=color, alpha=0.9))
                    heatmap_color = plt.cm.hot(heatmap)[:,:,:3]
                    blended = 0.7 * display_image + 0.3 * heatmap_color
                    if self.show_boundaries_var.get():
                        boundary_image = np.zeros_like(blended)
                        for seg_id, _ in segments_to_show:
                            mask = segments == seg_id
                            edges = ndimage.binary_erosion(mask) ^ mask
                            boundary_image[edges] = [1, 1, 1]
                        boundary_mask = np.any(boundary_image > 0, axis=2)
                        blended[boundary_mask] = boundary_image[boundary_mask]
                    display_image = blended
            self.ax_feedback_lime.imshow(display_image)
            pred_class = self.current_data['predicted_class']
            class_name = ['Non-Suspicious', 'Suspicious'][pred_class]
            confidence = self.current_data['confidence']
            cancer_type = self.current_data['cancer_type']
            risk_level = self.current_data['risk_level']
            title_text = f'LIME Heatmap: {class_name} ({confidence:.3f})\n{cancer_type} | {risk_level}\n({num_to_show}/{num_segments} segments)'
            self.ax_feedback_lime.set_title(title_text, fontsize=10)
            self.ax_feedback_lime.axis('off')
            self.canvas_feedback_lime.draw()
            self.canvas_feedback_lime.get_tk_widget().update()  # Force canvas refresh
            logging.info("Successfully updated feedback LIME heatmap")
        except Exception as e:
            logging.error(f"Failed to update feedback LIME heatmap: {str(e)}")
            self.ax_feedback_lime.clear()
            self.ax_feedback_lime.text(0.5, 0.5, 'Error updating heatmap',
                                      ha='center', va='center', fontsize=10)
            self.ax_feedback_lime.axis('off')
            self.canvas_feedback_lime.draw()

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
            f'Mouth Status: {"Open" if self.current_data["is_mouth_open"] else "Closed"}'
        )
        self.ax_gradcam.set_title(title_text, fontsize=10)
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
        hotspot_segments = self.current_data.get('hotspot_segments', [])
        num_to_show = min(self.hotspot_segments_var.get(), len(hotspot_segments))
        segments_to_show = hotspot_segments[:num_to_show]
        roi_mask = cv2.resize(self.current_data['roi_mask'], (CONFIG['IMAGE_SIZE'], CONFIG['IMAGE_SIZE']), interpolation=cv2.INTER_NEAREST)
        if segments_to_show:
            if self.show_boundaries_var.get():
                boundary_image = np.zeros_like(display_image)
                for seg_id, _ in segments_to_show:
                    mask = (segments == seg_id) & (roi_mask > 0)
                    edges = ndimage.binary_erosion(mask) ^ mask
                    boundary_image[edges] = [1, 1, 0]  # Yellow boundaries
                boundary_mask = np.any(boundary_image > 0, axis=2)
                display_image[boundary_mask] = boundary_image[boundary_mask]
            if self.show_numbers_var.get():
                for i, (seg_id, importance) in enumerate(segments_to_show):
                    mask = segments == seg_id
                    y_coords, x_coords = np.where(mask)
                    if len(y_coords) > 0:
                        centroid_y, centroid_x = int(np.mean(y_coords)), int(np.mean(x_coords))
                        status = self.segment_status.get(f'S{i}', tk.StringVar(value='')).get()
                        color = 'red' if status == 'Suspicious High Risk' else 'orange' if status == 'Suspicious Low Risk' else 'black'
                        self.ax_lime_gradcam.text(
                            centroid_x, centroid_y, f'S{i}',
                            fontsize=8, fontweight='bold', color='white',
                            ha='center', va='center',
                            bbox=dict(facecolor=color, alpha=0.9)
                        )
        self.ax_lime_gradcam.imshow(display_image)
        pred_class = self.current_data['predicted_class']
        class_name = ['Non-Suspicious', 'Suspicious'][pred_class]
        confidence = self.current_data['confidence']
        cancer_type = self.current_data['cancer_type']
        title_text = (
            f'LIME on Grad-CAM Overlay\nPrediction: {class_name} ({confidence:.3f})\n'
            f'{cancer_type}\nHotspot Segments: {num_to_show}/{len(hotspot_segments)}'
        )
        self.ax_lime_gradcam.set_title(title_text, fontsize=10)
        self.ax_lime_gradcam.axis('off')
        self.canvas_lime_gradcam.draw()

    def update_info_panel(self):
        if not self.current_data:
            return
        pred_class = self.current_data['predicted_class']
        class_name = ['Non-Suspicious', 'Suspicious'][pred_class]
        self.prediction_var.set(f"Class: {class_name}")
        self.confidence_var.set(f"Confidence: {self.current_data['confidence']:.3f}")
        self.probabilities_var.set(f"Probabilities: Non-Suspicious={self.current_data['non_suspicious_prob']:.3f}, Suspicious={self.current_data['suspicious_prob']:.3f}")
        self.cancer_type_var.set(f"Cancer Type: {self.current_data['cancer_type']}")
        self.risk_level_var.set(f"Risk Level: {self.current_data['risk_level']}")
        num_segments = len(self.current_data['positive_segments'])
        num_to_show = max(1, int(num_segments * self.threshold_var.get() / 100))
        self.segments_info_var.set(f"Segments: {num_to_show}/{num_segments} displayed")

    def update_remarks_panel(self):
        self.comments_text.delete("1.0", tk.END)
        self.comments_text.insert("1.0", self.doctor_comments)
        if self.current_data:
            pred_class = self.current_data['predicted_class']
            class_name = ['Non-Suspicious', 'Suspicious'][pred_class]
            cancer_type = self.current_data['cancer_type']
            risk_level = self.current_data['risk_level']
            confidence = self.current_data['confidence']
            comment = (
                f"Prediction: {class_name} (Confidence: {confidence:.3f})\n"
                f"Cancer Type: {cancer_type}\n"
                f"Risk Level: {risk_level}\n"
                f"Hotspot Segments: {len(self.current_data.get('hotspot_segments', []))}\n"
            )
            self.comments_text.insert(tk.END, comment)

    def generate_combined_visualization(self):
        if not self.current_data:
            return
        combined_dir = os.path.join(CONFIG['RESULTS_DIR'], 'combined_results')
        os.makedirs(combined_dir, exist_ok=True)
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        img_array = self.current_data['img_array']
        axes[0].imshow(img_array)
        axes[0].set_title('Original Image')
        axes[0].axis('off')
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
                            axes[1].text(centroid_x, centroid_y, f'S{i}',
                                        fontsize=8, fontweight='bold', color='white',
                                        ha='center', va='center',
                                        bbox=dict(facecolor='black', alpha=0.9))
                heatmap_color = plt.cm.hot(heatmap)[:,:,:3]
                blended = 0.7 * display_image + 0.3 * heatmap_color
                if self.show_boundaries_var.get():
                    boundary_image = np.zeros_like(blended)
                    for seg_id, _ in segments_to_show:
                        mask = segments == seg_id
                        edges = ndimage.binary_erosion(mask) ^ mask
                        boundary_image[edges] = [1, 1, 1]
                    boundary_mask = np.any(boundary_image > 0, axis=2)
                    blended[boundary_mask] = boundary_image[boundary_mask]
                display_image = blended
        axes[1].imshow(display_image)
        axes[1].set_title('LIME Heatmap')
        axes[1].axis('off')
        heatmap = self.current_data['gradcam_heatmap']
        heatmap = cv2.resize(heatmap, (CONFIG['IMAGE_SIZE'], CONFIG['IMAGE_SIZE']))
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        roi_array = np.array(self.current_data['roi_image'])
        overlay = roi_array * 0.6 + heatmap * 0.4
        axes[2].imshow(overlay.astype(np.uint8))
        axes[2].set_title('Grad-CAM Heatmap')
        axes[2].axis('off')
        plt.tight_layout()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(combined_dir, f'combined_{self.image_name}_{timestamp}.png')
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
        plt.close(fig)
        logging.info(f"Combined visualization saved to {output_path}")

    def save_results(self):
        if not self.current_data or not hasattr(self, 'current_image_path'):
            messagebox.showerror("Error", "No analysis results to save!")
            return
        results_dir = os.path.join(CONFIG['RESULTS_DIR'], 'lime_gradcam_results')
        os.makedirs(results_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result = {
            'image_path': self.current_image_path,
            'prediction': 'Suspicious' if self.current_data['predicted_class'] == 1 else 'Non-Suspicious',
            'confidence': self.current_data['confidence'],
            'non_suspicious_prob': self.current_data['non_suspicious_prob'],
            'suspicious_prob': self.current_data['suspicious_prob'],
            'cancer_type': self.current_data['cancer_type'],
            'risk_level': self.current_data['risk_level'],
            'num_segments': len(self.current_data['positive_segments']),
            'hotspot_segments': len(self.current_data.get('hotspot_segments', [])),
            'doctor_comments': self.doctor_comments,
            'segment_status': {k: v.get() for k, v in self.segment_status.items()},
            'manual_segments': self.manual_segment,
            'timestamp': timestamp
        }
        json_path = os.path.join(results_dir, f'results_{self.image_name}_{timestamp}.json')
        with open(json_path, 'w') as f:
            json.dump(result, f, indent=4)
        img_array = self.current_data['img_array']
        lime_heatmap = self.fig_lime.canvas.buffer_rgba()
        lime_img = Image.frombytes('RGBA', self.fig_lime.canvas.get_width_height(), lime_heatmap)
        lime_path = os.path.join(results_dir, f'lime_{self.image_name}_{timestamp}.png')
        lime_img.save(lime_path)
        gradcam_heatmap = self.fig_gradcam.canvas.buffer_rgba()
        gradcam_img = Image.frombytes('RGBA', self.fig_gradcam.canvas.get_width_height(), gradcam_heatmap)
        gradcam_path = os.path.join(results_dir, f'gradcam_{self.image_name}_{timestamp}.png')
        gradcam_img.save(gradcam_path)
        lime_gradcam_heatmap = self.fig_lime_gradcam.canvas.buffer_rgba()
        lime_gradcam_img = Image.frombytes('RGBA', self.fig_lime_gradcam.canvas.get_width_height(), lime_gradcam_heatmap)
        lime_gradcam_path = os.path.join(results_dir, f'lime_gradcam_{self.image_name}_{timestamp}.png')
        lime_gradcam_img.save(lime_gradcam_path)
        messagebox.showinfo("Success", f"Results saved to {results_dir}")
        logging.info(f"Results saved: JSON to {json_path}, Images to {results_dir}")

def main():
    logging.basicConfig(
        filename=os.path.join(CONFIG['RESULTS_DIR'], 'debug.log'),
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    root = tk.Tk()
    app = EnhancedLIMEGradCAMVisualization(root)
    root.mainloop()

if __name__ == "__main__":
    main()
