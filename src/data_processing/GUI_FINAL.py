import tkinter as tk
import torch
from tkinter import ttk, filedialog, messagebox
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
        self.zoom_level = 1.0
        self.polygons = []
        self.drawing_polygon = False
        self.dragging = False
        self.current_line_start = None
        self.current_polygon_idx = None
        self.cancer_types = [
            "Smokeless tobacco keratosis",
            "Homogeneous oral leukoplakia",
            "Oral lichen planus",
            "Speckled oral leukoplakia",
            "Erythoplakia",
            "Verrucous oral leukoplakia",
            "Proliferative verrucous leukoplakia",
            "Squamous cell carcinoma"
        ]
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
        self.fig_lime_gradcam = plt.figure(figsize=(4, 4), facecolor='white')
        self.ax_lime_gradcam = self.fig_lime_gradcam.add_subplot(111)
        self.canvas_lime_gradcam = FigureCanvasTkAgg(self.fig_lime_gradcam, panel4)
        self.canvas_lime_gradcam.get_tk_widget().pack(fill=tk.BOTH, expand=True)
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
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.canvas_feedback_original = tk.Canvas(left_frame, bg='white', relief=tk.SUNKEN, bd=1)
        self.canvas_feedback_original.pack(fill=tk.BOTH, expand=True)
        zoom_frame = ttk.Frame(left_frame)
        zoom_frame.pack(fill=tk.X, pady=(5, 0))
        ttk.Button(zoom_frame, text="Zoom In", command=lambda: self.zoom_image(1.2)).pack(side=tk.LEFT, padx=5)
        ttk.Button(zoom_frame, text="Zoom Out", command=lambda: self.zoom_image(0.8)).pack(side=tk.LEFT, padx=5)
        ttk.Button(zoom_frame, text="Reset Zoom", command=lambda: self.zoom_image(1.0, reset=True)).pack(side=tk.LEFT, padx=5)
        ttk.Button(zoom_frame, text="Start Polygon", command=self.start_polygon).pack(side=tk.LEFT, padx=5)
        ttk.Button(zoom_frame, text="Clear Polygon", command=self.clear_polygon).pack(side=tk.LEFT, padx=5)
        self.canvas_feedback_original.bind("<Button-1>", self.add_polygon_point)
        self.canvas_feedback_original.bind("<B1-Motion>", self.drag)
        self.canvas_feedback_original.bind("<ButtonRelease-1>", self.stop_drag)
        right_frame = ttk.Frame(feedback_container, width=200)
        right_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=5)
        ttk.Label(right_frame, text="Cancer Types").pack(anchor=tk.NW)
        self.cancer_listbox = tk.Listbox(right_frame, selectmode=tk.SINGLE, exportselection=0, height=len(self.cancer_types))
        for cancer_type in self.cancer_types:
            self.cancer_listbox.insert(tk.END, cancer_type)
        self.cancer_listbox.pack(fill=tk.Y, expand=True)
        self.cancer_listbox.bind('<<ListboxSelect>>', self.label_polygon)
        self.cancer_listbox.config(state='disabled')

    def setup_info_panel(self, parent):
        info_frame = ttk.LabelFrame(parent, text="Analysis Results", padding=5)
        info_frame.pack(fill=tk.X, pady=(5, 5))
        info_container = ttk.Frame(info_frame)
        info_container.pack(fill=tk.X, pady=5)
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
            self.current_data = None
            self.zoom_level = 1.0
            self.polygons = []   
            self.drawing_polygon = False
            self.dragging = False
            self.current_line_start = None
            self.current_polygon_idx = None
            self.cancer_listbox.config(state='disabled')
            self.display_original_image()
            self.display_feedback_original_image()
            self.update_feedback_panel()
            self.status_var.set(f"Image loaded: {os.path.basename(file_path)}")
            logging.info("Image loaded and polygon data cleared")

    def display_original_image(self):
        if hasattr(self, 'current_image_path'):
            image = Image.open(self.current_image_path).convert('RGB')
            canvas_width = self.canvas1.winfo_width()
            canvas_height = self.canvas1.winfo_height()
            if canvas_width <= 1 or canvas_height <= 1:  # Canvas not yet rendered
                canvas_width = 300
                canvas_height = 300
            img_width, img_height = image.size
            scale = min(canvas_width / img_width, canvas_height / img_height) * self.zoom_level
            new_width = int(img_width * scale)
            new_height = int(img_height * scale)
            image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            draw = ImageDraw.Draw(image)
            image_x = 0  # Top-left corner
            image_y = 0  # Top-left corner
            for idx, polygon in enumerate(self.polygons):
                scaled_points = [(x * scale, y * scale) for x, y in polygon['points']]  # No offset for centering
                for start_idx, end_idx in polygon['edges']:
                    if 0 <= start_idx < len(scaled_points) and 0 <= end_idx < len(scaled_points):
                        draw.line([scaled_points[start_idx], scaled_points[end_idx]], fill='red', width=2)
                for x, y in scaled_points:
                    draw.ellipse([x - 5, y - 5, x + 5, y + 5], fill='red')
                if polygon['label']:
                    points = np.array(polygon['points'])
                    centroid_x = np.mean(points[:, 0]) * scale
                    centroid_y = np.mean(points[:, 1]) * scale
                    draw.text((centroid_x, centroid_y), f"Polygon {idx + 1}: {polygon['label']}", fill='blue')
            photo = ImageTk.PhotoImage(image)
            self.canvas1.delete("all")
            self.canvas1.create_image(image_x, image_y, image=photo, anchor='nw')  # Anchor at top-left
            self.canvas1.photo = photo
            self.canvas1.image_width = new_width
            self.canvas1.image_height = new_height
            self.canvas1.scale = scale
            self.canvas1.image_x = image_x
            self.canvas1.image_y = image_y
            self.canvas1.original_width = img_width
            self.canvas1.original_height = img_height
            logging.info(f"Original image displayed: canvas_size=({canvas_width}, {canvas_height}), image_size=({new_width}, {new_height}), original_size=({img_width}, {img_height}), image_pos=({image_x:.2f}, {image_y:.2f}), scale={scale:.2f}")

    def display_feedback_original_image(self):
        if hasattr(self, 'current_image_path'):
            image = Image.open(self.current_image_path).convert('RGB')
            canvas_width = self.canvas_feedback_original.winfo_width()
            canvas_height = self.canvas_feedback_original.winfo_height()
            if canvas_width <= 1 or canvas_height <= 1:  
                canvas_width = 300
                canvas_height = 300
            img_width, img_height = image.size
            scale = min(canvas_width / img_width, canvas_height / img_height) * self.zoom_level
            new_width = int(img_width * scale)
            new_height = int(img_height * scale)
            image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            draw = ImageDraw.Draw(image)
            image_x = 0  
            image_y = 0  
            for idx, polygon in enumerate(self.polygons):
                scaled_points = [(x * scale, y * scale) for x, y in polygon['points']]  # No offset for centering
                for start_idx, end_idx in polygon['edges']:
                    if 0 <= start_idx < len(scaled_points) and 0 <= end_idx < len(scaled_points):
                        draw.line([scaled_points[start_idx], scaled_points[end_idx]], fill='red', width=2)
                for x, y in scaled_points:
                    draw.ellipse([x - 5, y - 5, x + 5, y + 5], fill='red')
                if polygon['label']:
                    points = np.array(polygon['points'])
                    centroid_x = np.mean(points[:, 0]) * scale
                    centroid_y = np.mean(points[:, 1]) * scale
                    draw.text((centroid_x, centroid_y), f"Polygon {idx + 1}: {polygon['label']}", fill='blue')
            photo = ImageTk.PhotoImage(image)
            self.canvas_feedback_original.delete("all")
            self.canvas_feedback_original.create_image(image_x, image_y, image=photo, anchor='nw')  # Anchor at top-left
            self.canvas_feedback_original.photo = photo
            self.canvas_feedback_original.image_width = new_width
            self.canvas_feedback_original.image_height = new_height
            self.canvas_feedback_original.scale = scale
            self.canvas_feedback_original.image_x = image_x
            self.canvas_feedback_original.image_y = image_y
            self.canvas_feedback_original.original_width = img_width
            self.canvas_feedback_original.original_height = img_height
            logging.info(f"Feedback image displayed: canvas_size=({canvas_width}, {canvas_height}), image_size=({new_width}, {new_height}), original_size=({img_width}, {img_height}), image_pos=({image_x:.2f}, {image_y:.2f}), scale={scale:.2f}")

    def zoom_image(self, factor, reset=False):
        if reset:
            self.zoom_level = 1.0
        else:
            self.zoom_level *= factor
            self.zoom_level = max(0.5, min(self.zoom_level, 3.0))
        self.display_original_image()
        self.display_feedback_original_image()
        self.status_var.set(f"Zoom level: {self.zoom_level:.2f}x")
        logging.info(f"Zoom level set to {self.zoom_level:.2f}x")

    def start_polygon(self):
        if self.drawing_polygon:
            self.status_var.set("Polygon drawing already in progress")
            return
        self.polygons.append({'points': [], 'edges': [], 'label': None})
        self.current_polygon_idx = len(self.polygons) - 1
        self.drawing_polygon = True
        self.dragging = False
        self.current_line_start = None
        self.cancer_listbox.config(state='disabled')
        self.display_original_image()
        self.display_feedback_original_image()
        self.status_var.set("Click to add polygon points or select existing point to connect")
        logging.info("Started new polygon drawing")

    def add_polygon_point(self, event):
        if not self.drawing_polygon or not hasattr(self, 'current_image_path'):
            return
        canvas_width = self.canvas_feedback_original.winfo_width()
        canvas_height = self.canvas_feedback_original.winfo_height()
        if canvas_width <= 1 or canvas_height <= 1:
            canvas_width = 300
            canvas_height = 300
        image = Image.open(self.current_image_path).convert('RGB')
        img_width, img_height = image.size
        scale = min(canvas_width / img_width, canvas_height / img_height) * self.zoom_level
        image_x = 0  # Top-left corner
        image_y = 0  # Top-left corner
        x = (event.x - image_x) / scale
        y = (event.y - image_y) / scale
        x = max(0, min(x, img_width))
        y = max(0, min(y, img_height))
        current_polygon = self.polygons[self.current_polygon_idx]
        if current_polygon['points']:
            distances = [((x - px) ** 2 + (y - py) ** 2) ** 0.5 for px, py in current_polygon['points']]
            min_distance = min(distances)
            closest_idx = distances.index(min_distance)
            if min_distance * scale < 10:
                self.current_line_start = closest_idx
                self.dragging = True
                self.status_var.set(f"Selected point {closest_idx} at ({current_polygon['points'][closest_idx][0]:.1f}, {current_polygon['points'][closest_idx][1]:.1f})")
                logging.info(f"Selected existing point {closest_idx} at ({current_polygon['points'][closest_idx][0]:.1f}, {current_polygon['points'][closest_idx][1]:.1f})")
                return
        current_polygon['points'].append((x, y))
        self.current_line_start = len(current_polygon['points']) - 1
        self.dragging = True
        self.display_original_image()
        self.display_feedback_original_image()
        self.status_var.set(f"Added point at ({x:.1f}, {y:.1f})")
        logging.info(f"Added point: ({x:.1f}, {y:.1f}) to polygon {self.current_polygon_idx}")

    def drag(self, event):
        if not self.drawing_polygon or not self.dragging or not hasattr(self, 'current_image_path'):
            return
        canvas_width = self.canvas_feedback_original.winfo_width()
        canvas_height = self.canvas_feedback_original.winfo_height()
        if canvas_width <= 1 or canvas_height <= 1:
            canvas_width = 300
            canvas_height = 300
        image = Image.open(self.current_image_path).convert('RGB')
        img_width, img_height = image.size
        scale = min(canvas_width / img_width, canvas_height / img_height) * self.zoom_level
        image_x = 0  # Top-left corner
        image_y = 0  # Top-left corner
        x = (event.x - image_x) / scale
        y = (event.y - image_y) / scale
        x = max(0, min(x, img_width))
        y = max(0, min(y, img_height))
        current_polygon = self.polygons[self.current_polygon_idx]
        if len(current_polygon['points']) > 1:
            distances = [(i, ((x - px) ** 2 + (y - py) ** 2) ** 0.5) for i, (px, py) in enumerate(current_polygon['points']) if i != self.current_line_start]
            if distances:
                closest_idx, min_distance = min(distances, key=lambda x: x[1])
                if min_distance * scale < 10:
                    if closest_idx != self.current_line_start and (self.current_line_start, closest_idx) not in current_polygon['edges'] and (closest_idx, self.current_line_start) not in current_polygon['edges']:
                        current_polygon['edges'].append((self.current_line_start, closest_idx))
                        self.current_line_start = closest_idx
                        self.dragging = True
                        if closest_idx == 0 and len(current_polygon['points']) >= 3:
                            self.drawing_polygon = False
                            self.cancer_listbox.config(state='normal')
                            self.cancer_listbox.focus_set()
                            self.status_var.set("Polygon closed. Select a cancer type from the list.")
                            logging.info(f"Polygon {self.current_polygon_idx} closed, listbox enabled")
                        else:
                            self.status_var.set(f"Connected to point {closest_idx} at ({current_polygon['points'][closest_idx][0]:.1f}, {current_polygon['points'][closest_idx][1]:.1f})")
                            logging.info(f"Connected point {self.current_line_start} to point {closest_idx} in polygon {self.current_polygon_idx}")
                    else:
                        self.status_var.set("Already connected or same point")
                else:
                    self.status_var.set("Drag closer to a point to connect")
            else:
                self.status_var.set("No other points to connect to")
        else:
            self.status_var.set("Need at least two points to connect")
        self.display_original_image()
        self.display_feedback_original_image()

    def stop_drag(self, event):
        if self.dragging:
            self.dragging = False
            self.current_line_start = None
            self.display_original_image()
            self.display_feedback_original_image()
            self.status_var.set("Drag stopped, click to add another point or select to connect")
            logging.info("Drag stopped")

    def label_polygon(self, event):
        logging.info("label_polygon triggered")
        if not self.polygons or self.drawing_polygon:
            logging.info("No polygons or drawing in progress, exiting label_polygon")
            return
        state = self.cancer_listbox.cget("state")
        logging.info(f"Listbox state: {state}")
        if state == 'disabled':
            logging.info("Listbox is disabled, cannot select")
            self.status_var.set("Error: Cancer type listbox is disabled")
            return
        selection = self.cancer_listbox.curselection()
        logging.info(f"Selection: {selection}")
        if not selection:
            logging.info("No selection made")
            self.status_var.set("Please select a cancer type")
            return
        cancer_type = self.cancer_listbox.get(selection[0])
        logging.info(f"Selected cancer type: {cancer_type}")
        self.polygons[self.current_polygon_idx]['label'] = cancer_type
        self.cancer_listbox.config(state='disabled')
        self.display_original_image()
        self.display_feedback_original_image()
        self.status_var.set(f"Polygon {self.current_polygon_idx + 1} labeled as {cancer_type}")
        logging.info(f"Polygon {self.current_polygon_idx} labeled as {cancer_type}")
    
    def clear_polygon(self):
        self.polygons = []
        self.drawing_polygon = False
        self.dragging = False
        self.current_line_start = None
        self.current_polygon_idx = None
        self.cancer_listbox.config(state='disabled')
        self.display_original_image()
        self.display_feedback_original_image()
        self.status_var.set("Polygons cleared")
        logging.info("Polygons cleared")

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
            self.update_visualizations()
            self.update_info_panel()
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
                self.update_info_panel()
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
                    self.update_info_panel()
                    self.update_feedback_panel()
            else:
                messagebox.showerror("Error", "Top Contributors must be between 10 and 100")
                self.threshold_entry.delete(0, tk.END)
                self.threshold_entry.insert(0, str(self.threshold_var.get()))
        except ValueError:
            messagebox.showerror("Error", "Please enter a valid number")
            self.threshold_entry.delete(0, tk.END)
            self.threshold_entry.insert(0, str(self.threshold_var.get()))

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
        gradcam_confidence = self.current_data['gradcam_confidence']
        title_text = (
            f'Grad-CAM Heatmap\nPrediction: {gradcam_class_name} '
            f'(Confidence: {gradcam_confidence:.3f})\n'
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
        positive_segments = self.current_data['positive_segments']
        num_to_show = max(1, int(len(positive_segments) * self.threshold_var.get() / 100))
        segments_to_show = positive_segments[:num_to_show]
        roi_mask = cv2.resize(self.current_data['roi_mask'], (CONFIG['IMAGE_SIZE'], CONFIG['IMAGE_SIZE']), interpolation=cv2.INTER_NEAREST)
        if segments_to_show:
            if self.show_boundaries_var.get():
                boundary_image = np.zeros_like(display_image)
                for seg_id, _ in segments_to_show:
                    mask = (segments == seg_id) & (roi_mask > 0)
                    edges = ndimage.binary_erosion(mask) ^ mask
                    boundary_image[edges] = [1, 1, 0]
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
        cancer_type = self.current_data['cancer_type']
        title_text = (
            f'LIME on Grad-CAM Overlay\nPrediction: {class_name} ({confidence:.3f})\n'
            f'{cancer_type}\nSegments: {num_to_show}/{len(positive_segments)}'
        )
        self.ax_lime_gradcam.set_title(title_text, fontsize=10)
        self.ax_lime_gradcam.axis('off')
        self.canvas_lime_gradcam.draw()

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
        bars = self.ax_contrib.bar(segment_labels, importances, color='#D3D3D3')
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
        self.display_feedback_original_image()
        self.root.update_idletasks()
        logging.info("Updated feedback panel (original image only)")

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
        if not hasattr(self, 'current_image_path') or not self.current_data:
            messagebox.showerror("Error", "No analysis results to save!")
            self.status_var.set("No results to save")
            return
        try:
            output_dir = CONFIG['RESULTS_DIR']
            os.makedirs(output_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_name = f"{self.image_name}_{timestamp}"
            
            original_img = Image.open(self.current_image_path).convert('RGB')
            original_path = os.path.join(output_dir, f"{base_name}_original.png")
            original_img.save(original_path)

            lime_path = os.path.join(output_dir, f"{base_name}_lime.png")
            self.fig_lime.savefig(lime_path, format='png', bbox_inches='tight')
            lime_img = Image.open(lime_path).convert('RGB')

            gradcam_path = os.path.join(output_dir, f"{base_name}_gradcam.png")
            self.fig_gradcam.savefig(gradcam_path, format='png', bbox_inches='tight')
            gradcam_img = Image.open(gradcam_path).convert('RGB')

            lime_gradcam_path = os.path.join(output_dir, f"{base_name}_lime_gradcam.png")
            self.fig_lime_gradcam.savefig(lime_gradcam_path, format='png', bbox_inches='tight')
            lime_gradcam_img = Image.open(lime_gradcam_path).convert('RGB')

            contrib_path = os.path.join(output_dir, f"{base_name}_contrib.png")
            self.fig_contrib.savefig(contrib_path, format='png', bbox_inches='tight')
            contrib_img = Image.open(contrib_path).convert('RGB')

            feedback_path = os.path.join(output_dir, f"{base_name}_feedback.png")
            feedback_img = Image.open(self.current_image_path).convert('RGB')
            draw = ImageDraw.Draw(feedback_img)
            for idx, polygon in enumerate(self.polygons):
                for start_idx, end_idx in polygon['edges']:
                    draw.line([polygon['points'][start_idx], polygon['points'][end_idx]], fill='red', width=2)
                for x, y in polygon['points']:
                    draw.ellipse([x - 5, y - 5, x + 5, y + 5], fill='red')
                if polygon['label']:
                    points = np.array(polygon['points'])
                    centroid_x = np.mean(points[:, 0])
                    centroid_y = np.mean(points[:, 1])
                    draw.text((centroid_x, centroid_y), f"Polygon {idx + 1}: {polygon['label']}", fill='blue')
            feedback_img.save(feedback_path)

            metadata = {
                'image_name': self.image_name,
                'timestamp': timestamp,
                'predicted_class': self.current_data.get('predicted_class', 'N/A'),
                'confidence': self.current_data.get('confidence', 'N/A'),
                'non_suspicious_prob': self.current_data.get('non_suspicious_prob', 'N/A'),
                'suspicious_prob': self.current_data.get('suspicious_prob', 'N/A'),
                'gradcam_class': self.current_data.get('gradcam_class', 'N/A'),
                'gradcam_confidence': self.current_data.get('gradcam_confidence', 'N/A'),
                'polygons': self.polygons,
                'segments_count': len(self.current_data.get('positive_segments', []))
            }
            metadata_path = os.path.join(output_dir, f"{base_name}_metadata.json")
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=4)

            self.status_var.set(f"Results saved to {output_dir}")
            logging.info(f"Results saved: original={original_path}, lime={lime_path}, gradcam={gradcam_path}, lime_gradcam={lime_gradcam_path}, contrib={contrib_path}, feedback={feedback_path}, metadata={metadata_path}")
            messagebox.showinfo("Save Successful", f"Results saved to:\n{output_dir}")
        except Exception as e:
            logging.error(f"Failed to save results: {str(e)}")
            messagebox.showerror("Error", f"Failed to save results: {str(e)}")
            self.status_var.set("Failed to save results")
        
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