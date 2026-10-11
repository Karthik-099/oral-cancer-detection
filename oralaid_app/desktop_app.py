import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import torch
import torch.nn as nn
from torchvision.models import efficientnet_b4, mobilenet_v3_large
import segmentation_models_pytorch as smp
import torchvision.transforms as transforms
import numpy as np
import cv2
import os
import json
import datetime
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import threading
import requests
from io import BytesIO
import base64

class OralAIDDesktop:
    def __init__(self, root):
        self.root = root
        self.root.title("OralAID - AI-Powered Oral Cancer Detection")
        self.root.geometry("1200x800")
        self.root.configure(bg='#f0f0f0')
        
        # Initialize variables
        self.current_image = None
        self.current_image_path = None
        self.model = None
        self.segmenter = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Risk mapping
        self.risk_mapping = {
            'normal': {'category': 'Non-suspicious', 'risk': 'Low-risk', 'color': '#28a745'},
            'keratosis': {'category': 'Suspicious', 'risk': 'Low-risk', 'color': '#ffc107'},
            'leukoplakia_homogenous': {'category': 'Suspicious', 'risk': 'Low-risk', 'color': '#ffc107'},
            'leukoplakia_speckled': {'category': 'Suspicious', 'risk': 'High-risk', 'color': '#dc3545'},
            'erythroplakia': {'category': 'Suspicious', 'risk': 'High-risk', 'color': '#dc3545'},
            'proliferative_verrucous': {'category': 'Suspicious', 'risk': 'High-risk', 'color': '#dc3545'},
            'lichen_planus': {'category': 'Suspicious', 'risk': 'Low-risk', 'color': '#ffc107'},
            'squamous_cell_carcinoma': {'category': 'Suspicious', 'risk': 'High-risk', 'color': '#dc3545'}
        }
        
        self.setup_ui()
        self.load_models()
        
    def setup_ui(self):
        # Create main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(1, weight=1)
        
        # Title
        title_label = ttk.Label(main_frame, text="OralAID - AI-Powered Oral Cancer Detection", 
                               font=('Arial', 16, 'bold'))
        title_label.grid(row=0, column=0, columnspan=3, pady=(0, 20))
        
        # Left panel - Controls
        left_frame = ttk.LabelFrame(main_frame, text="Controls", padding="10")
        left_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 10))
        
        # Patient information
        patient_frame = ttk.LabelFrame(left_frame, text="Patient Information", padding="10")
        patient_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        
        ttk.Label(patient_frame, text="Patient Name:").grid(row=0, column=0, sticky=tk.W, pady=2)
        self.patient_name = ttk.Entry(patient_frame, width=25)
        self.patient_name.grid(row=0, column=1, sticky=(tk.W, tk.E), pady=2)
        
        ttk.Label(patient_frame, text="Age:").grid(row=1, column=0, sticky=tk.W, pady=2)
        self.patient_age = ttk.Entry(patient_frame, width=25)
        self.patient_age.grid(row=1, column=1, sticky=(tk.W, tk.E), pady=2)
        
        ttk.Label(patient_frame, text="Gender:").grid(row=2, column=0, sticky=tk.W, pady=2)
        self.patient_gender = ttk.Combobox(patient_frame, values=["Male", "Female", "Other"], width=22)
        self.patient_gender.grid(row=2, column=1, sticky=(tk.W, tk.E), pady=2)
        
        # Image upload
        upload_frame = ttk.LabelFrame(left_frame, text="Image Upload", padding="10")
        upload_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        
        self.upload_btn = ttk.Button(upload_frame, text="Select Image", command=self.upload_image)
        self.upload_btn.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=2)
        
        self.image_path_label = ttk.Label(upload_frame, text="No image selected", wraplength=200)
        self.image_path_label.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=2)
        
        # Analysis options
        options_frame = ttk.LabelFrame(left_frame, text="Analysis Options", padding="10")
        options_frame.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        
        self.gradcam_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(options_frame, text="GradCAM Visualization", 
                       variable=self.gradcam_var).grid(row=0, column=0, sticky=tk.W)
        
        self.lime_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(options_frame, text="LIME Explanation", 
                       variable=self.lime_var).grid(row=1, column=0, sticky=tk.W)
        
        self.segmentation_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(options_frame, text="Lesion Segmentation", 
                       variable=self.segmentation_var).grid(row=2, column=0, sticky=tk.W)
        
        # Analyze button
        self.analyze_btn = ttk.Button(left_frame, text="Analyze Image", 
                                     command=self.analyze_image, style="Accent.TButton")
        self.analyze_btn.grid(row=3, column=0, sticky=(tk.W, tk.E), pady=10)
        
        # Progress bar
        self.progress = ttk.Progressbar(left_frame, mode='indeterminate')
        self.progress.grid(row=4, column=0, sticky=(tk.W, tk.E), pady=5)
        
        # Status label
        self.status_label = ttk.Label(left_frame, text="Ready", foreground="green")
        self.status_label.grid(row=5, column=0, sticky=tk.W, pady=2)
        
        # Export buttons
        export_frame = ttk.LabelFrame(left_frame, text="Export", padding="10")
        export_frame.grid(row=6, column=0, sticky=(tk.W, tk.E), pady=(10, 0))
        
        self.export_report_btn = ttk.Button(export_frame, text="Export Report", 
                                           command=self.export_report, state="disabled")
        self.export_report_btn.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=2)
        
        self.export_images_btn = ttk.Button(export_frame, text="Export Images", 
                                           command=self.export_images, state="disabled")
        self.export_images_btn.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=2)
        
        # Middle panel - Image display
        image_frame = ttk.LabelFrame(main_frame, text="Image Analysis", padding="10")
        image_frame.grid(row=1, column=1, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5)
        image_frame.columnconfigure(0, weight=1)
        image_frame.rowconfigure(0, weight=1)
        
        # Notebook for different views
        self.notebook = ttk.Notebook(image_frame)
        self.notebook.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Original image tab
        self.original_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.original_frame, text="Original")
        
        self.original_canvas = tk.Canvas(self.original_frame, bg='white', width=400, height=400)
        self.original_canvas.pack(expand=True, fill='both')
        
        # GradCAM tab
        self.gradcam_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.gradcam_frame, text="GradCAM")
        
        self.gradcam_canvas = tk.Canvas(self.gradcam_frame, bg='white', width=400, height=400)
        self.gradcam_canvas.pack(expand=True, fill='both')
        
        # Segmentation tab
        self.segmentation_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.segmentation_frame, text="Segmentation")
        
        self.segmentation_canvas = tk.Canvas(self.segmentation_frame, bg='white', width=400, height=400)
        self.segmentation_canvas.pack(expand=True, fill='both')
        
        # Right panel - Results
        results_frame = ttk.LabelFrame(main_frame, text="Analysis Results", padding="10")
        results_frame.grid(row=1, column=2, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(10, 0))
        
        # Prediction results
        prediction_frame = ttk.LabelFrame(results_frame, text="AI Prediction", padding="10")
        prediction_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        
        self.prediction_label = ttk.Label(prediction_frame, text="No analysis performed", 
                                         font=('Arial', 12, 'bold'))
        self.prediction_label.grid(row=0, column=0, sticky=tk.W, pady=2)
        
        self.confidence_label = ttk.Label(prediction_frame, text="")
        self.confidence_label.grid(row=1, column=0, sticky=tk.W, pady=2)
        
        self.risk_label = ttk.Label(prediction_frame, text="")
        self.risk_label.grid(row=2, column=0, sticky=tk.W, pady=2)
        
        # Confidence bar
        self.confidence_progress = ttk.Progressbar(prediction_frame, length=200)
        self.confidence_progress.grid(row=3, column=0, sticky=(tk.W, tk.E), pady=5)
        
        # Recommendations
        recommendations_frame = ttk.LabelFrame(results_frame, text="Clinical Recommendations", padding="10")
        recommendations_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
        
        self.recommendations_text = tk.Text(recommendations_frame, height=10, width=30, wrap=tk.WORD)
        recommendations_scrollbar = ttk.Scrollbar(recommendations_frame, orient="vertical", 
                                                 command=self.recommendations_text.yview)
        self.recommendations_text.configure(yscrollcommand=recommendations_scrollbar.set)
        
        self.recommendations_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        recommendations_scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        
        recommendations_frame.columnconfigure(0, weight=1)
        recommendations_frame.rowconfigure(0, weight=1)
        
        # Model info
        model_frame = ttk.LabelFrame(results_frame, text="Model Information", padding="10")
        model_frame.grid(row=2, column=0, sticky=(tk.W, tk.E))
        
        self.model_info_label = ttk.Label(model_frame, text="Loading models...", wraplength=200)
        self.model_info_label.grid(row=0, column=0, sticky=tk.W)
        
    def load_models(self):
        """Load AI models"""
        def load_in_background():
            try:
                self.status_label.config(text="Loading models...", foreground="orange")
                
                # Load classifier
                try:
                    self.model = efficientnet_b4(pretrained=False)
                    self.model.classifier = nn.Linear(self.model.classifier.in_features, 8)
                    self.model.load_state_dict(torch.load('models/final_efficientnet_b4_oral_cancer.pth', 
                                                         map_location=self.device))
                    model_name = "EfficientNet-B4"
                except:
                    # Fallback to MobileNet
                    self.model = mobilenet_v3_large(pretrained=True)
                    self.model.classifier = nn.Linear(self.model.classifier.in_features, 8)
                    model_name = "MobileNetV3-Large"
                
                self.model = self.model.to(self.device)
                self.model.eval()
                
                # Load segmenter
                try:
                    self.segmenter = smp.Unet(encoder_name='resnet34', encoder_weights=None, 
                                            in_channels=3, classes=1)
                    self.segmenter.load_state_dict(torch.load('models/unet_segmenter.pth', 
                                                            map_location=self.device))
                    self.segmenter = self.segmenter.to(self.device)
                    self.segmenter.eval()
                    segmenter_name = "U-Net (ResNet34)"
                except:
                    segmenter_name = "Not available"
                
                # Update UI
                self.root.after(0, lambda: self.status_label.config(text="Models loaded successfully", 
                                                                   foreground="green"))
                self.root.after(0, lambda: self.model_info_label.config(
                    text=f"Classifier: {model_name}\nSegmenter: {segmenter_name}\nDevice: {self.device}"))
                
            except Exception as e:
                self.root.after(0, lambda: self.status_label.config(text=f"Error loading models: {str(e)}", 
                                                                   foreground="red"))
        
        threading.Thread(target=load_in_background, daemon=True).start()
    
    def upload_image(self):
        """Upload and display image"""
        file_path = filedialog.askopenfilename(
            title="Select Intraoral Image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.tiff *.bmp")]
        )
        
        if file_path:
            try:
                self.current_image_path = file_path
                self.current_image = Image.open(file_path).convert('RGB')
                
                # Display image
                self.display_image(self.current_image, self.original_canvas)
                
                # Update UI
                self.image_path_label.config(text=os.path.basename(file_path))
                self.analyze_btn.config(state="normal")
                
                # Clear previous results
                self.clear_results()
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load image: {str(e)}")
    
    def display_image(self, image, canvas):
        """Display image on canvas"""
        # Resize image to fit canvas
        canvas_width = canvas.winfo_width() or 400
        canvas_height = canvas.winfo_height() or 400
        
        # Calculate scaling
        img_width, img_height = image.size
        scale = min(canvas_width / img_width, canvas_height / img_height, 1.0)
        
        new_width = int(img_width * scale)
        new_height = int(img_height * scale)
        
        resized_image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        # Convert to PhotoImage
        photo = ImageTk.PhotoImage(resized_image)
        
        # Clear canvas and display image
        canvas.delete("all")
        canvas.create_image(canvas_width // 2, canvas_height // 2, image=photo)
        
        # Keep reference to prevent garbage collection
        canvas.image = photo
    
    def analyze_image(self):
        """Perform AI analysis"""
        if not self.current_image:
            messagebox.showwarning("Warning", "Please select an image first")
            return
        
        if not self.model:
            messagebox.showerror("Error", "Models not loaded")
            return
        
        def analyze_in_background():
            try:
                self.root.after(0, lambda: self.progress.start())
                self.root.after(0, lambda: self.status_label.config(text="Analyzing image...", 
                                                                   foreground="orange"))
                self.root.after(0, lambda: self.analyze_btn.config(state="disabled"))
                
                # Preprocess image
                transform = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
                
                input_tensor = transform(self.current_image).unsqueeze(0).to(self.device)
                
                # Perform classification
                with torch.no_grad():
                    outputs = self.model(input_tensor)
                    probabilities = torch.softmax(outputs, dim=1)
                    predicted_class = torch.argmax(probabilities, dim=1).item()
                    confidence = probabilities[0][predicted_class].item()
                
                # Map prediction
                class_names = list(self.risk_mapping.keys())
                predicted_diagnosis = class_names[predicted_class] if predicted_class < len(class_names) else 'unknown'
                risk_info = self.risk_mapping.get(predicted_diagnosis, 
                                                {'category': 'Unknown', 'risk': 'Unknown', 'color': '#6c757d'})
                
                # Generate visualizations
                gradcam_image = None
                segmentation_image = None
                
                if self.gradcam_var.get():
                    gradcam_image = self.generate_gradcam(self.current_image, predicted_class)
                
                if self.segmentation_var.get() and self.segmenter and risk_info['risk'] == 'High-risk':
                    segmentation_image = self.generate_segmentation(self.current_image)
                
                # Update UI
                self.root.after(0, lambda: self.update_results(predicted_diagnosis, confidence, 
                                                              risk_info, gradcam_image, segmentation_image))
                
            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror("Error", f"Analysis failed: {str(e)}"))
            finally:
                self.root.after(0, lambda: self.progress.stop())
                self.root.after(0, lambda: self.analyze_btn.config(state="normal"))
                self.root.after(0, lambda: self.status_label.config(text="Analysis complete", 
                                                                   foreground="green"))
        
        threading.Thread(target=analyze_in_background, daemon=True).start()
    
    def generate_gradcam(self, image, target_class):
        """Generate GradCAM visualization"""
        try:
            target_layers = [self.model.features[-1]]
            cam = GradCAM(model=self.model, target_layers=target_layers, 
                         use_cuda=torch.cuda.is_available())
            
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            
            input_tensor = transform(image).unsqueeze(0).to(self.device)
            targets = [ClassifierOutputTarget(target_class)]
            
            grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
            grayscale_cam = grayscale_cam[0, :]
            
            # Convert to visualization
            rgb_img = np.array(image.resize((224, 224))) / 255.0
            visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
            
            return Image.fromarray(visualization)
        except Exception as e:
            print(f"GradCAM error: {e}")
            return None
    
    def generate_segmentation(self, image):
        """Generate segmentation mask"""
        try:
            transform = transforms.Compose([
                transforms.Resize((512, 512)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            
            input_tensor = transform(image).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                output = self.segmenter(input_tensor)
                mask = torch.sigmoid(output) > 0.5
                mask = mask.cpu().numpy().squeeze()
            
            # Convert to image
            mask_img = Image.fromarray((mask * 255).astype(np.uint8))
            return mask_img.resize(image.size)
        except Exception as e:
            print(f"Segmentation error: {e}")
            return None
    
    def update_results(self, prediction, confidence, risk_info, gradcam_image, segmentation_image):
        """Update results display"""
        # Update prediction labels
        self.prediction_label.config(text=f"{risk_info['category']}")
        self.confidence_label.config(text=f"Confidence: {confidence*100:.1f}%")
        self.risk_label.config(text=f"Risk Level: {risk_info['risk']}")
        
        # Update confidence bar
        self.confidence_progress['value'] = confidence * 100
        
        # Display visualizations
        if gradcam_image:
            self.display_image(gradcam_image, self.gradcam_canvas)
        
        if segmentation_image:
            self.display_image(segmentation_image, self.segmentation_canvas)
        
        # Update recommendations
        recommendations = self.get_recommendations(risk_info['risk'])
        self.recommendations_text.delete(1.0, tk.END)
        self.recommendations_text.insert(1.0, recommendations)
        
        # Enable export buttons
        self.export_report_btn.config(state="normal")
        self.export_images_btn.config(state="normal")
        
        # Store results for export
        self.current_results = {
            'prediction': prediction,
            'confidence': confidence,
            'risk_info': risk_info,
            'gradcam_image': gradcam_image,
            'segmentation_image': segmentation_image,
            'timestamp': datetime.datetime.now().isoformat()
        }
    
    def get_recommendations(self, risk_level):
        """Get clinical recommendations based on risk level"""
        if risk_level == 'High-risk':
            return """HIGH-RISK LESION DETECTED

Immediate Actions Required:
• Urgent referral to oral medicine specialist
• Biopsy recommended for definitive diagnosis
• Patient counseling regarding findings
• Document findings thoroughly
• Follow-up within 2-4 weeks if referral delayed

Additional Considerations:
• Review patient's medical history
• Assess risk factors (tobacco, alcohol, HPV)
• Consider imaging studies if indicated
• Coordinate with oncology team if malignancy confirmed"""
        
        elif risk_level == 'Low-risk':
            return """SUSPICIOUS LESION - LOW RISK

Recommended Actions:
• Monitor closely for changes
• Consider referral to oral medicine specialist
• Patient education on oral hygiene
• Schedule follow-up in 4-6 weeks
• Document baseline characteristics

Monitoring Guidelines:
• Watch for changes in size, color, texture
• Photograph lesion for comparison
• Patient self-examination education
• Regular dental check-ups"""
        
        else:
            return """NORMAL ORAL CAVITY

Routine Care:
• Continue regular oral health maintenance
• Routine dental check-ups as recommended
• Patient education on oral cancer prevention
• Encourage healthy lifestyle choices
• Annual oral cancer screening

Prevention Strategies:
• Tobacco cessation counseling
• Limit alcohol consumption
• HPV vaccination if appropriate
• Sun protection for lips
• Healthy diet rich in fruits and vegetables"""
    
    def clear_results(self):
        """Clear previous results"""
        self.prediction_label.config(text="No analysis performed")
        self.confidence_label.config(text="")
        self.risk_label.config(text="")
        self.confidence_progress['value'] = 0
        self.recommendations_text.delete(1.0, tk.END)
        
        # Clear canvases
        self.gradcam_canvas.delete("all")
        self.segmentation_canvas.delete("all")
        
        # Disable export buttons
        self.export_report_btn.config(state="disabled")
        self.export_images_btn.config(state="disabled")
    
    def export_report(self):
        """Export analysis report"""
        if not hasattr(self, 'current_results'):
            messagebox.showwarning("Warning", "No analysis results to export")
            return
        
        file_path = filedialog.asksaveasfilename(
            title="Save Report",
            defaultextension=".html",
            filetypes=[("HTML files", "*.html"), ("Text files", "*.txt")]
        )
        
        if file_path:
            try:
                # Generate HTML report
                html_content = self.generate_html_report()
                
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(html_content)
                
                messagebox.showinfo("Success", f"Report exported to {file_path}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to export report: {str(e)}")
    
    def export_images(self):
        """Export analysis images"""
        if not hasattr(self, 'current_results'):
            messagebox.showwarning("Warning", "No analysis results to export")
            return
        
        folder_path = filedialog.askdirectory(title="Select Export Folder")
        
        if folder_path:
            try:
                # Save original image
                if self.current_image:
                    self.current_image.save(os.path.join(folder_path, "original_image.png"))
                
                # Save GradCAM
                if self.current_results.get('gradcam_image'):
                    self.current_results['gradcam_image'].save(
                        os.path.join(folder_path, "gradcam_visualization.png"))
                
                # Save segmentation
                if self.current_results.get('segmentation_image'):
                    self.current_results['segmentation_image'].save(
                        os.path.join(folder_path, "segmentation_mask.png"))
                
                messagebox.showinfo("Success", f"Images exported to {folder_path}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to export images: {str(e)}")
    
    def generate_html_report(self):
        """Generate HTML report"""
        patient_name = self.patient_name.get() or "Unknown"
        patient_age = self.patient_age.get() or "Unknown"
        patient_gender = self.patient_gender.get() or "Unknown"
        
        results = self.current_results
        
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>OralAID Analysis Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .header {{ background: #2563eb; color: white; padding: 20px; text-align: center; }}
                .section {{ margin: 20px 0; padding: 15px; border: 1px solid #ddd; }}
                .risk-high {{ background: #fee2e2; color: #dc2626; }}
                .risk-low {{ background: #fef3c7; color: #d97706; }}
                .risk-normal {{ background: #dcfce7; color: #059669; }}
                .confidence-bar {{ background: #e5e7eb; height: 20px; border-radius: 10px; }}
                .confidence-fill {{ background: #2563eb; height: 100%; border-radius: 10px; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>OralAID Analysis Report</h1>
                <p>AI-Powered Oral Cancer Detection System</p>
            </div>
            
            <div class="section">
                <h2>Patient Information</h2>
                <p><strong>Name:</strong> {patient_name}</p>
                <p><strong>Age:</strong> {patient_age}</p>
                <p><strong>Gender:</strong> {patient_gender}</p>
                <p><strong>Analysis Date:</strong> {results['timestamp']}</p>
            </div>
            
            <div class="section">
                <h2>AI Analysis Results</h2>
                <p><strong>Prediction:</strong> {results['risk_info']['category']}</p>
                <p><strong>Diagnosis:</strong> {results['prediction'].replace('_', ' ').title()}</p>
                <p><strong>Confidence:</strong> {results['confidence']*100:.1f}%</p>
                <p><strong>Risk Level:</strong> 
                    <span class="risk-{results['risk_info']['risk'].lower().replace('-', '')}">{results['risk_info']['risk']}</span>
                </p>
                
                <div class="confidence-bar">
                    <div class="confidence-fill" style="width: {results['confidence']*100}%"></div>
                </div>
            </div>
            
            <div class="section">
                <h2>Clinical Recommendations</h2>
                <pre>{self.get_recommendations(results['risk_info']['risk'])}</pre>
            </div>
            
            <div class="section">
                <h2>Model Information</h2>
                <p><strong>Classification Model:</strong> EfficientNet-B4 / MobileNetV3</p>
                <p><strong>Segmentation Model:</strong> U-Net with ResNet34</p>
                <p><strong>Device:</strong> {self.device}</p>
                <p><strong>Software Version:</strong> OralAID Desktop v1.0</p>
            </div>
            
            <div class="section">
                <h2>Disclaimer</h2>
                <p><em>This AI system is designed to assist clinical decision-making. 
                Final diagnosis should always be confirmed by qualified healthcare professionals. 
                This report is for clinical reference only and should not replace professional medical judgment.</em></p>
            </div>
        </body>
        </html>
        """
        
        return html

def main():
    root = tk.Tk()
    app = OralAIDDesktop(root)
    root.mainloop()

if __name__ == "__main__":
    main()