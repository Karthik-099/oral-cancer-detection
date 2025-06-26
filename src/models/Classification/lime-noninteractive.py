import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.models import efficientnet_b3, EfficientNet_B3_Weights
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('tkagg')
import seaborn as sns
import os
import numpy as np
from lime import lime_image
from skimage.segmentation import slic, mark_boundaries
import cv2
import warnings
import time
from matplotlib.colors import LinearSegmentedColormap

warnings.filterwarnings('ignore')

global_val_df = None

CONFIG = {
    'DATA_DIR': '/home/user/Desktop/oral_cancer/gdc_oral_cancer_dataset/oral-cancer-detection/data/original',
    'CSV_PATH': '/home/user/Desktop/oral_cancer/gdc_oral_cancer_dataset/oral-cancer-detection/data/processed_labels.csv',
    'MODEL_LOAD_PATH': 'models/best_efficientnet_oral_cancer.pth',  # Changed to load path
    'RESULTS_DIR': 'results_final',
    'TRAIN_CSV_PATH': 'data/train_split.csv',
    'TEST_CSV_PATH': 'data/test_split.csv',
    'BATCH_SIZE': 16,
    'EPOCHS': 10,
    'LEARNING_RATE': 0.001,
    'NUM_CLASSES': 2,
    'IMAGE_SIZE': 300,
    'DEVICE': torch.device('cuda' if torch.cuda.is_available() else 'cpu')
}

os.makedirs(os.path.dirname(CONFIG['TRAIN_CSV_PATH']), exist_ok=True)
os.makedirs('models', exist_ok=True)
os.makedirs(CONFIG['RESULTS_DIR'], exist_ok=True)

class OralCancerDataset(Dataset):
    def __init__(self, df, data_dir, transform=None):
        self.df = df.reset_index(drop=True)
        self.data_dir = data_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_name = self.df.iloc[idx]['image_path']
        label = int(self.df.iloc[idx]['label'])
        img_path = os.path.join(self.data_dir, img_name)
        
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            image = Image.new('RGB', (CONFIG['IMAGE_SIZE'], CONFIG['IMAGE_SIZE']), color=(128, 128, 128))
        
        if self.transform:
            image = self.transform(image)
        
        return image, label, img_path

class EnhancedEfficientNet(nn.Module):
    def __init__(self, num_classes=2, dropout_rate=0.4):
        super(EnhancedEfficientNet, self).__init__()
        
        self.backbone = efficientnet_b3(weights=EfficientNet_B3_Weights.IMAGENET1K_V1)
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

def get_transforms():
    train_transform = transforms.Compose([
        transforms.Resize((CONFIG['IMAGE_SIZE'] + 32, CONFIG['IMAGE_SIZE'] + 32)),
        transforms.RandomCrop((CONFIG['IMAGE_SIZE'], CONFIG['IMAGE_SIZE'])),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((CONFIG['IMAGE_SIZE'], CONFIG['IMAGE_SIZE'])),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_transform

def detect_mouth_region(image):
    img_array = np.array(image)
    hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
    lower_red1 = np.array([0, 50, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 50, 50])
    upper_red2 = np.array([180, 255, 255])
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = mask1 + mask2
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return mask > 0

def create_gradient_overlay(image, segments, feature_importance, top_segments, is_suspicious=True):
    """Create gradient overlay based on segment importance with yellow to red gradient"""
    overlay = np.zeros_like(image, dtype=np.float32)
    
    if not top_segments:
        return overlay
    
    # Get importance values for top segments only
    top_importance_values = [abs(imp) for seg_id, imp in feature_importance if seg_id in top_segments]
    
    if top_importance_values:
        min_importance = min(top_importance_values)
        max_importance = max(top_importance_values)
        importance_range = max_importance - min_importance if max_importance != min_importance else 1
        
        for segment_id, importance in feature_importance:
            if segment_id in top_segments:
                mask = segments == segment_id
                if np.any(mask):
                    # Normalize importance within the range of top segments
                    abs_importance = abs(importance)
                    normalized_importance = (abs_importance - min_importance) / importance_range
                    
                    # Create yellow to red gradient
                    # Yellow (1,1,0) -> Orange (1,0.5,0) -> Red (1,0,0)
                    if normalized_importance <= 0.5:
                        # Yellow to Orange
                        ratio = normalized_importance * 2  # 0 to 1
                        color = np.array([1.0, 1.0 - ratio * 0.5, 0.0])  # (1,1,0) to (1,0.5,0)
                    else:
                        # Orange to Red
                        ratio = (normalized_importance - 0.5) * 2  # 0 to 1
                        color = np.array([1.0, 0.5 - ratio * 0.5, 0.0])  # (1,0.5,0) to (1,0,0)
                    
                    # Apply consistent opacity
                    alpha = 0.6
                    overlay[mask] = color * alpha
    
    return overlay

def create_segment_boundaries_with_labels(image, segments, top_segments, feature_importance):
    """Create segment boundaries with transparent colors and labels - only for top segments"""
    # Create boundary image
    boundary_image = np.array(image, dtype=np.float32) / 255.0
    
    # Create custom colors for boundaries (more vibrant and visible)
    boundary_colors = [
        (1, 0, 1),      # Magenta
        (0, 1, 1),      # Cyan  
        (1, 1, 0),      # Yellow
        (0, 0, 1),      # Blue
        (1, 0.5, 0)     # Orange
    ]
    
    label_positions = []
    
    # Only process top segments
    for i, (segment_id, importance) in enumerate(top_segments.items()):
        mask = segments == segment_id
        if np.any(mask):
            # Create a temporary image with only this segment
            temp_segments = np.zeros_like(segments)
            temp_segments[mask] = 1
            
            # Get segment boundary with thicker, more visible lines
            boundaries = mark_boundaries(
                np.zeros_like(boundary_image), 
                temp_segments, 
                color=boundary_colors[i % len(boundary_colors)], 
                mode='thick'
            )
            
            # Apply boundary with higher opacity for better visibility
            boundary_mask = np.any(boundaries != 0, axis=2)
            alpha = 0.9  # Higher opacity for better visibility
            boundary_image[boundary_mask] = (1 - alpha) * boundary_image[boundary_mask] + alpha * boundaries[boundary_mask]
            
            # Calculate centroid for label placement
            y_coords, x_coords = np.where(mask)
            if len(y_coords) > 0 and len(x_coords) > 0:
                centroid_y, centroid_x = int(np.mean(y_coords)), int(np.mean(x_coords))
                label_positions.append((centroid_x, centroid_y, f'S{segment_id}', boundary_colors[i % len(boundary_colors)]))
    
    return boundary_image, label_positions

class EnhancedLIMEExplainer:
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.mean = np.array([0.485, 0.456, 0.406])
        self.std = np.array([0.229, 0.224, 0.225])
        self.class_names = ['Non-Suspicious', 'Suspicious']
    
    def preprocess_image(self, image):
        if image.max() > 1.0:
            image = image / 255.0
        normalized = (image - self.mean) / self.std
        tensor = torch.FloatTensor(normalized).permute(2, 0, 1).unsqueeze(0)
        return tensor.to(self.device)
    
    def predict_fn(self, images):
        self.model.eval()
        predictions = []
        with torch.no_grad():
            for img in images:
                tensor = self.preprocess_image(img)
                output = self.model(tensor)
                prob = torch.nn.functional.softmax(output, dim=1)
                predictions.append(prob.cpu().numpy()[0])
        return np.array(predictions)
    
    def enhanced_segmentation(self, image):
        mouth_mask = detect_mouth_region(Image.fromarray(image.astype(np.uint8)))
        segments = slic(image, n_segments=150, compactness=20, start_label=1, 
                       mask=mouth_mask, enforce_connectivity=True)
        return segments
    
    def explain_image_with_annotations(self, image_path, num_samples=1500):
        original_image = Image.open(image_path).convert('RGB')
        resized_image = original_image.resize((CONFIG['IMAGE_SIZE'], CONFIG['IMAGE_SIZE']))
        img_array = np.array(resized_image)
        
        img_tensor = self.preprocess_image(img_array)
        with torch.no_grad():
            output = self.model(img_tensor)
            probs = torch.nn.functional.softmax(output, dim=1)
            predicted_class = torch.argmax(probs, dim=1).item()
            confidence = probs.max().item()
        
        explainer = lime_image.LimeImageExplainer()
        explanation = explainer.explain_instance(
            img_array,
            self.predict_fn,
            top_labels=2,
            hide_color=0,
            num_samples=num_samples,
            segmentation_fn=self.enhanced_segmentation
        )
        
        filtered_df = global_val_df[global_val_df['image_path'] == image_path]
        if not filtered_df.empty:
            cancer_type = filtered_df['cancer_type'].values[0]
            cancer_type_name = {
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
            }.get(cancer_type, 'Unspecified')
        else:
            cancer_type = -1
            cancer_type_name = 'Unspecified'
        
        return explanation, resized_image, predicted_class, confidence, cancer_type, cancer_type_name

def create_annotated_lime_visualization(model, image_path, save_path=None):
    explainer = EnhancedLIMEExplainer(model, CONFIG['DEVICE'])
    explanation, resized_image, pred_class, confidence, cancer_type, cancer_type_name = explainer.explain_image_with_annotations(image_path)
    
    temp_pos, mask_pos = explanation.get_image_and_mask(
        label=pred_class, positive_only=True, num_features=15, hide_rest=False
    )
    segments = explainer.enhanced_segmentation(np.array(resized_image))
    
    # Create base annotated image
    annotated_image = np.array(resized_image).astype(np.float32) / 255.0
    
    # Get feature importance scores
    feature_importance = explanation.local_exp[pred_class]
    
    # Create importance dictionary for all segments
    importance_dict = {seg_id: imp for seg_id, imp in feature_importance}
    
    # Get max absolute importance for normalization
    if importance_dict:
        max_abs_importance = max(abs(imp) for imp in importance_dict.values())
    else:
        max_abs_importance = 1.0
    
    # Create colored overlay based on contribution levels
    colored_overlay = np.zeros_like(annotated_image)
    
    # Get only positive contributions for coloring
    positive_contributions = [(seg_id, imp) for seg_id, imp in feature_importance if imp > 0]
    
    if positive_contributions:
        # Get min and max positive contributions for normalization
        positive_importances = [imp for _, imp in positive_contributions]
        min_positive = min(positive_importances)
        max_positive = max(positive_importances)
        
        # Color only positively contributing segments
        for segment_id, importance in positive_contributions:
            mask = segments == segment_id
            if np.any(mask):
                # Normalize importance from min to max positive range (0 to 1)
                if max_positive > min_positive:
                    normalized_importance = (importance - min_positive) / (max_positive - min_positive)
                else:
                    normalized_importance = 1.0
                
                # Create red to yellow gradient based on contribution level
                # normalized_importance = 0 -> yellow (min contribution)
                # normalized_importance = 1 -> red (max contribution)
                red_component = 1.0
                green_component = 1.0 - normalized_importance * 0.7  # From 1.0 (yellow) to 0.3 (red-ish)
                blue_component = 0.0
                
                color = np.array([red_component, green_component, blue_component])
                
                # Apply color with transparency based on contribution strength
                alpha = 0.3 + normalized_importance * 0.4  # Vary alpha from 0.3 to 0.7
                colored_overlay[mask] = color * alpha
    
    # Blend the colored overlay with the original image
    alpha_blend = 0.5
    annotated_image = (1 - alpha_blend) * annotated_image + alpha_blend * colored_overlay
    
    # Add boundaries for all segments
    boundary_image = np.copy(annotated_image)
    
    # Add boundaries for all segments in light gray
    unique_segments = np.unique(segments)
    for segment_id in unique_segments:
        if segment_id == 0:  # Skip background
            continue
            
        mask = segments == segment_id
        if np.any(mask):
            # Create boundary for this segment
            boundaries = mark_boundaries(np.zeros_like(boundary_image), segments == segment_id, 
                                       color=(0.7, 0.7, 0.7), mode='thick')  # Light gray boundaries
            
            # Apply boundary
            boundary_mask = np.any(boundaries > 0, axis=2)
            boundary_image[boundary_mask] = boundaries[boundary_mask]
    
    # Determine top contributing segments for labels and dark red boundaries
    if pred_class == 0:
        relevant_features = [(seg_id, imp) for seg_id, imp in feature_importance if imp < 0]
        relevant_features.sort(key=lambda x: x[1])
        top_features = relevant_features[:5]
        chart_title = 'Top 5 Segments Supporting Non-suspicious\n(Match with annotations)'
        positive_evidence_title = f'Positive Evidence\n(Normal Supporting Features)'
        bar_color = 'green'
    else:
        relevant_features = [(seg_id, imp) for seg_id, imp in feature_importance if imp > 0]
        relevant_features.sort(key=lambda x: x[1], reverse=True)
        top_features = relevant_features[:5]
        chart_title = 'Top 5 Segments Supporting suspicious\n(Match with annotations)'
        positive_evidence_title = f'Positive Evidence\n(Cancer Supporting Features)'
        bar_color = 'red'
    
    important_segments = {seg_id: imp for seg_id, imp in top_features}
    
    # Add dark red boundaries for top contributing segments and labels
    label_positions = []
    for i, (segment_id, importance) in enumerate(important_segments.items()):
        mask = segments == segment_id
        if np.any(mask):
            # Add dark red boundary for important segments
            boundaries = mark_boundaries(np.zeros_like(boundary_image), segments == segment_id, 
                                       color=(0.8, 0, 0), mode='thick')  # Dark red boundaries
            
            boundary_mask = np.any(boundaries > 0, axis=2)
            boundary_image[boundary_mask] = boundaries[boundary_mask]
            
            # Calculate centroid for label placement
            y_coords, x_coords = np.where(mask)
            if len(y_coords) > 0 and len(x_coords) > 0:
                centroid_y, centroid_x = int(np.mean(y_coords)), int(np.mean(x_coords))
                label_positions.append((centroid_x, centroid_y, f'S{segment_id}', (0.8, 0, 0)))
    
    # Convert back to uint8 for display
    boundary_image = np.clip(boundary_image * 255, 0, 255).astype(np.uint8)
    
    # Create importance heatmap
    importance_scores = np.zeros_like(mask_pos, dtype=float)
    for region_id, score in feature_importance:
        importance_scores[mask_pos == region_id] = score
    
    # Get true label information
    filtered_df = global_val_df[global_val_df['image_path'] == image_path]
    if not filtered_df.empty:
        true_label = filtered_df['label'].values[0]
        true_label_name = 'Suspicious' if true_label == 1 else 'Non-suspicious'
    else:
        print(f"Warning: Image path {image_path} not found in validation DataFrame.")
        true_label_name = "Unknown"
        true_label = -1
    
    # Create the visualization
    fig, axes = plt.subplots(2, 3, figsize=(20, 14))
    
    # Original Image
    axes[0, 0].imshow(resized_image)
    axes[0, 0].set_title(f'Original Image\n{os.path.basename(image_path)}\nTrue Label: {true_label_name}', fontweight='bold')
    axes[0, 0].axis('off')
    
    # Importance Heatmap
    im = axes[0, 1].imshow(importance_scores, cmap='RdYlBu_r')
    plt.colorbar(im, ax=axes[0, 1], fraction=0.046, pad=0.04)
    axes[0, 1].set_title(f'Importance Heatmap\nConfidence: {confidence:.3f}\nSuspicious Type: {cancer_type_name}', fontweight='bold')
    axes[0, 1].axis('off')
    
    # Annotated image with colored segments and boundaries
    axes[0, 2].imshow(boundary_image)
    
    # Add segment labels for top contributing segments
    for x, y, label, color in label_positions:
        axes[0, 2].text(x, y, label, fontsize=10, fontweight='bold', 
                       color='white', ha='center', va='center',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.7))
    
    color = 'green' if pred_class == 0 else 'red'
    axes[0, 2].set_title(f'Positively Contributing Segments (Red-Yellow Gradient)\nPrediction: {explainer.class_names[pred_class]}', 
                         fontweight='bold', color=color)
    axes[0, 2].axis('off')
    
    # Positive evidence
    axes[1, 0].imshow(mark_boundaries(temp_pos / 255.0, mask_pos))
    axes[1, 0].set_title(positive_evidence_title, fontweight='bold')
    axes[1, 0].axis('off')
    
    # All segments with boundaries (original style)
    all_boundaries = mark_boundaries(np.array(resized_image) / 255.0, segments, color=(0, 1, 0), mode='thick')
    axes[1, 1].imshow(all_boundaries)
    axes[1, 1].set_title('All Segments\n(Superpixel Boundaries)', fontweight='bold')
    axes[1, 1].axis('off')
    
    # Bar chart for top contributing segments
    segments_ids = [f'Segment {seg_id}' for seg_id, _ in top_features]
    importances = [imp for _, imp in top_features]
    colors = [bar_color for _ in importances]
    
    y_pos = np.arange(len(segments_ids))
    bars = axes[1, 2].barh(y_pos, importances, color=colors, alpha=0.7)
    axes[1, 2].set_yticks(y_pos)
    axes[1, 2].set_yticklabels(segments_ids)
    axes[1, 2].set_xlabel('Importance Score')
    axes[1, 2].set_title(chart_title, fontweight='bold')
    axes[1, 2].grid(True, alpha=0.3)
    axes[1, 2].invert_yaxis()
    if pred_class == 0:
        axes[1, 2].invert_xaxis()
    
    # Add value labels on bars
    for i, bar in enumerate(bars):
        width = bar.get_width()
        ha = 'right' if pred_class == 0 else 'left'
        x_pos = width if pred_class == 0 else width
        axes[1, 2].text(x_pos, bar.get_y() + bar.get_height()/2, 
                        f'{width:.3f}', ha=ha, va='center', fontweight='bold')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Enhanced LIME visualization saved: {save_path}")
    plt.close()  # Close the figure to free memory
    
    return {
        'prediction': pred_class,
        'confidence': confidence,
        'class_name': explainer.class_names[pred_class],
        'top_segments': important_segments,
        'true_label': true_label,
        'cancer_type': cancer_type,
        'cancer_type_name': cancer_type_name
    }

def load_trained_model(model_path):
    """Load pre-trained model from path"""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    model = EnhancedEfficientNet(num_classes=CONFIG['NUM_CLASSES']).to(CONFIG['DEVICE'])
    
    try:
        checkpoint = torch.load(model_path, map_location=CONFIG['DEVICE'])
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            best_acc = checkpoint.get('best_acc', 'Unknown')
            print(f" Model loaded successfully from {model_path}")
            print(f"   Model accuracy: {best_acc}")
        else:
            # Direct state dict
            model.load_state_dict(checkpoint)
            print(f" Model loaded successfully from {model_path}")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        raise
    
    model.eval()
    return model

def main():
    global global_val_df
    
    # Load CSV file
    if not os.path.exists(CONFIG['CSV_PATH']):
        print(" CSV file not found!")
        return
    
    df = pd.read_csv(CONFIG['CSV_PATH'])
    df = df.dropna(subset=['label'])
    print(f" Loaded {len(df)} samples")
    print(f"Class distribution:\n{df['label'].value_counts()}")
    
    # Set global validation dataframe
    global_val_df = df
    
    # Load pre-trained model instead of training
    print(f"\n Loading pre-trained model from: {CONFIG['MODEL_LOAD_PATH']}")
    try:
        model = load_trained_model(CONFIG['MODEL_LOAD_PATH'])
    except Exception as e:
        print(f" Failed to load model: {e}")
        return
    
    # Sample images for analysis
    sample_images = [
        '/home/user/Desktop/oral_cancer/gdc_oral_cancer_dataset/oral-cancer-detection/data/original/patient_279/image_4.jpeg',
        '/home/user/Desktop/oral_cancer/gdc_oral_cancer_dataset/oral-cancer-detection/data/original/patient_47/image_4.jpeg',
        '/home/user/Desktop/oral_cancer/gdc_oral_cancer_dataset/oral-cancer-detection/data/original/patient_257/image_4.jpeg',
        '/home/user/Desktop/oral_cancer/gdc_oral_cancer_dataset/oral-cancer-detection/data/original/patient_257/image_9.jpeg',
        '/home/user/Desktop/oral_cancer/gdc_oral_cancer_dataset/oral-cancer-detection/data/original/patient_236/image_1.jpeg',
        '/home/user/Desktop/oral_cancer/gdc_oral_cancer_dataset/oral-cancer-detection/data/original/patient_491/image_3.jpeg',
        '/home/user/Desktop/oral_cancer/gdc_oral_cancer_dataset/oral-cancer-detection/data/original/patient_452/image_1.jpeg',
        '/home/user/Desktop/oral_cancer/gdc_oral_cancer_dataset/oral-cancer-detection/data/original/patient_472/image_1.jpeg',
        '/home/user/Desktop/oral_cancer/gdc_oral_cancer_dataset/oral-cancer-detection/data/original/patient_349/image_2.jpeg'
    ]
    
    if sample_images:
        print(f"\n🔍 Analyzing {len(sample_images)} images with Enhanced LIME...")
        for i, img_path in enumerate(sample_images):
            if os.path.exists(img_path):
                print(f"Processing image {i+1}: {os.path.basename(img_path)}")
                save_path = os.path.join(CONFIG['RESULTS_DIR'], f'enhanced_lime_analysis_{i+1}.png')
                result = create_annotated_lime_visualization(model, img_path, save_path)
                true_label_str = "Unknown" if result['true_label'] == -1 else ('Suspicious' if result['true_label'] == 1 else 'Non-suspicious')
                print(f"Result: {result['class_name']} (Confidence: {result['confidence']:.3f}), "
                      f"True Label: {true_label_str}, Cancer Type: {result['cancer_type_name']}")
            else:
                print(f"⚠️ Image not found: {img_path}")
    
    print(f"\n✅ Analysis complete! Results saved in: {CONFIG['RESULTS_DIR']}")

if __name__ == '__main__':
    main()