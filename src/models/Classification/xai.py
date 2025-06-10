import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.models import efficientnet_b3, EfficientNet_B3_Weights
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
from lime import lime_image
from skimage.segmentation import slic, mark_boundaries
import cv2
import warnings
import time

warnings.filterwarnings('ignore')

# Enhanced Configuration
CONFIG = {
    'DATA_DIR': '/home/user/Desktop/oral_cancer/gdc_oral_cancer_dataset/oral-cancer-detection/data/original',
    'CSV_PATH': '/home/user/Desktop/oral_cancer/gdc_oral_cancer_dataset/oral-cancer-detection/data/processed_labels.csv',
    'MODEL_SAVE_PATH': 'models/best_efficientnet_oral_cancer.pth',
    'RESULTS_DIR': 'results',
    'TRAIN_CSV_PATH': 'data/train_split.csv',
    'TEST_CSV_PATH': 'data/test_split.csv',
    'BATCH_SIZE': 16,
    'EPOCHS': 15,
    'LEARNING_RATE': 0.001,
    'NUM_CLASSES': 2,
    'IMAGE_SIZE': 224,
    'DEVICE': torch.device('cuda' if torch.cuda.is_available() else 'cpu')
}

# Create directories
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
        
        # Enhanced classifier with better regularization
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
        transforms.Resize((CONFIG['IMAGE_SIZE'], CONFIG['IMAGE_SIZE'])),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15),
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
    """Detect and mask the mouth region using color-based segmentation"""
    img_array = np.array(image)
    
    # Convert to HSV for better color segmentation
    hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
    
    # Define range for pink/red colors (mouth interior)
    lower_red1 = np.array([0, 50, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 50, 50])
    upper_red2 = np.array([180, 255, 255])
    
    # Create masks for mouth region
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = mask1 + mask2
    
    # Morphological operations to clean up the mask
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    return mask > 0

class EnhancedLIMEExplainer:
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.mean = np.array([0.485, 0.456, 0.406])
        self.std = np.array([0.229, 0.224, 0.225])
        self.class_names = ['Normal', 'Cancer']
    
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
        """Enhanced segmentation focusing on mouth region"""
        # Detect mouth region
        mouth_mask = detect_mouth_region(Image.fromarray(image.astype(np.uint8)))
        
        # Use SLIC with more segments for better precision
        segments = slic(image, n_segments=150, compactness=20, start_label=1, 
                       mask=mouth_mask, enforce_connectivity=True)
        
        return segments
    
    def explain_image_with_annotations(self, image_path, num_samples=1500):
        original_image = Image.open(image_path).convert('RGB')
        resized_image = original_image.resize((CONFIG['IMAGE_SIZE'], CONFIG['IMAGE_SIZE']))
        img_array = np.array(resized_image)
        
        # Get model prediction
        img_tensor = self.preprocess_image(img_array)
        with torch.no_grad():
            output = self.model(img_tensor)
            probs = torch.nn.functional.softmax(output, dim=1)
            predicted_class = torch.argmax(probs, dim=1).item()
            confidence = probs.max().item()
        
        # Generate LIME explanation with enhanced segmentation
        explainer = lime_image.LimeImageExplainer()
        explanation = explainer.explain_instance(
            img_array,
            self.predict_fn,
            top_labels=2,
            hide_color=0,
            num_samples=num_samples,
            segmentation_fn=self.enhanced_segmentation
        )
        
        return explanation, resized_image, predicted_class, confidence

def create_annotated_lime_visualization(model, image_path, save_path=None):
    """Create LIME visualization with superpixel annotations"""
    explainer = EnhancedLIMEExplainer(model, CONFIG['DEVICE'])
    explanation, resized_image, pred_class, confidence = explainer.explain_image_with_annotations(image_path)
    
    # Get explanation components
    temp_pos, mask_pos = explanation.get_image_and_mask(
        label=pred_class, positive_only=True, num_features=15, hide_rest=False
    )
    
    # Get segments
    segments = explainer.enhanced_segmentation(np.array(resized_image))
    
    # Create annotated version with superpixel numbers
    annotated_image = np.array(resized_image).copy()
    
    # Get feature importance
    feature_importance = explanation.local_exp[pred_class]
    important_segments = {f[0]: f[1] for f in feature_importance[:10]}
    
    # Annotate important segments
    for segment_id, importance in important_segments.items():
        if abs(importance) > 0.01:  # Only annotate significant segments
            mask = segments == segment_id
            if np.any(mask):
                # Find centroid of the segment
                y_coords, x_coords = np.where(mask)
                centroid_y, centroid_x = int(np.mean(y_coords)), int(np.mean(x_coords))
                
                # Choose color based on importance
                color = (0, 255, 0) if importance > 0 else (255, 0, 0)
                
                # Draw annotation
                cv2.putText(annotated_image, f'{segment_id}', 
                           (centroid_x-10, centroid_y+5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                cv2.circle(annotated_image, (centroid_x, centroid_y), 3, color, -1)
    
    # Create visualization
    fig, axes = plt.subplots(2, 3, figsize=(20, 14))
    
    # Original Image
    axes[0, 0].imshow(resized_image)
    axes[0, 0].set_title(f'Original Image\n{os.path.basename(image_path)}', fontweight='bold')
    axes[0, 0].axis('off')
    
    # Prediction
    axes[0, 1].imshow(resized_image)
    color = 'green' if pred_class == 0 else 'red'
    axes[0, 1].set_title(f'Prediction: {explainer.class_names[pred_class]}\nConfidence: {confidence:.3f}', 
                        fontweight='bold', color=color)
    axes[0, 1].axis('off')
    
    # Annotated superpixels
    axes[0, 2].imshow(annotated_image)
    axes[0, 2].set_title('Annotated Superpixels\n(Numbers show segment IDs)', fontweight='bold')
    axes[0, 2].axis('off')
    
    # Positive evidence
    axes[1, 0].imshow(mark_boundaries(temp_pos / 255.0, mask_pos))
    axes[1, 0].set_title('Positive Evidence\n(Cancer Supporting Features)', fontweight='bold')
    axes[1, 0].axis('off')
    
    # Segment boundaries
    axes[1, 1].imshow(mark_boundaries(np.array(resized_image) / 255.0, segments))
    axes[1, 1].set_title('All Segments\n(Superpixel Boundaries)', fontweight='bold')
    axes[1, 1].axis('off')
    
    # Feature importance
    feature_importance.sort(key=lambda x: abs(x[1]), reverse=True)
    top_features = feature_importance[:10]
    
    segments_ids = [f'Segment {f[0]}' for f in top_features]
    importances = [f[1] for f in top_features]
    colors = ['green' if imp > 0 else 'red' for imp in importances]
    
    y_pos = np.arange(len(segments_ids))
    bars = axes[1, 2].barh(y_pos, importances, color=colors, alpha=0.7)
    axes[1, 2].set_yticks(y_pos)
    axes[1, 2].set_yticklabels(segments_ids)
    axes[1, 2].set_xlabel('Importance Score')
    axes[1, 2].set_title('Segment Contribution\n(Match with annotations)', fontweight='bold')
    axes[1, 2].grid(True, alpha=0.3)
    axes[1, 2].invert_yaxis()
    
    # Add value labels on bars
    for i, bar in enumerate(bars):
        width = bar.get_width()
        axes[1, 2].text(width/2, bar.get_y() + bar.get_height()/2, 
                       f'{width:.3f}', ha='center', va='center', fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Enhanced LIME visualization saved: {save_path}")
    
    plt.show()
    
    return {
        'prediction': pred_class,
        'confidence': confidence,
        'class_name': explainer.class_names[pred_class],
        'top_segments': important_segments
    }

def train_model(model, train_loader, val_loader, device, epochs=15):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=CONFIG['LEARNING_RATE'], weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    best_acc = 0.0
    train_accs = []
    val_accs = []
    
    print(f"Starting training on {device}")
    print(f"Training samples: {len(train_loader.dataset)}, Validation: {len(val_loader.dataset)}")
    print("-" * 60)
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_correct = 0
        train_total = 0
        
        for images, labels, _ in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
        
        # Validation phase
        model.eval()
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, labels, _ in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        train_acc = 100 * train_correct / train_total
        val_acc = 100 * val_correct / val_total
        
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        
        scheduler.step()
        
        print(f'Epoch [{epoch+1}/{epochs}] Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%')
        
        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_acc': best_acc,
            }, CONFIG['MODEL_SAVE_PATH'])
            print(f'✅ New best model saved! Accuracy: {best_acc:.2f}%')
    
    print(f"\n FINAL BEST MODEL ACCURACY: {best_acc:.2f}%")
    return best_acc

def main():
    print(" Enhanced EfficientNet Oral Cancer Detection with Annotated LIME")
    print(f"Device: {CONFIG['DEVICE']}")
    print("-" * 60)
    
    if not os.path.exists(CONFIG['CSV_PATH']):
        print(" CSV file not found!")
        return
    
    # Load and split data
    df = pd.read_csv(CONFIG['CSV_PATH'])
    print(f" Loaded {len(df)} samples")
    print(f"Class distribution:\n{df['label'].value_counts()}")
    
    # Split and save train/test CSVs
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])
    train_df.to_csv(CONFIG['TRAIN_CSV_PATH'], index=False)
    val_df.to_csv(CONFIG['TEST_CSV_PATH'], index=False)
    print(f" Train/test splits saved to {CONFIG['TRAIN_CSV_PATH']} and {CONFIG['TEST_CSV_PATH']}")
    
    # Get transforms and create datasets
    train_transform, val_transform = get_transforms()
    train_dataset = OralCancerDataset(train_df, CONFIG['DATA_DIR'], transform=train_transform)
    val_dataset = OralCancerDataset(val_df, CONFIG['DATA_DIR'], transform=val_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=CONFIG['BATCH_SIZE'], shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG['BATCH_SIZE'], shuffle=False, num_workers=4)
    
    # Initialize and train model
    model = EnhancedEfficientNet(num_classes=CONFIG['NUM_CLASSES']).to(CONFIG['DEVICE'])
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {total_params:,}")
    
    print("\n Starting training...")
    best_acc = train_model(model, train_loader, val_loader, CONFIG['DEVICE'], CONFIG['EPOCHS'])
    
    # Load best model for LIME analysis
    if os.path.exists(CONFIG['MODEL_SAVE_PATH']):
        checkpoint = torch.load(CONFIG['MODEL_SAVE_PATH'])
        model.load_state_dict(checkpoint['model_state_dict'])
        print(" Loaded best trained model for LIME analysis")
    
    model.eval()
    
    # Select sample images for enhanced LIME analysis
    sample_images = ['/home/user/Desktop/oral_cancer/gdc_oral_cancer_dataset/oral-cancer-detection/data/original/patient_15/image_1.jpeg','/home/user/Desktop/oral_cancer/gdc_oral_cancer_dataset/oral-cancer-detection/data/original/patient_142/image_1.jpeg','/home/user/Desktop/oral_cancer/gdc_oral_cancer_dataset/oral-cancer-detection/data/original/patient_166/image_2.jpeg','/home/user/Desktop/oral_cancer/gdc_oral_cancer_dataset/oral-cancer-detection/data/original/patient_179/image_1.jpeg','/home/user/Desktop/oral_cancer/gdc_oral_cancer_dataset/oral-cancer-detection/data/original/patient_210/image_1.jpeg']
    for i in range(min(3, len(val_df))):
        img_name = val_df.iloc[i]['image_path']
        img_path = os.path.join(CONFIG['DATA_DIR'], img_name)
        sample_images.append(img_path)
    
    # Perform enhanced LIME analysis
    if sample_images:
        print(f"\n🔍 Analyzing {len(sample_images)} images with Enhanced LIME...")
        for i, img_path in enumerate(sample_images):
            if os.path.exists(img_path):
                print(f"Processing image {i+1}: {os.path.basename(img_path)}")
                save_path = os.path.join(CONFIG['RESULTS_DIR'], f'enhanced_lime_analysis_{i+1}.png')
                result = create_annotated_lime_visualization(model, img_path, save_path)
                print(f"Result: {result['class_name']} (Confidence: {result['confidence']:.3f})")
            else:
                print(f"⚠️ Image not found: {img_path}")
    
    print(f"\n Analysis complete! Best Model Accuracy: {best_acc:.2f}%")
    print(f" Results saved in: {CONFIG['RESULTS_DIR']}")

if __name__ == "__main__":
    main()