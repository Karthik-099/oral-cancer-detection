import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
import timm
import os
import numpy as np
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import pandas as pd
from PIL import Image
import warnings
import json
import time
from datetime import datetime
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import random
warnings.filterwarnings('ignore')

# Set seeds for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

# Improved Configuration
class Config:
    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Improved Hyperparameters
    BATCH_SIZE = 16  # Increased for more stable gradients
    LEARNING_RATE = 0.0001  # Slightly higher initial LR
    NUM_EPOCHS = 100  # More epochs with proper early stopping
    IMAGE_SIZE = 384  # Balanced size for EfficientNet-B4
    WEIGHT_DECAY = 1e-4  # Standard weight decay
    PATIENCE = 20  # More patience for convergence
    DROPOUT_RATE = 0.3  # Reduced dropout
    LABEL_SMOOTHING = 0.05  # Reduced label smoothing
    
    # Simplified training settings (removed complex augmentations initially)
    WARMUP_EPOCHS = 10
    GRADIENT_CLIPPING = 1.0
    
    # Paths - UPDATE THESE
    DATA_DIR = "/home/user/Desktop/oral_cancer/gdc_oral_cancer_dataset/oral-cancer-detection/data/bm"
    CSV_PATH = "/home/user/Desktop/oral_cancer/gdc_oral_cancer_dataset/oral-cancer-detection/data/processed_labels.csv"
    MODEL_SAVE_PATH = "/home/user/Desktop/oral_cancer/gdc_oral_cancer_dataset/oral-cancer-detection/models/improved_efficientnet_b4_oral_cancer.pth"
    RESULTS_DIR = "/home/user/Desktop/oral_cancer/gdc_oral_cancer_dataset/oral-cancer-detection/results"
    
    # Cross-validation
    N_FOLDS = 5
    CURRENT_FOLD = 0
    TEST_SIZE = 0.15
    RANDOM_STATE = 42

config = Config()

# Print device info
print(f"🔧 Using device: {config.device}")
if torch.cuda.is_available():
    print(f"   GPU: {torch.cuda.get_device_name(0)}")
    print(f"   CUDA Version: {torch.version.cuda}")

# Create directories
os.makedirs(os.path.dirname(config.MODEL_SAVE_PATH), exist_ok=True)
os.makedirs(config.RESULTS_DIR, exist_ok=True)

# Improved Data Augmentation - Start Simple
def get_improved_transforms():
    train_transform = A.Compose([
        A.Resize(config.IMAGE_SIZE, config.IMAGE_SIZE),
        # Basic augmentations first
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),
        A.RandomRotate90(p=0.3),
        # Color augmentations
        A.RandomBrightnessContrast(
            brightness_limit=0.2, 
            contrast_limit=0.2, 
            p=0.5
        ),
        A.HueSaturationValue(
            hue_shift_limit=10, 
            sat_shift_limit=20, 
            val_shift_limit=10, 
            p=0.3
        ),
        # Geometric augmentations
        A.ShiftScaleRotate(
            shift_limit=0.05, 
            scale_limit=0.1, 
            rotate_limit=15, 
            p=0.3
        ),
        # Noise and blur (mild)
        A.OneOf([
            A.GaussianBlur(blur_limit=3, p=1.0),
            A.GaussNoise(var_limit=(10.0, 50.0), p=1.0),
        ], p=0.2),
        # Normalization
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

    val_transform = A.Compose([
        A.Resize(config.IMAGE_SIZE, config.IMAGE_SIZE),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])
    
    return train_transform, val_transform

# Improved Dataset with better error handling
class ImprovedOralCancerDataset(Dataset):
    def __init__(self, root_dir, csv_path, indices=None, transform=None):
        self.df = pd.read_csv(csv_path)
        if indices is not None:
            self.df = self.df.iloc[indices].reset_index(drop=True)
        self.root_dir = root_dir
        self.transform = transform
        
        # Verify CSV structure
        required_cols = {'patient_id', 'image_path', 'label'}
        missing_cols = required_cols - set(self.df.columns)
        if missing_cols:
            raise ValueError(f"CSV missing required columns: {missing_cols}")
            
        # Verify labels are 0 and 1
        unique_labels = set(self.df['label'].unique())
        if not unique_labels.issubset({0, 1}):
            raise ValueError(f"Labels must be 0 or 1, found: {unique_labels}")
        
        # Verify images exist and filter out missing ones
        self._verify_images()
            
        print(f"📊 Dataset loaded: {len(self.df)} samples")
        print(f"   Suspicious (1): {(self.df['label'] == 1).sum()} samples")
        print(f"   Non-suspicious (0): {(self.df['label'] == 0).sum()} samples")

    def _verify_images(self):
        """Verify that images exist and filter out missing ones"""
        valid_indices = []
        missing_count = 0
        
        print("🔍 Verifying image paths...")
        for idx, row in tqdm(self.df.iterrows(), total=len(self.df), desc="Checking images"):
            img_path = self._get_image_path(row)
            if os.path.exists(img_path):
                valid_indices.append(idx)
            else:
                missing_count += 1
                if missing_count <= 5:  # Show first 5 missing files
                    print(f"❌ Missing: {img_path}")
        
        if missing_count > 0:
            print(f"⚠️  Found {missing_count} missing images, filtering them out")
            self.df = self.df.iloc[valid_indices].reset_index(drop=True)
    
    def _get_image_path(self, row):
        """Get the correct image path"""
        image_path = str(row['image_path'])
        
        # If image_name contains full path, use it directly
        if os.path.isabs(image_path) and os.path.exists(image_path):
            return image_path
        else:
            # Otherwise, construct path using root_dir and patient_id
            patient_id = str(row['patient_id'])
            image_filename = os.path.basename(image_path)
            return os.path.join(self.root_dir, patient_id, image_filename)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = self._get_image_path(row)
        
        try:
            # Load image with OpenCV
            image = cv2.imread(img_path)
            if image is None:
                # Try PIL as fallback
                image = Image.open(img_path).convert('RGB')
                image = np.array(image)
            else:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            label = int(row['label'])
            
            if self.transform:
                augmented = self.transform(image=image)
                image = augmented['image']
                
            return image, label
            
        except Exception as e:
            print(f"❌ Error loading {img_path}: {e}")
            # Create a dummy image instead of failing
            dummy_image = np.zeros((config.IMAGE_SIZE, config.IMAGE_SIZE, 3), dtype=np.uint8)
            if self.transform:
                augmented = self.transform(image=dummy_image)
                dummy_image = augmented['image']
            else:
                dummy_image = torch.zeros(3, config.IMAGE_SIZE, config.IMAGE_SIZE)
            return dummy_image, 0

# Alternative: Simple approach using timm's built-in classifier modification
class ImprovedEfficientNetClassifier(nn.Module):
    def __init__(self, num_classes=2, dropout_rate=0.3):
        super().__init__()
        # Create model with correct number of classes directly
        self.model = timm.create_model(
            'efficientnet_b4', 
            pretrained=True, 
            num_classes=num_classes,
            drop_rate=dropout_rate
        )
        
        print(f"🔧 Model created with {num_classes} output classes")
        
    def forward(self, x):
        return self.model(x)

# Simplified training function
def train_epoch_simple(model, train_loader, criterion, optimizer, epoch):
    """Simplified training without complex augmentations"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc=f'Training Epoch {epoch}')
    for batch_idx, (images, labels) in enumerate(pbar):
        images, labels = images.to(config.device), labels.to(config.device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRADIENT_CLIPPING)
        
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        if batch_idx % 10 == 0:  # Update less frequently
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100 * correct / total:.2f}%'
            })
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100 * correct / total
    
    return epoch_loss, epoch_acc

def validate_epoch_simple(model, val_loader, criterion):
    """Simplified validation"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_predicted = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc='Validation')
        for images, labels in pbar:
            images, labels = images.to(config.device), labels.to(config.device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            probs = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            all_predicted.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100 * correct / total:.2f}%'
            })
    
    epoch_loss = running_loss / len(val_loader)
    epoch_acc = 100 * correct / total
    
    # Calculate AUC if we have both classes
    try:
        auc_score = roc_auc_score(all_labels, np.array(all_probs)[:, 1])
    except:
        auc_score = 0.5  # If we can't calculate AUC, use 0.5
    
    return epoch_loss, epoch_acc, auc_score, all_predicted, all_labels

# Improved learning rate scheduler
class WarmupCosineScheduler:
    def __init__(self, optimizer, warmup_epochs, total_epochs, min_lr=1e-7):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr = min_lr
        self.base_lr = optimizer.param_groups[0]['lr']
        self.current_epoch = 0
    
    def step(self):
        if self.current_epoch < self.warmup_epochs:
            # Linear warmup
            lr = self.base_lr * (self.current_epoch + 1) / self.warmup_epochs
        else:
            # Cosine annealing
            progress = (self.current_epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            lr = self.min_lr + (self.base_lr - self.min_lr) * 0.5 * (1 + np.cos(np.pi * progress))
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        
        self.current_epoch += 1

# Improved training with K-fold
def train_with_improved_kfold():
    """Improved K-fold training"""
    # Load data
    df = pd.read_csv(config.CSV_PATH)
    
    # Check class distribution
    class_counts = df['label'].value_counts().sort_index()
    print(f"📊 Overall class distribution:")
    for label, count in class_counts.items():
        print(f"   Class {label}: {count} samples ({count/len(df)*100:.1f}%)")
    
    # Prepare for cross-validation
    kfold = StratifiedKFold(n_splits=config.N_FOLDS, shuffle=True, random_state=config.RANDOM_STATE)
    
    fold_results = []
    
    for fold, (train_idx, val_idx) in enumerate(kfold.split(df, df['label'])):
        print(f"\n🔄 Training Fold {fold + 1}/{config.N_FOLDS}")
        print("=" * 60)
        
        config.CURRENT_FOLD = fold
        
        # Create data loaders
        train_transform, val_transform = get_improved_transforms()
        
        train_dataset = ImprovedOralCancerDataset(config.DATA_DIR, config.CSV_PATH, train_idx, train_transform)
        val_dataset = ImprovedOralCancerDataset(config.DATA_DIR, config.CSV_PATH, val_idx, val_transform)
        
        # Skip fold if dataset is empty
        if len(train_dataset) == 0 or len(val_dataset) == 0:
            print(f"⚠️ Skipping fold {fold} due to empty dataset")
            continue
        
        # Calculate class weights more carefully
        train_labels = [train_dataset[i][1] for i in range(len(train_dataset))]
        class_counts = Counter(train_labels)
        
        # Ensure we have both classes
        if len(class_counts) < 2:
            print(f"⚠️ Skipping fold {fold} due to missing class")
            continue
        
        total_samples = len(train_labels)
        class_weights = {
            0: total_samples / (2 * class_counts[0]) if class_counts[0] > 0 else 1.0,
            1: total_samples / (2 * class_counts[1]) if class_counts[1] > 0 else 1.0
        }
        
        print(f"📊 Fold {fold + 1} class distribution:")
        print(f"   Train - Class 0: {class_counts[0]}, Class 1: {class_counts[1]}")
        print(f"   Weights - Class 0: {class_weights[0]:.3f}, Class 1: {class_weights[1]:.3f}")
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.BATCH_SIZE,
            shuffle=True,
            num_workers=2,  # Reduced for stability
            pin_memory=True if config.device.type == 'cuda' else False,
            drop_last=True  # Ensure consistent batch sizes
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=config.BATCH_SIZE,
            shuffle=False,
            num_workers=2,
            pin_memory=True if config.device.type == 'cuda' else False
        )
        
        # Initialize model
        model = ImprovedEfficientNetClassifier(num_classes=2, dropout_rate=config.DROPOUT_RATE).to(config.device)
        
        # Loss function and optimizer
        class_weights_tensor = torch.tensor([class_weights[0], class_weights[1]]).to(config.device)
        criterion = nn.CrossEntropyLoss(weight=class_weights_tensor, label_smoothing=config.LABEL_SMOOTHING)
        
        optimizer = optim.AdamW(
            model.parameters(), 
            lr=config.LEARNING_RATE, 
            weight_decay=config.WEIGHT_DECAY,
            betas=(0.9, 0.999)
        )
        
        # Improved scheduler
        scheduler = WarmupCosineScheduler(optimizer, config.WARMUP_EPOCHS, config.NUM_EPOCHS)
        
        # Training loop
        best_val_acc = 0.0
        best_auc = 0.0
        patience_counter = 0
        
        train_losses, val_losses = [], []
        train_accs, val_accs = [], []
        auc_scores = []
        
        for epoch in range(config.NUM_EPOCHS):
            # Update learning rate
            scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']
            
            # Train
            train_loss, train_acc = train_epoch_simple(model, train_loader, criterion, optimizer, epoch)
            
            # Validate
            val_loss, val_acc, auc_score, val_pred, val_true = validate_epoch_simple(model, val_loader, criterion)
            
            # Store metrics
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            train_accs.append(train_acc)
            val_accs.append(val_acc)
            auc_scores.append(auc_score)
            
            print(f"\nEpoch {epoch+1}/{config.NUM_EPOCHS} (LR: {current_lr:.6f})")
            print(f"Train - Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%")
            print(f"Val - Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%, AUC: {auc_score:.4f}")
            
            # Save best model based on AUC (more reliable for imbalanced data)
            if auc_score > best_auc:
                best_val_acc = val_acc
                best_auc = auc_score
                patience_counter = 0
                
                # Save model
                model_path = config.MODEL_SAVE_PATH.replace('.pth', f'_fold_{fold}.pth')
                torch.save({
                    'fold': fold,
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_val_acc': best_val_acc,
                    'best_auc': best_auc,
                    'config': config.__dict__
                }, model_path)
                
                print(f" New best model saved! Val Acc: {best_val_acc:.2f}%, AUC: {best_auc:.4f}")
            else:
                patience_counter += 1
                print(f"⏳ Patience: {patience_counter}/{config.PATIENCE}")
            
            # Early stopping
            if patience_counter >= config.PATIENCE:
                print(f" Early stopping triggered after {epoch+1} epochs")
                break
        
        fold_results.append({
            'fold': fold,
            'best_val_acc': best_val_acc,
            'best_auc': best_auc,
            'train_losses': train_losses,
            'val_losses': val_losses,
            'train_accs': train_accs,
            'val_accs': val_accs,
            'auc_scores': auc_scores
        })
        
        print(f"\n Fold {fold + 1} Results:")
        print(f"   Best Val Acc: {best_val_acc:.2f}%")
        print(f"   Best AUC: {best_auc:.4f}")
    
    # Calculate average performance
    if fold_results:
        avg_acc = np.mean([result['best_val_acc'] for result in fold_results])
        avg_auc = np.mean([result['best_auc'] for result in fold_results])
        std_acc = np.std([result['best_val_acc'] for result in fold_results])
        std_auc = np.std([result['best_auc'] for result in fold_results])
        
        print(f"\n📊 Cross-Validation Results:")
        print(f"   Average Validation Accuracy: {avg_acc:.2f}% ± {std_acc:.2f}%")
        print(f"   Average AUC: {avg_auc:.4f} ± {std_auc:.4f}")
        
        # Save results
        with open(os.path.join(config.RESULTS_DIR, 'improved_kfold_results.json'), 'w') as f:
            json.dump(fold_results, f, indent=2)
    else:
        print("❌ No successful folds completed!")
    
    return fold_results

def main():
    """Main training function"""
    print("🚀 Starting Improved EfficientNet-B4 Oral Cancer Detection Training")
    print(f"📁 Data directory: {config.DATA_DIR}")
    print(f"📄 CSV file: {config.CSV_PATH}")
    
    # Verify paths exist
    if not os.path.exists(config.DATA_DIR):
        print(f"❌ Data directory does not exist: {config.DATA_DIR}")
        return None
    
    if not os.path.exists(config.CSV_PATH):
        print(f"❌ CSV file does not exist: {config.CSV_PATH}")
        return None
    
    # Train with improved K-fold cross-validation
    results = train_with_improved_kfold()
    
    return results

if __name__ == "__main__":
    try:
        results = main()
        if results:
            print("\n✅ Training complete!")
        else:
            print("\n❌ Training failed!")
    except Exception as e:
        print(f"❌ Error during training: {e}")
        import traceback
        traceback.print_exc()