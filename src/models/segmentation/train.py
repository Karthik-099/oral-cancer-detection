import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from PIL import Image
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
import torch.optim as optim

# Custom Dataset (with pseudo-masks for now)
class SegmentationDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.df = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = self.df.iloc[idx]['image_path']  # Path to original image
        label = self.df.iloc[idx]['label']
        image = Image.open(img_path).convert('RGB')
        # Pseudo-mask: If label=1, assume the whole image is the lesion (for now)
        mask = Image.new('L', image.size, int(label * 255))  # 0 or 255
        if self.transform:
            image = self.transform(image)
            # Resize mask to match the transformed image size
            mask = transforms.Resize((512, 512))(mask)
            mask = transforms.ToTensor()(mask)
        return image, mask

# Transforms for U-Net (resize on-the-fly)
transform = transforms.Compose([
    transforms.Resize((512, 512)),  # Resize to 512x512 for U-Net
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load and split dataset
df = pd.read_csv('../../../data/processed_labels.csv')
train_df, temp_df = train_test_split(df, test_size=0.3, random_state=42)
val_df, test_df = train_test_split(temp_df, test_size=0.33, random_state=42)
train_df.to_csv('../../../data/train_labels.csv', index=False)
val_df.to_csv('../../../data/val_labels.csv', index=False)
test_df.to_csv('../../../data/test_labels.csv', index=False)

# Create datasets and loaders
train_dataset = SegmentationDataset('../../../data/train_labels.csv', transform=transform)
val_dataset = SegmentationDataset('../../../data/val_labels.csv', transform=transform)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

# Load model
model = smp.Unet(encoder_name='resnet34', encoder_weights='imagenet', in_channels=3, classes=1)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# Loss and optimizer
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop with validation
num_epochs = 10
for epoch in range(num_epochs):
    # Training phase
    model.train()
    train_loss = 0.0
    for images, masks in train_loader:
        images, masks = images.to(device), masks.to(device)
        outputs = model(images)
        loss = criterion(outputs, masks)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    avg_train_loss = train_loss / len(train_loader)
    
    # Validation phase
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for images, masks in val_loader:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            loss = criterion(outputs, masks)
            val_loss += loss.item()
    avg_val_loss = val_loss / len(val_loader)
    
    print(f'Epoch [{epoch+1}/{num_epochs}]')
    print(f'Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')

# Save the model
torch.save(model.state_dict(), '../../../models/unet_segmenter.pth')

# Evaluate on test set (using Dice coefficient for segmentation)
def dice_coefficient(pred, target, threshold=0.5):
    pred = torch.sigmoid(pred) > threshold  # Apply sigmoid and threshold
    pred = pred.float()
    target = target.float()
    intersection = (pred * target).sum()
    return (2. * intersection + 1e-6) / (pred.sum() + target.sum() + 1e-6)

test_dataset = SegmentationDataset('../../../data/test_labels.csv', transform=transform)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

model.eval()
dice_scores = []
with torch.no_grad():
    for images, masks in test_loader:
        images, masks = images.to(device), masks.to(device)
        outputs = model(images)
        dice = dice_coefficient(outputs, masks)
        dice_scores.append(dice.item())
avg_dice = sum(dice_scores) / len(dice_scores)
print(f'Test Dice Coefficient: {avg_dice:.4f}')