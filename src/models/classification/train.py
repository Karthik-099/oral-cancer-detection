import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from torchvision.models import mobilenet_v2
from torchvision.models.mobilenetv2 import MobileNet_V2_Weights
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split  # Fixed import
import os

# Custom Dataset
class OralCancerDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.df = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = self.df.iloc[idx]['image_path']  # Path to original image
        label = self.df.iloc[idx]['label']
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label

# Transforms for MobileNet (resize on-the-fly)
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to 224x224 for MobileNet
    transforms.RandomHorizontalFlip(),  # Add augmentation
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
train_dataset = OralCancerDataset('../../../data/train_labels.csv', transform=transform)
val_dataset = OralCancerDataset('../../../data/val_labels.csv', transform=transform)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)  # Reduced batch size
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)  # Reduced batch size

# Load model
model = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)  # Updated to use weights
model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)  # Binary classification
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-4)  # Reduced lr, added weight decay

# Training loop with validation
num_epochs = 10
for epoch in range(num_epochs):
    # Training phase
    model.train()
    train_loss = 0.0
    train_batches = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        train_batches += 1
    avg_train_loss = train_loss / train_batches if train_batches > 0 else 0
    
    # Validation phase
    model.eval()
    val_loss = 0.0
    val_batches = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            val_batches += 1
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    avg_val_loss = val_loss / val_batches if val_batches > 0 else 0
    val_accuracy = 100 * correct / total if total > 0 else 0
    
    print(f'Epoch [{epoch+1}/{num_epochs}]')
    print(f'Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%')

# Create the models directory if it doesn't exist
model_dir = '../../../models'
os.makedirs(model_dir, exist_ok=True)

# Save the model
model_path = os.path.join(model_dir, 'mobilenet_classifier.pth')
torch.save(model.state_dict(), model_path)
print(f"Model saved to {model_path}")

# Evaluate on test set
test_dataset = OralCancerDataset('../../../data/test_labels.csv', transform=transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

model.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
test_accuracy = 100 * correct / total
print(f'Test Accuracy: {test_accuracy:.2f}%')