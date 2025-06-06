#Libraries
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.models import resnet18, ResNet18_Weights
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import os


# function
class OralCancerDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = self.df.iloc[idx]['image_path']
        label = self.df.iloc[idx]['label']
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            
            image = Image.new('RGB', (224, 224), (0, 0, 0))
        if self.transform:
            image = self.transform(image)
        return image, label


# Transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Loading processed data
csv_path = r'C:\Users\jnupu\OneDrive\Desktop\nhilikha\oral-cancer-detection\data\processed_labels.csv'
try:
    df = pd.read_csv(csv_path)
    print(f"Loaded {csv_path} with {len(df)} rows.")
except FileNotFoundError:
    print(f"Error: {csv_path} not found.")
    exit()

# Verify exsistance patient_id and label columns
if 'patient_id' not in df.columns or 'label' not in df.columns:
    print("Error: 'patient_id' or 'label' column missing in the DataFrame.")
    exit()

patient_ids = df['patient_id'].unique()
print(f"Total unique patients: {len(patient_ids)}")

#train-test split on patient IDs
train_pids, test_pids = train_test_split(patient_ids, test_size=0.2, random_state=42)

# Create train and test DataFrames based on patient IDs
train_df = df[df['patient_id'].isin(train_pids)]
test_df = df[df['patient_id'].isin(test_pids)]

# Verify stratification by checking label distribution
train_label_dist = train_df.groupby('patient_id')['label'].first().value_counts()
test_label_dist = test_df.groupby('patient_id')['label'].first().value_counts()
print("Training set label distribution (per patient):\n", train_label_dist)
print("Test set label distribution (per patient):\n", test_label_dist)

# Save splits to CSV
train_csv_path = r'C:\Users\jnupu\OneDrive\Desktop\nhilikha\oral-cancer-detection\data\train_split.csv'
test_csv_path = r'C:\Users\jnupu\OneDrive\Desktop\nhilikha\oral-cancer-detection\data\test_split.csv'
train_df.to_csv(train_csv_path, index=False)
test_df.to_csv(test_csv_path, index=False)
print(f"Train split saved to {train_csv_path} ({len(train_df)} images, {len(train_df['patient_id'].unique())} patients)")
print(f"Test split saved to {test_csv_path} ({len(test_df)} images, {len(test_df['patient_id'].unique())} patients)")

# Create datasets
train_dataset = OralCancerDataset(train_df, transform=transform)
test_dataset = OralCancerDataset(test_df, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


# Load model
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# weights = ResNet18_Weights.IMAGENET1K_V1
# model = resnet18(weights=weights)
# model.fc = nn.Linear(model.fc.in_features, 2)
# model = model.to(device)
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.parameters(), lr=0.001)
from torchvision.models import resnet50, ResNet50_Weights
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
weights = ResNet50_Weights.IMAGENET1K_V1
model = resnet50(weights=weights)
model.fc = nn.Linear(model.fc.in_features, 2)
model = model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


best_acc = 0.0
num_epochs = 10
train_losses = []
test_accuracies = []

for epoch in range(num_epochs):
    # Training phase
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
    
    epoch_loss = running_loss / len(train_dataset)
    train_losses.append(epoch_loss)
    print(f"Epoch [{epoch+1}/{num_epochs}] Training Loss: {epoch_loss:.4f}")

    # Evaluation phase
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    acc = accuracy_score(y_true, y_pred)
    test_accuracies.append(acc)
    print(f"Epoch [{epoch+1}/{num_epochs}] Test Accuracy: {acc:.4f}")



