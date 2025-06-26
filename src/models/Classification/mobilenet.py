import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.models import mobilenet_v2
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from torchvision.transforms.functional import to_pil_image


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

# Load the processed data
csv_path = r'/home/user/Desktop/oral_cancer/gdc_oral_cancer_dataset/oral-cancer-detection/data/processed_labels.csv'
try:
    df = pd.read_csv(csv_path)
    print(f"Loaded {csv_path} with {len(df)} rows.")
except FileNotFoundError:
    print(f"Error: {csv_path} not found.")
    exit()

# Verify patient_id and label columns exist
if 'patient_id' not in df.columns or 'label' not in df.columns:
    print("Error: 'patient_id' or 'label' column missing in the DataFrame.")
    exit()

patient_ids = df['patient_id'].unique()
print(f"Total unique patients: {len(patient_ids)}")

# Perform train-test split on patient IDs
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
train_csv_path = r'/home/user/Desktop/oral_cancer/gdc_oral_cancer_dataset/oral-cancer-detection/data/train_split.csv'
test_csv_path = r'/home/user/Desktop/oral_cancer/gdc_oral_cancer_dataset/oral-cancer-detection/data/test_split.csv'
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
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = mobilenet_v2(weights='MobileNet_V2_Weights.IMAGENET1K_V1') 
model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
best_acc = 0.0
num_epochs = 20
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
    print(f"Epoch [{epoch+1}/{num_epochs}] ✅ Training Loss: {epoch_loss:.4f}")

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
    print(f"Epoch [{epoch+1}/{num_epochs}] ✅ Test Accuracy: {acc:.4f}")

# Save the model state dict after training
model_save_path = r'/home/user/Desktop/oral_cancer/gdc_oral_cancer_dataset/oral-cancer-detection/models/mobilenetv2_oral_cancer.pth'
torch.save(model.state_dict(), model_save_path)
print(f"Model saved to {model_save_path}")

# ----------- Grad-CAM Function -------------

def generate_gradcam(model, image_tensor, class_index, target_layer):
    model.eval()
    image_tensor = image_tensor.unsqueeze(0).to(device)

    gradients = []
    activations = []

    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0].detach())

    def forward_hook(module, input, output):
        activations.append(output.detach())

    forward_handle = target_layer.register_forward_hook(forward_hook)
    backward_handle = target_layer.register_backward_hook(backward_hook)

    output = model(image_tensor)
    pred_class = torch.argmax(output)
    model.zero_grad()

    one_hot = torch.zeros_like(output)
    one_hot[0, class_index] = 1
    output.backward(gradient=one_hot)

    gradients_ = gradients[0]
    activations_ = activations[0]

    weights = torch.mean(gradients_, dim=(2, 3), keepdim=True)
    gradcam_map = torch.sum(weights * activations_, dim=1).squeeze()
    gradcam_map = torch.clamp(gradcam_map, min=0)
    gradcam_map /= gradcam_map.max()

    heatmap = gradcam_map.cpu().numpy()
    heatmap = cv2.resize(heatmap, (224, 224))
    heatmap = np.uint8(255 * heatmap)
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    original_image = image_tensor.squeeze().cpu()
    original_image = to_pil_image(original_image)
    original_np = np.array(original_image)
    overlay = cv2.addWeighted(original_np, 0.6, heatmap_color, 0.4, 0)

    plt.figure(figsize=(6, 3))
    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(original_image)
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title("Grad-CAM Overlay")
    plt.imshow(overlay)
    plt.axis('off')
    plt.tight_layout()
    plt.show()

    forward_handle.remove()
    backward_handle.remove()

# ----------- Run Grad-CAM on a Test Sample -------------
# Pick a sample from the test set
sample_image, sample_label = test_dataset[1]  # You can change index to see other images

# Use the last convolutional layer
target_layer = model.features[-1]

# Visualize Grad-CAM
generate_gradcam(model, sample_image, sample_label, target_layer)
