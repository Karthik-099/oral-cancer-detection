"""
Training Pipeline
Trains EfficientNet-B4 classifier from metadata CSV + annotation feedback
"""
import os
import json
import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score, accuracy_score
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from models.classifier import OralCancerClassifier, CLASS_NAMES


# ─── Dataset ─────────────────────────────────────────────────────────────────

class OralCancerDataset(Dataset):
    """
    Expects metadata CSV with columns:
      image_path, label, age, gender, tobacco_use, alcohol_use, family_history
    """
    LABEL_MAP = {name: i for i, name in enumerate(CLASS_NAMES)}

    def __init__(self, df, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = Image.open(row['image_path']).convert('RGB')
        if self.transform:
            image = self.transform(image)

        label = self.LABEL_MAP.get(row['label'], 0)

        metadata = torch.tensor([
            float(row.get('age', 40)) / 100.0,
            1.0 if str(row.get('gender', '')).lower() == 'male' else 0.0,
            float(row.get('tobacco_use', 0)),
            float(row.get('alcohol_use', 0)),
            float(row.get('family_history', 0)),
        ], dtype=torch.float32)

        return image, metadata, label


# ─── Transforms ──────────────────────────────────────────────────────────────

TRAIN_TRANSFORM = transforms.Compose([
    transforms.Resize((380, 380)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

VAL_TRANSFORM = transforms.Compose([
    transforms.Resize((380, 380)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


# ─── Trainer ─────────────────────────────────────────────────────────────────

class Trainer:
    def __init__(self, model, device, save_path, epochs=30, lr=1e-4, batch_size=16):
        self.model = model.to(device)
        self.device = device
        self.save_path = save_path
        self.epochs = epochs
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=epochs)
        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        self.batch_size = batch_size
        self.history = {'train_loss': [], 'val_loss': [], 'val_acc': [], 'val_f1': []}

    def train(self, train_df, val_df):
        train_ds = OralCancerDataset(train_df, TRAIN_TRANSFORM)
        val_ds = OralCancerDataset(val_df, VAL_TRANSFORM)
        train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True, num_workers=2)
        val_loader = DataLoader(val_ds, batch_size=self.batch_size, num_workers=2)

        best_f1 = 0.0
        for epoch in range(1, self.epochs + 1):
            train_loss = self._train_epoch(train_loader)
            val_loss, val_acc, val_f1 = self._val_epoch(val_loader)
            self.scheduler.step()

            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['val_f1'].append(val_f1)

            print(f"Epoch {epoch:03d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Acc: {val_acc:.4f} | F1: {val_f1:.4f}")

            if val_f1 > best_f1:
                best_f1 = val_f1
                torch.save(self.model.state_dict(), self.save_path)
                print(f"  ✅ Saved best model (F1={best_f1:.4f})")

        return self.history

    def _train_epoch(self, loader):
        self.model.train()
        total_loss = 0
        for images, metadata, labels in loader:
            images, metadata, labels = images.to(self.device), metadata.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            out = self.model(images, metadata)
            loss = self.criterion(out, labels)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
        return total_loss / len(loader)

    def _val_epoch(self, loader):
        self.model.eval()
        total_loss, all_preds, all_labels = 0, [], []
        with torch.no_grad():
            for images, metadata, labels in loader:
                images, metadata, labels = images.to(self.device), metadata.to(self.device), labels.to(self.device)
                out = self.model(images, metadata)
                loss = self.criterion(out, labels)
                total_loss += loss.item()
                preds = out.argmax(dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels.cpu().numpy())
        acc = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
        return total_loss / len(loader), acc, f1


# ─── Entry Point ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--metadata', default='data/metadata.csv')
    parser.add_argument('--save_path', default='../models/efficientnet_classifier.pth')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-4)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    df = pd.read_csv(args.metadata)
    train_df, val_df = train_test_split(df, test_size=0.2, stratify=df['label'], random_state=42)
    print(f"Train: {len(train_df)} | Val: {len(val_df)}")

    model = OralCancerClassifier()
    trainer = Trainer(model, device, args.save_path, args.epochs, args.lr, args.batch_size)
    history = trainer.train(train_df, val_df)

    with open('training_history.json', 'w') as f:
        json.dump(history, f, indent=2)
    print("✅ Training complete")


if __name__ == '__main__':
    main()
