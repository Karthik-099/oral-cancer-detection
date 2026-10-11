"""
Retrain Pipeline
Ingests doctor annotation feedback (JSON coordinates + corrected labels)
and fine-tunes the classifier on new data
"""
import os
import json
import glob
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image, ImageDraw
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from models.classifier import OralCancerClassifier, CLASS_NAMES, load_classifier
from training.train import OralCancerDataset, TRAIN_TRANSFORM, VAL_TRANSFORM, Trainer


ANNOTATION_DIR = os.path.join(os.path.dirname(__file__), '..', 'static', 'annotations')
UPLOAD_DIR = os.path.join(os.path.dirname(__file__), '..', 'uploads')
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', '..', 'models', 'efficientnet_classifier.pth')


def load_feedback_annotations():
    """Load all doctor feedback JSON files and build a DataFrame"""
    records = []
    for json_file in glob.glob(os.path.join(ANNOTATION_DIR, '*.json')):
        with open(json_file) as f:
            data = json.load(f)

        image_path = os.path.join(UPLOAD_DIR, data.get('image_filename', ''))
        if not os.path.exists(image_path):
            continue

        corrected_label = data.get('corrected_label') or data.get('ai_prediction', 'Normal')
        records.append({
            'image_path': image_path,
            'label': corrected_label,
            'age': data.get('patient_age', 40),
            'gender': data.get('patient_gender', 'Unknown'),
            'tobacco_use': 0,
            'alcohol_use': 0,
            'family_history': 0,
        })

    return pd.DataFrame(records)


def retrain_from_feedback(epochs=10, lr=5e-5):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    df = load_feedback_annotations()

    if len(df) < 4:
        print(f"⚠️  Only {len(df)} feedback samples — need at least 4 to retrain")
        return {'status': 'skipped', 'reason': 'insufficient_data', 'samples': len(df)}

    split = max(1, int(len(df) * 0.2))
    val_df = df.sample(n=split, random_state=42)
    train_df = df.drop(val_df.index)

    model = load_classifier(MODEL_PATH, device)
    model.train()

    trainer = Trainer(model, device, MODEL_PATH, epochs=epochs, lr=lr, batch_size=8)
    history = trainer.train(train_df, val_df)

    result = {
        'status': 'success',
        'samples_used': len(df),
        'epochs': epochs,
        'final_val_acc': history['val_acc'][-1] if history['val_acc'] else 0,
        'final_val_f1': history['val_f1'][-1] if history['val_f1'] else 0,
    }
    print(f"✅ Retrain complete: {result}")
    return result


if __name__ == '__main__':
    retrain_from_feedback()
