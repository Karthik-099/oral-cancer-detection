import os
import cv2
import numpy as np
from pathlib import Path

os.makedirs('../data/processed/classification', exist_ok=True)
os.makedirs('../data/processed/segmentation', exist_ok=True)
for patient_folder in Path('../data/raw').glob('patient_*'):
    patient_id = patient_folder.name
    os.makedirs(f'../data/processed/classification/{patient_id}', exist_ok=True)
    os.makedirs(f'../data/processed/segmentation/{patient_id}', exist_ok=True)
    for img_path in patient_folder.glob('*.jpg'):
        img = cv2.imread(str(img_path))
        img_cls = cv2.resize(img, (224, 224))
        img_seg = cv2.resize(img, (512, 512))
        img_cls = img_cls / 255.0
        img_seg = img_seg / 255.0
        cv2.imwrite(f'../data/processed/classification/{patient_id}/{img_path.name}', img_cls * 255)
        cv2.imwrite(f'../data/processed/segmentation/{patient_id}/{img_path.name}', img_seg * 255)