#This is the file which has the metrices for blur as well as the removal of reflection.

import pandas as pd
import os
from pathlib import Path
from PIL import Image
import torch
from torchvision import transforms
import cv2
import numpy as np
import warnings

warnings.filterwarnings("ignore")
pd.options.display.max_rows = 1000
pd.options.display.max_columns = 1000

print("Current working directory:", os.getcwd())

# Load metadata
metadata_path = r"C:\Users\jnupu\OneDrive\Desktop\nhilikha\oral-cancer-detection\data\metadata.csv"
if not os.path.exists(metadata_path):
    print(f"Error: {metadata_path} not found. Please create a valid metadata.csv file.")
    exit(1)

try:
    df = pd.read_csv(metadata_path)
except pd.errors.EmptyDataError:
    print(f"Error: {metadata_path} is empty. Please provide a valid metadata file with data.")
    exit(1)

print(f"Initial number of rows in df: {len(df)}")
print("Columns in df:", df.columns.tolist())

if not df.empty:
    df.dropna(subset=['Risk', 'Specialist diagnosis'], inplace=True)
    print(f"Number of rows after dropna: {len(df)}")
    df.replace('-', None, inplace=True)
    required_columns = ['FLW username', 'Risk', 'Specialist diagnosis', 'Gender', 'Study setting', 
                       'Cigarette/Bidi', 'Smokeless tobacco', 'Areca nut', 'Alcohol', 
                       'Able to open mouth', 'Lesion/patch']
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        print(f"Error: Missing required columns in metadata.csv: {missing_cols}")
        exit(1)

    df['patient_id'] = df['FLW username'].apply(lambda x: int(str(x).split()[1]) if pd.notna(x) else -1)
    print("Unique patient IDs in metadata.csv:", sorted(df['patient_id'].unique()))
    df = df[(df['Risk'] != 0) | (df['Specialist diagnosis'] != 0)]
    print(f"Number of rows in df after filtering: {len(df)}")

    use_map = {'never': 0, 'past': 1, 'current': 2}
    for col in ['Cigarette/Bidi', 'Smokeless tobacco', 'Areca nut', 'Alcohol']:
        print(f"Unique values in {col}:", df[col].unique())
        df[col] = df[col].astype(str).str.lower().map(use_map).fillna(0)

    print("Unique values in Gender:", df['Gender'].unique())
    df['Gender'] = df['Gender'].astype(str).str.lower().map({'male': 1, 'female': 0}).fillna(0)

    print("Unique values in Study setting:", df['Study setting'].unique())
    df['Study setting'] = df['Study setting'].astype(str).str.lower().map({'hospital': 1, 'primary care': 0, 'field/primary care': 0}).fillna(0)

    print("Unique values in Risk:", df['Risk'].unique())
    df['Risk'] = df['Risk'].astype(str).str.lower().map({'low risk': 0, 'high risk': 1}).fillna(0)

    print("Unique values in Specialist diagnosis:", df['Specialist diagnosis'].unique())
    df['Specialist diagnosis'] = df['Specialist diagnosis'].astype(str).str.lower().map({
        'non suspicious': 0, 'non-suspicious': 0, 'suspicious': 1
    }).fillna(0)

    print("Unique values in Able to open mouth:", df['Able to open mouth'].unique())
    df['Able to open mouth'] = df['Able to open mouth'].astype(str).str.lower().map({'yes': 1, 'no': 0}).fillna(0)
    print("Unique values in Lesion/patch:", df['Lesion/patch'].unique())
    df['Lesion/patch'] = df['Lesion/patch'].astype(str).str.lower().map({'yes': 1, 'no': 0}).fillna(0)

patient_metadata_dict = {}
if not df.empty:
    for pid in df['patient_id'].unique():
        if pid != -1:
            patient_metadata_dict[pid] = df[df['patient_id'] == pid].iloc[0].to_dict()

# Verify raw directory
raw_base_dir = r"C:\Users\jnupu\OneDrive\Desktop\nhilikha\oral-cancer-detection\data\raw"
if not os.path.exists(raw_base_dir):
    print(f"Error: Raw directory {raw_base_dir} does not exist.")
    exit(1)

# Modified to detect directories named '1', '2', '3', etc.
all_patient_dirs = [d for d in os.listdir(raw_base_dir) if os.path.isdir(os.path.join(raw_base_dir, d)) and d.isdigit()]
if not all_patient_dirs:
    print(f"Error: No patient directories (e.g., '1', '2', '3') found in {raw_base_dir}.")
    print(f"Contents of {raw_base_dir}: {os.listdir(raw_base_dir)}")
    exit(1)

patient_ids = sorted([int(d) for d in all_patient_dirs])
print(f"All patient IDs found in raw directory: {patient_ids}")

# Check for mismatch between metadata and directory patient IDs
metadata_pids = set(df['patient_id'].unique()) - {-1}
raw_pids = set(patient_ids)
missing_pids = metadata_pids - raw_pids
if missing_pids:
    print(f"Warning: Patient IDs in metadata.csv but not in raw directory: {missing_pids}")

# Collect images
image_paths_per_patient = {}
total_images = 0
for pid in patient_ids:
    raw_patient_dir = os.path.join(raw_base_dir, str(pid))  # Use directory name as the patient ID (e.g., '1')
    images = []
    for ext in ['*.jpg', '*.JPG', '*.jpeg', '*.JPEG', '*.png', '*.PNG']:
        images.extend(Path(raw_patient_dir).glob(ext))
    images = list(set(images))
    image_paths_per_patient[pid] = images
    total_images += len(images)
    if not images:
        print(f"Warning: No изображения found for patient {pid} in {raw_patient_dir}")
    else:
        print(f"Patient {pid} has {len(images)} images: {[img.name for img in images]}")

print(f"Total unique images to process: {total_images}")
if total_images == 0:
    print("Error: No images found in any patient directories. Cannot proceed.")
    exit(1)

# Blur detection
def is_blurred(image, threshold=100):
    image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    variance = laplacian.var()
    return variance < threshold, variance

# Reflection removal
def remove_reflections(image):
    image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)
    image_inpaint = cv2.inpaint(image_cv, mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
    return Image.fromarray(cv2.cvtColor(image_inpaint, cv2.COLOR_BGR2RGB))

# Transform pipeline
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((224, 224)),
    transforms.ToPILImage()
])

# Process images
data = []
total_images_processed = 0

for pid in patient_ids:
    raw_patient_dir = os.path.join(raw_base_dir, str(pid))
    print(f"Processing patient_id: {pid}, Raw directory: {raw_patient_dir}")
    
    if not os.path.exists(raw_patient_dir):
        print(f"Directory {raw_patient_dir} does not exist, skipping.")
        continue
    
    images = image_paths_per_patient.get(pid, [])
    if not images:
        print(f"No images found for patient {pid} in {raw_patient_dir}")
        continue
    
    patient_metadata = patient_metadata_dict.get(pid, {
        'Specialist diagnosis': 0, 'Risk': 0, 'Gender': 0, 'Study setting': 0,
        'Cigarette/Bidi': 0, 'Smokeless tobacco': 0, 'Areca nut': 0, 'Alcohol': 0,
        'Able to open mouth': 0, 'Lesion/patch': 0
    })
    
    for img_file in images:
        try:
            img = Image.open(img_file).convert('RGB')
            img = remove_reflections(img)  # Remove reflections
            is_blur, blur_score = is_blurred(img, threshold=100)
            if is_blur:
                print(f"Skipping blurred image {img_file.name} for patient {pid} (Laplacian variance: {blur_score})")
                continue
            
            img_transformed = transform(img)
            img_transformed.save(img_file, quality=95)
            
            data.append({
                'image_path': str(img_file),
                'label': patient_metadata['Specialist diagnosis'],
                'patient_id': pid,
                'risk': patient_metadata['Risk'],
                'gender': patient_metadata['Gender'],
                'study_setting': patient_metadata['Study setting'],
                'cigarette_bidi': patient_metadata['Cigarette/Bidi'],
                'smokeless_tobacco': patient_metadata['Smokeless tobacco'],
                'areca_nut': patient_metadata['Areca nut'],
                'alcohol': patient_metadata['Alcohol'],
                'able_to_open_mouth': patient_metadata['Able to open mouth'],
                'lesion_patch': patient_metadata['Lesion/patch'],
                'blur_score': blur_score
            })
            
            total_images_processed += 1
            print(f"Processed image {img_file.name} for patient {pid} (Blur score: {blur_score}, {total_images_processed}/{total_images} images done)")
            
        except Exception as e:
            print(f"Error processing image {img_file}: {e}")
            continue

if not data:
    print("Error: No valid images were processed. Check image files, directories, or blur threshold.")
    exit(1)

processed_df = pd.DataFrame(data)
print("Processed DataFrame shape:", processed_df.shape)
print("Processed DataFrame columns:", processed_df.columns.tolist())
print(processed_df.head())

csv_path = r"C:\Users\jnupu\OneDrive\Desktop\nhilikha\oral-cancer-detection\data\processed_labels.csv"
processed_df.to_csv(csv_path, index=False)
print(f"Preprocessing complete. Processed data saved to '{csv_path}'.")
