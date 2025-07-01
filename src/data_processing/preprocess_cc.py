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

# --- Load Metadata ---
metadata_path = r"/home/user/Desktop/oral_cancer/gdc_oral_cancer_dataset/oral-cancer-detection/data/more-data-1.csv"
if not os.path.exists(metadata_path):
    print(f"Error: {metadata_path} not found.")
    exit(1)

try:
    df = pd.read_csv(metadata_path)
except pd.errors.EmptyDataError:
    print(f"Error: {metadata_path} is empty.")
    exit(1)

print(f"Initial rows in df: {len(df)}")
print("Columns in df:", df.columns.tolist())

# --- Clean Metadata ---
if not df.empty:
    df.dropna(subset=['Risk', 'Suspicion'], inplace=True)  # Updated column name
    print(f"Rows after dropna: {len(df)}")
    df.replace('-', None, inplace=True)

    # Updated required columns
    required_columns = [
        'No.', 'Risk', 'Suspicion', 'Gender', 'Study setting', 
        'Cigarette/Bidi', 'Smokeless tobacco', 'Areca nut', 'Alcohol', 
        'Able to open mouth', 'Lesion/patch'
    ]
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        print(f"Error: Missing columns: {missing_cols}")
        exit(1)

    # Assign patient_id directly from 'No.' (no extraction needed)
    df['patient_id'] = df['No.'].astype(int)
    print("Unique patient IDs:", sorted(df['patient_id'].unique()))

    # Filter valid rows
    df = df[(df['Risk'] != 0) | (df['Suspicion'] != 0)]
    print(f"Rows after filtering: {len(df)}")

    # --- Value Mappings ---
    # Updated mappings for capitalized values
    use_map = {'never': 0, 'past': 1, 'current': 2}
    for col in ['Cigarette/Bidi', 'Smokeless tobacco', 'Areca nut', 'Alcohol']:
        print(f"Unique values in {col}:", df[col].unique())
        df[col] = df[col].astype(str).str.lower().map(use_map).fillna(0)

    print("Unique values in Gender:", df['Gender'].unique())
    df['Gender'] = df['Gender'].astype(str).str.lower().map({'male': 1, 'female': 0}).fillna(0)

    print("Unique values in Study setting:", df['Study setting'].unique())
    df['Study setting'] = df['Study setting'].astype(str).map({
        'Hospital': 1, 
        'Field/primary care': 0
    }).fillna(0)

    print("Unique values in Risk:", df['Risk'].unique())
    df['Risk'] = df['Risk'].astype(str).str.lower().map({'low risk': 0, 'high risk': 1}).fillna(0)

    print("Unique values in Suspicion:", df['Suspicion'].unique())
    df['Suspicion'] = df['Suspicion'].astype(str).str.lower().map({
        'non suspicious': 0, 'non-suspicious': 0, 'suspicious': 1
    }).fillna(0)

    print("Unique values in Able to open mouth:", df['Able to open mouth'].unique())
    df['Able to open mouth'] = df['Able to open mouth'].astype(str).str.lower().map({'yes': 1, 'no': 0}).fillna(0)

    print("Unique values in Lesion/patch:", df['Lesion/patch'].unique())
    df['Lesion/patch'] = df['Lesion/patch'].astype(str).str.lower().map({'yes': 1, 'no': 0}).fillna(0)

# --- Store Metadata ---
patient_metadata_dict = {}
if not df.empty:
    for pid in df['patient_id'].unique():
        patient_metadata_dict[pid] = df[df['patient_id'] == pid].iloc[0].to_dict()

# --- Verify Image Directory ---
raw_base_dir = r"/home/user/Desktop/oral_cancer/gdc_oral_cancer_dataset/oral-cancer-detection/data/bm"
if not os.path.exists(raw_base_dir):
    print(f"Error: Directory {raw_base_dir} does not exist.")
    exit(1)

all_patient_dirs = [d for d in os.listdir(raw_base_dir) if os.path.isdir(os.path.join(raw_base_dir, d)) and d.lower().startswith('patient_')]
if not all_patient_dirs:
    print(f"Error: No patient directories (e.g., 'patient_1') found in {raw_base_dir}.")
    exit(1)

# Extract numeric IDs from "patient_1", "Patient_2", etc.
patient_ids = []
patient_dir_mapping = {}  # Map patient ID to actual directory name
for folder in all_patient_dirs:
    try:
        pid = int(folder.split('_')[-1])  # Splits "patient_1" → 1
        patient_ids.append(pid)
        patient_dir_mapping[pid] = folder  # Store the actual folder name
    except ValueError:
        print(f"Skipping invalid folder: {folder}")

patient_ids = sorted(patient_ids)
print(f"Found patient IDs: {patient_ids}")
print(f"Patient directory mapping: {dict(list(patient_dir_mapping.items())[:5])}...")  # Show first 5 mappings

# --- Check for Missing Patients ---
metadata_pids = set(df['patient_id'].unique())
raw_pids = set(patient_ids)
missing_pids = metadata_pids - raw_pids
if missing_pids:
    print(f"Warning: Patients in metadata but missing images: {missing_pids}")

# --- Diagnostic: Check sample directories ---
print("\n=== DIAGNOSTIC: Checking sample directories ===")
sample_dirs = list(patient_dir_mapping.values())[:3]  # Check first 3 directories
for sample_dir in sample_dirs:
    sample_path = os.path.join(raw_base_dir, sample_dir)
    if os.path.exists(sample_path):
        all_files = list(Path(sample_path).glob("*"))
        print(f"Directory: {sample_dir}")
        print(f"  Total files: {len(all_files)}")
        if all_files:
            file_extensions = [f.suffix.lower() for f in all_files if f.is_file()]
            unique_extensions = list(set(file_extensions))
            print(f"  File extensions: {unique_extensions}")
            print(f"  Sample files: {[f.name for f in all_files[:5]]}")  # Show first 5 files
        print()
print("=== END DIAGNOSTIC ===\n")

# --- Image Processing ---
def is_blurred(image, threshold=20):
    image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    variance = laplacian.var()
    
    # Debug output to understand blur distribution
    print(f"Blur score: {variance:.1f} (Threshold: {threshold})")
    
    return variance < threshold, variance

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((384, 384)),
    transforms.ToPILImage()
])

data = []
total_images_processed = 0

for pid in patient_ids:
    # Use the actual directory name instead of just the numeric ID
    actual_dir_name = patient_dir_mapping.get(pid)
    if not actual_dir_name:
        print(f"Warning: No directory mapping found for patient {pid}")
        continue
        
    raw_patient_dir = os.path.join(raw_base_dir, actual_dir_name)
    
    # Debug: Check if directory exists
    if not os.path.exists(raw_patient_dir):
        print(f"Warning: Directory does not exist: {raw_patient_dir}")
        continue
    
    # Debug: Check what files are actually in the directory
    all_files = list(Path(raw_patient_dir).glob("*"))
    if not all_files:
        print(f"Warning: Directory is empty for patient {pid}: {raw_patient_dir}")
        continue
    
    # Check for various image extensions
    image_patterns = [
        "*.[jJpP][pPnN][gG]",  # jpg, jpeg, png, JPG, JPEG, PNG
        "*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG",
        "*.bmp", "*.BMP", "*.tiff", "*.TIFF", "*.tif", "*.TIF"
    ]
    
    images = []
    for pattern in image_patterns:
        images.extend(list(Path(raw_patient_dir).glob(pattern)))
    
    # Remove duplicates
    images = list(set(images))
    
    if not images:
        # Show what files are actually there
        file_extensions = [f.suffix.lower() for f in all_files if f.is_file()]
        unique_extensions = list(set(file_extensions))
        print(f"Warning: No image files found for patient {pid} in {raw_patient_dir}")
        print(f"  Directory contains {len(all_files)} files with extensions: {unique_extensions}")
        if len(all_files) <= 10:  # Show filenames if not too many
            print(f"  Files: {[f.name for f in all_files]}")
        continue

    patient_metadata = patient_metadata_dict.get(pid, {
        'Suspicion': 0, 'Risk': 0, 'Gender': 0, 'Study setting': 0,
        'Cigarette/Bidi': 0, 'Smokeless tobacco': 0, 'Areca nut': 0, 'Alcohol': 0,
        'Able to open mouth': 0, 'Lesion/patch': 0
    })

    print(f"Processing patient {pid} with {len(images)} images...")

    for img_file in images:
        try:
            img = Image.open(img_file).convert('RGB')
            is_blur, blur_score = is_blurred(img)
            if is_blur:
                print(f"Skipping blurred image {img_file.name} (score: {blur_score})")
                continue

            img_transformed = transform(img)
            img_transformed.save(img_file, quality=95)

            data.append({
                'image_path': str(img_file),
                'label': patient_metadata['Suspicion'],  # Updated column name
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
            if total_images_processed % 10 == 0:  # Print progress every 10 images
                print(f"Progress: {total_images_processed} images processed")

        except Exception as e:
            print(f"Error processing {img_file}: {e}")
            continue

# --- Save Output ---
if not data:
    print("Error: No valid images processed.")
    print("Debug info:")
    print(f"- Found {len(patient_ids)} patient directories")
    print(f"- Patient IDs in metadata: {len(metadata_pids)}")
    print(f"- Sample patient directories: {list(patient_dir_mapping.values())[:5]}")
    exit(1)

processed_df = pd.DataFrame(data)
csv_path = r"/home/user/Desktop/oral_cancer/gdc_oral_cancer_dataset/oral-cancer-detection/data/processed_labels.csv"
processed_df.to_csv(csv_path, index=False)
print(f"Successfully processed {total_images_processed} images")
print(f"Saved processed data to {csv_path}")
print(f"Data shape: {processed_df.shape}")
print(f"Label distribution: {processed_df['label'].value_counts().to_dict()}")