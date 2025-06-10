import pandas as pd
import os
from pathlib import Path
from PIL import Image
import torch
from torchvision import transforms
import warnings

warnings.filterwarnings("ignore")

pd.options.display.max_rows = 1000
pd.options.display.max_columns = 1000

print("Current working directory:", os.getcwd())

try:
    df = pd.read_csv('/home/user/Desktop/oral_cancer/gdc_oral_cancer_dataset/oral-cancer-detection/data/metadata.csv')
except FileNotFoundError:
    print("Error: metadata.csv not found. Creating an empty DataFrame.")
    df = pd.DataFrame()

print(f"Initial number of rows in df: {len(df)}")
print("Columns in df:", df.columns.tolist())

if not df.empty:
    df.dropna(subset=['Risk', 'Specialist diagnosis'], inplace=True)
    print(f"Number of rows after dropna: {len(df)}")

    df.replace('-', 0, inplace=True)

    # Process FLW username: Extract the number (e.g., "FLW 123" -> 123)
    df['patient_id'] = df['No.'].astype(int)
    print("Unique patient IDs in metadata.csv:", sorted(df['patient_id'].unique()))

    # Map tobacco/alcohol use: 'never'->0, 'past'->1, 'current'->2
    use_map = {'never': 0, 'past': 1, 'current': 2}
    for col in ['Cigarette/Bidi', 'Smokeless tobacco', 'Areca nut', 'Alcohol']:
        print(f"Unique values in {col}:", df[col].unique())
        df[col] = df[col].astype(str).str.lower().map(use_map).fillna(0)

    # Encode gender: 'male'->1, 'female'->0
    print("Unique values in Gender:", df['Gender'].unique())
    df['Gender'] = df['Gender'].astype(str).str.lower().map({'male': 1, 'female': 0})

    # Encode study setting: 'hospital'->1, 'primary care'->0
    print("Unique values in Study setting:", df['Study setting'].unique())
    df['Study setting'] = df['Study setting'].astype(str).str.lower().map({'hospital': 1, 'primary care': 0})

    # Encode Risk: 'low risk'->0, 'high risk'->1
    print("Unique values in Risk:", df['Risk'].unique())
    df['Risk'] = df['Risk'].astype(str).str.lower().map({'low risk': 0, 'high risk': 1})

    # Encode Specialist diagnosis: 'non suspicious'->0, 'suspicious'->1
    print("Unique values in Specialist diagnosis:", df['Specialist diagnosis'].unique())
    df['Specialist diagnosis'] = df['Specialist diagnosis'].astype(str).str.lower().map({
        'non suspicious': 0,
        'non-suspicious': 0,
        'suspicious': 1
    })

    # Encode 'yes' -> 1 and 'no' -> 0
    print("Unique values in Able to open mouth:", df['Able to open mouth'].unique())
    df['Able to open mouth'] = df['Able to open mouth'].astype(str).str.lower().map({'yes': 1, 'no': 0}).fillna(0)
    print("Unique values in Lesion/patch:", df['Lesion/patch'].unique())
    df['Lesion/patch'] = df['Lesion/patch'].astype(str).str.lower().map({'yes': 1, 'no': 0}).fillna(0)

# Create a dictionary of patient metadata for quick lookup
patient_metadata_dict = {}
if not df.empty:
    for pid in df['patient_id'].unique():
        patient_metadata_dict[pid] = df[df['patient_id'] == pid].iloc[0].to_dict()

# Find all patient directories in raw/
raw_base_dir = '/home/user/Desktop/oral_cancer/gdc_oral_cancer_dataset/oral-cancer-detection/data/original'
all_patient_dirs = [d for d in os.listdir(raw_base_dir) if os.path.isdir(os.path.join(raw_base_dir, d)) and d.startswith('patient_')]
patient_ids = sorted([int(d.split('_')[1]) for d in all_patient_dirs])
print(f"All patient IDs found in raw directory: {patient_ids}")

# Calculate total unique images across all patient directories
total_images = 0
image_paths_per_patient = {}

for pid in patient_ids:
    raw_patient_dir = os.path.join(raw_base_dir, f'patient_{pid}')
    images = []
    images.extend(Path(raw_patient_dir).glob('*.jpg'))
    images.extend(Path(raw_patient_dir).glob('*.JPG'))
    images.extend(Path(raw_patient_dir).glob('*.jpeg'))
    images.extend(Path(raw_patient_dir).glob('*.JPEG'))
    images.extend(Path(raw_patient_dir).glob('*.png'))
    images.extend(Path(raw_patient_dir).glob('*.PNG'))
    # Remove duplicates
    images = list(set(images))
    image_paths_per_patient[pid] = images
    total_images += len(images)
    print(f"Patient {pid} has {len(images)} unique images")

print(f"Total unique images to process: {total_images}")

# Define a transform pipeline for in-place modification
transform = transforms.Compose([
    transforms.ToTensor(),  # Convert PIL Image to tensor
    transforms.Resize((300, 300)),  # Resize to 300x300 for EfficientNet-B3
    # transforms.RandomHorizontalFlip(p=0.5),  # Example augmentation
    # transforms.RandomRotation(degrees=10),  # Example augmentation
    # transforms.ColorJitter(brightness=0.2, contrast=0.2),  # Example augmentation
    transforms.ToPILImage()  # Convert back to PIL Image for saving
])

# Define cancer type mapping
cancer_type_map = {
    'homogenous oral leukoplakia': 1,
    'smokeless tobacco keratosis': 2,
    'squamous cell carcinoma of oral mucous membrane': 3,
    'proliferative verrucous oral leukoplakia': 4,
    'erythroplakia': 5,
    'oral lichen planus': 6,
    'speckled oral leukoplakia': 7,
    'oral cavity normal': 8,
    'benign': 9,
    'other': 10
}

# Map images to patients and modify in place
data = []
total_images_processed = 0

for pid in patient_ids:
    raw_patient_dir = os.path.join(raw_base_dir, f'patient_{pid}')
    print(f"Processing patient_id: {pid}, Raw directory: {raw_patient_dir}")
    
    # Check if the raw patient directory exists
    if not os.path.exists(raw_patient_dir):
        print(f"Directory {raw_patient_dir} does not exist, skipping.")
        continue
    
    # Debug: List all files in the directory
    all_files = list(Path(raw_patient_dir).glob('*'))
    print(f"All files in {raw_patient_dir}: {[f.name for f in all_files]}")
    
    # Get unique images for this patient
    images = image_paths_per_patient.get(pid, [])
    print(f"Found {len(images)} unique images in {raw_patient_dir}: {[img.name for img in images]}")
    
    # Get metadata for this patient (use defaults if not in metadata.csv)
    if pid in patient_metadata_dict:
        patient_metadata = patient_metadata_dict[pid]
    else:
        print(f"Warning: No metadata found for patient {pid}. Using default values.")
        patient_metadata = {
            'Specialist diagnosis': 0,
            'Risk': 0,
            'Gender': 0,
            'Study setting': 0,
            'Cigarette/Bidi': 0,
            'Smokeless tobacco': 0,
            'Areca nut': 0,
            'Alcohol': 0,
            'Able to open mouth': 0,
            'Lesion/patch': 0,
            'Provisional diagnosis': 'oral cavity normal'
        }
    
    # Determine cancer label and type
    risk = patient_metadata['Risk']
    specialist_diagnosis = patient_metadata['Specialist diagnosis']
    provisional_diagnosis = str(patient_metadata.get('Provisional diagnosis', 'oral cavity normal'))  # Cast to string
    
    # Debug: Print provisional diagnosis before processing
    print(f"Patient {pid}: Provisional diagnosis (raw) = {provisional_diagnosis}")
    
    provisional_diagnosis = provisional_diagnosis.lower()
    
    # Cancer label: 0 for (non suspicious, low risk), 1 for others
    cancer_label = 0 if risk == 0 and specialist_diagnosis == 0 else 1
    
    # Cancer type: Map provisional diagnosis to numerical value
    cancer_type = cancer_type_map.get(provisional_diagnosis, 8)  # Default to 'oral cavity normal' (8) if not found
    
    # Debug: Print patient metadata and labels
    print(f"Patient {pid}: Specialist diagnosis={specialist_diagnosis} ({'suspicious' if specialist_diagnosis == 1 else 'non suspicious'}), "
          f"Risk={risk} ({'high risk' if risk == 1 else 'low risk'}), "
          f"Provisional diagnosis={provisional_diagnosis}, "
          f"cancer_label={cancer_label}, cancer_type={cancer_type}")
    
    for img_file in images:
        try:
            # Open the image
            img = Image.open(img_file).convert('RGB')
            
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
                'cancer_label': cancer_label,
                'cancer_type': cancer_type
            })
            
            total_images_processed += 1
            print(f"Processed image {img_file.name} for patient {pid} ({total_images_processed}/{total_images} images done)")
            
        except Exception as e:
            print(f"Error processing image {img_file}: {e}")
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
                'cancer_label': cancer_label,
                'cancer_type': cancer_type
            })

processed_df = pd.DataFrame(data)
print("Processed DataFrame shape:", processed_df.shape)
print(processed_df.head())

csv_path = '/home/user/Desktop/oral_cancer/gdc_oral_cancer_dataset/oral-cancer-detection/data/processed_labels.csv'
processed_df.to_csv(csv_path, index=False)

print(f"Preprocessing complete. Processed data saved to '{csv_path}'.")
