import pandas as pd
import os
from pathlib import Path
from PIL import Image

# Set display options for better visibility
pd.options.display.max_rows = 1000
pd.options.display.max_columns = 1000

# Read the CSV file
df = pd.read_csv('../../data/metadata.csv')  # Updated path

# Drop rows with NaNs in key columns
df.dropna(subset=['Risk', 'Specialist diagnosis'], inplace=True)

# Replace '-' with 0
df.replace('-', 0, inplace=True)

# Process FLW username: Extract the number (e.g., "FLW 123" -> 123)
df['patient_id'] = df['FLW username'].apply(lambda x: int(str(x).split()[1]))

# Remove rows where both Risk and Specialist diagnosis are 0
df = df[(df['Risk'] != 0) | (df['Specialist diagnosis'] != 0)]

# Map tobacco/alcohol use: 'never'->0, 'past'->1, 'current'->2
use_map = {'never': 0, 'past': 1, 'current': 2}
for col in ['Cigarette/Bidi', 'Smokeless tobacco', 'Areca nut', 'Alcohol']:
    df[col] = df['Cigarette/Bidi'].astype(str).str.lower().map(use_map).fillna(0)

# Encode gender: 'male'->1, 'female'->0
df['Gender'] = df['Gender'].astype(str).str.lower().map({'male': 1, 'female': 0})

# Encode study setting: 'hospital'->1, 'primary care'->0
df['Study setting'] = df['Study setting'].astype(str).str.lower().map({'hospital': 1, 'primary care': 0})

# Encode Risk: 'low risk'->0, 'high risk'->1
df['Risk'] = df['Risk'].astype(str).str.lower().map({'low risk': 0, 'high risk': 1})

# Encode Specialist diagnosis: 'non suspicious'->0, 'suspicious'->1
df['Specialist diagnosis'] = df['Specialist diagnosis'].astype(str).str.lower().map({
    'non suspicious': 0,
    'non-suspicious': 0,
    'suspicious': 1
})

# Encode 'yes' -> 1 and 'no' -> 0 for these two binary columns
df['Able to open mouth'] = df['Able to open mouth'].astype(str).str.lower().map({'yes': 1, 'no': 0}).fillna(0)
df['Lesion/patch'] = df['Lesion/patch'].astype(str).str.lower().map({'yes': 1, 'no': 0}).fillna(0)

# Map images to patients and resize for both MobileNet and U-Net
data = []
for idx, row in df.iterrows():
    patient_id = row['patient_id']
    raw_patient_dir = f'../../data/raw/patient_{patient_id}'  # Updated path
    mobilenet_patient_dir = f'../../data/processed/mobilenet/patient_{patient_id}'  # Updated path
    unet_patient_dir = f'../../data/processed/unet/patient_{patient_id}'  # Updated path
    
    # Create processed directories if they don't exist
    os.makedirs(mobilenet_patient_dir, exist_ok=True)
    os.makedirs(unet_patient_dir, exist_ok=True)
    
    # Check if the raw patient directory exists
    if not os.path.exists(raw_patient_dir):
        continue
    
    # Find all .jpg images for this patient
    for img_file in Path(raw_patient_dir).glob('*.jpg'):
        # Open the image
        img = Image.open(img_file)
        
        # Resize for MobileNet (224x224)
        img_mobilenet = img.resize((224, 224))
        mobilenet_img_path = os.path.join(mobilenet_patient_dir, img_file.name)
        img_mobilenet.save(mobilenet_img_path)
       
        # Resize for U-Net (512x512)
        img_unet = img.resize((512, 512))
        unet_img_path = os.path.join(unet_patient_dir, img_file.name)
        img_unet.save(unet_img_path)
        
        data.append({
            'mobilenet_image_path': mobilenet_img_path,  # For MobileNet
            'unet_image_path': unet_img_path,  # For U-Net
            'label': row['Specialist diagnosis'],  # Use Specialist diagnosis as the label
            'patient_id': patient_id,
            'risk': row['Risk'],
            'gender': row['Gender'],
            'study_setting': row['Study setting'],
            'cigarette_bidi': row['Cigarette/Bidi'],
            'smokeless_tobacco': row['Smokeless tobacco'],
            'areca_nut': row['Areca nut'],
            'alcohol': row['Alcohol'],
            'able_to_open_mouth': row['Able to open mouth'],
            'lesion_patch': row['Lesion/patch']
        })

# Create a new DataFrame with image paths and labels
processed_df = pd.DataFrame(data)

# Save the processed DataFrame to a CSV file
processed_df.to_csv('../../data/processed_labels.csv', index=False)  # Updated path

print("Preprocessing complete. Processed data saved to '../../data/processed_labels.csv'.")
print(processed_df.head())