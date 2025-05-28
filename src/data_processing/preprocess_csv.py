import pandas as pd
import os
from pathlib import Path

# Set display options for better visibility
pd.options.display.max_rows = 1000
pd.options.display.max_columns = 1000

# Read the CSV file
df = pd.read_csv('../../data/metadata.csv')
print(f"Initial number of rows in df: {len(df)}")
print("Columns in df:", df.columns.tolist())

# Drop rows with NaNs in key columns
df.dropna(subset=['Risk', 'Specialist diagnosis'], inplace=True)
print(f"Number of rows after dropna: {len(df)}")

# Replace '-' with 0
df.replace('-', 0, inplace=True)

# Process FLW username: Extract the number (e.g., "FLW 123" -> 123)
df['patient_id'] = df['FLW username'].apply(lambda x: int(str(x).split()[1]))

# Remove rows where both Risk and Specialist diagnosis are 0
df = df[(df['Risk'] != 0) | (df['Specialist diagnosis'] != 0)]
print(f"Number of rows in df after filtering: {len(df)}")

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

# Map images to patients (store original image paths)
data = []
for idx, row in df.iterrows():
    patient_id = row['patient_id']
    raw_patient_dir = f'../../data/raw/patient_{patient_id}'
    
    # Check if the raw patient directory exists
    print(f"Checking raw directory: {raw_patient_dir}")
    if not os.path.exists(raw_patient_dir):
        print(f"Directory {raw_patient_dir} does not exist, skipping...")
        continue
    
    # Find all .jpeg and .jpg images for this patient (case-insensitive)
    img_files = list(Path(raw_patient_dir).glob('*.[jJ][pP][eE]?[gG]'))
    print(f"Found {len(img_files)} images in {raw_patient_dir}")
    for img_file in img_files:
        print(f"Processing image path: {img_file}")
        data.append({
            'image_path': str(img_file),  # Store the original image path
            'label': row['Specialist diagnosis'],
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
processed_df.to_csv('../../data/processed_labels.csv', index=False)

print("Preprocessing complete. Processed data saved to '../../data/processed_labels.csv'.")
print(processed_df.head())