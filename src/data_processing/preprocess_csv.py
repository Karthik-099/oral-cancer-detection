import pandas as pd
import os
from pathlib import Path

df = pd.read_csv('../data/metadata.csv')
diagnosis_map = {
    'Oral cavity normal': 'Non-suspicious', 'Benign': 'Non-suspicious', 'Other': 'Non-suspicious',
    'Smokeless tobacco keratosis': 'Suspicious', 'Homogenous oral leukoplakia': 'Suspicious',
    'Oral lichen planus': 'Suspicious', 'Oral submucous fibrosis': 'Not applicable',
    'Speckled oral leukoplakia': 'Suspicious', 'Erythroplakia': 'Suspicious',
    'Verrucous oral leukoplakia': 'Suspicious', 'Proliferative verrucous leukoplakia': 'Suspicious',
    'Squamous cell carcinoma of oral mucous membrane': 'Suspicious', 'Non-diagnostic procedure finding': 'Not applicable'
}
data = []
for idx, row in df.iterrows():
    patient_id = idx + 1
    label = diagnosis_map.get(row['Provisional diagnosis'], 'Not applicable')
    if label == 'Not applicable':
        continue
    label_binary = 1 if label == 'Suspicious' else 0
    patient_dir = f'../data/raw/patient_{patient_id}'
    for img_file in Path(patient_dir).glob('*.jpg'):
        data.append({
            'image_path': str(img_file),
            'label': label_binary,
            'patient_id': patient_id,
            'provisional_diagnosis': row['Provisional diagnosis']
        })
processed_df = pd.DataFrame(data)
processed_df.to_csv('../data/processed_labels.csv', index=False)