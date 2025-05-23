import pandas as pd
import requests
import os
from urllib.parse import urlparse

df = pd.read_csv('../data/metadata.csv')
for i in range(1, 11):
    os.makedirs(f'../data/raw/patient_{i}', exist_ok=True)
for idx, row in df.iterrows():
    patient_id = idx + 1
    for col in [f'Image {i}' for i in range(1, 13)]:
        if pd.notna(row[col]):
            url = row[col]
            filename = os.path.basename(urlparse(url).path) + '.jpg'
            dest_path = f'../data/raw/patient_{patient_id}/{filename}'
            response = requests.get(url, stream=True)
            if response.status_code == 200:
                with open(dest_path, 'wb') as f:
                    f.write(response.content)