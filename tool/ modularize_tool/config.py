import os

# Setup base directory
BASE_DIR = '/home/user/Desktop/oral_cancer/gdc_oral_cancer_dataset/oral-cancer-detection'

# Create results directories
RESULTS_DIR = os.path.join(BASE_DIR, 'src/models/Classification/lime_results')
os.makedirs(os.path.join(RESULTS_DIR, 'debug'), exist_ok=True)
os.makedirs(os.path.join(RESULTS_DIR, 'lime_gradcam_results'), exist_ok=True)
os.makedirs(os.path.join(RESULTS_DIR, 'combined_results'), exist_ok=True)

CONFIG = {
    'MODEL_PATH': os.path.join(BASE_DIR, 'models/trained_efficientnet_b4_oral_cancer.pth'),
    'DATA_DIR': os.path.join(BASE_DIR, 'data/bm'),
    'CSV_PATH': os.path.join(BASE_DIR, 'data/processed_labels.csv'),
    'RESULTS_DIR': RESULTS_DIR,
    'IMAGE_SIZE': 300,
    'DEVICE': 'cuda' if torch.cuda.is_available() else 'cpu',
    'HEATMAP_THRESHOLD': 0.3,
    'MIN_MUCOSAL_AREA': 400,
    'BYPASS_TOOL_MASK': True,
    'BYPASS_ROI_MASK': False,
    'NUM_CLASSES': 2,
    'HOTSPOT_THRESHOLD': 0.8,
}

CANCER_TYPES = {
    1: 'Homogenous Oral Leukoplakia',
    2: 'Smokeless Tobacco Keratosis',
    3: 'Squamous Cell Carcinoma',
    4: 'Proliferative Verrucous Oral Leukoplakia',
    5: 'Erythroplakia',
    6: 'Oral Lichen Planus',
    7: 'Speckled Oral Leukoplakia',
    8: 'Oral Cavity Normal',
    9: 'Benign',
    10: 'Other'
}