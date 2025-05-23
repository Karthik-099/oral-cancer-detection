import os
from pathlib import Path

project_root = "oral_cancer_detection"
dirs = [
    "data/raw", "data/processed", "data/annotations", "data/logs",
    "src/data_processing", "src/models", "src/training", "src/inference", "src/frontend", "src/utils",
    "logs/training", "logs/inference", "models", "notebooks", "docker"
]

for i in range(1, 11):
    dirs.extend([f"data/raw/patient_{i}", f"data/processed/patient_{i}", f"data/annotations/patient_{i}"])

files = {
    "data/metadata.csv": "",
    "data/logs/preprocessing.log": "",
    "src/data_processing/download_images.py": "",
    "src/data_processing/preprocess.py": "",
    "src/data_processing/annotate.py": "",
    "src/models/classification.py": "",
    "src/models/segmentation.py": "",
    "src/training/train_classifier.py": "",
    "src/training/train_segmentation.py": "",
    "src/inference/predict.py": "",
    "src/frontend/app.py": "",
    "src/frontend/index.html": "",
    "src/frontend/styles.css": "",
    "src/frontend/script.js": "",
    "src/utils/metrics.py": "",
    "logs/training/classifier.log": "",
    "logs/training/segmentation.log": "",
    "logs/inference/inference.log": "",
    "notebooks/experiments.ipynb": "",
    "docker/Dockerfile": "",
    "requirements.txt": "",
    "README.md": "",
    ".gitignore": ""
}

for directory in dirs:
    os.makedirs(os.path.join(project_root, directory), exist_ok=True)

for file_path, content in files.items():
    with open(os.path.join(project_root, file_path), "w") as f:
        f.write(content)

print("Project structure created successfully!")