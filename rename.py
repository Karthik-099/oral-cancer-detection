import os

# Set this to your target directory
base_dir = '/home/user/Desktop/oral_cancer/gdc_oral_cancer_dataset/oral-cancer-detection/data/original'

# List all items in the directory
for folder_name in os.listdir(base_dir):
    folder_path = os.path.join(base_dir, folder_name)
    
    # Check if it's a directory and name is numeric
    if os.path.isdir(folder_path) and folder_name.isdigit():
        new_name = f'patient_{folder_name}'
        new_path = os.path.join(base_dir, new_name)
        
        # Rename the folder
        os.rename(folder_path, new_path)
        print(f"Renamed {folder_name} -> {new_name}")
    else:
        print(f"Skipped {folder_name}, not a numeric directory.")