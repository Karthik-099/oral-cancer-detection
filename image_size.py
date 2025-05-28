from PIL import Image
import os
import sys

def check_image_dimensions(directory):
    if not os.path.exists(directory):
        print(f"Error: Directory '{directory}' not found")
        return False
    found_images = False
    for filename in os.listdir(directory):
        if filename.lower().endswith('.jpeg'):
            found_images = True
            try:
                with Image.open(os.path.join(directory, filename)) as img:
                    width, height = img.size
                    print(f"Image: {filename}")
                    print(f"Dimensions: {width}x{height} pixels\n")
            except Exception as e:
                print(f"Error: Failed to open {filename} - {str(e)}\n")
    return found_images

if len(sys.argv) != 2:
    print("Usage: python image_size.py <patient_id>")
    print("Example: python image_size.py 123")
    sys.exit(1)

patient_id = sys.argv[1]

directory = rf"/home/karthik/oral-cancer-detection/data/raw{patient_id}"

if not check_image_dimensions(directory):
    print(f"No .jpeg images found in '{directory}'. Check the patient ID and directory structure.")