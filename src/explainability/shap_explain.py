import torch
import torch.nn as nn
from torchvision.models import mobilenet_v2
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import shap
import matplotlib.pyplot as plt

# Define device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


mobilenet = mobilenet_v2(pretrained=False)
mobilenet.classifier[1] = nn.Linear(mobilenet.classifier[1].in_features, 2)
mobilenet.load_state_dict(torch.load('../../models/mobilenet_classifier.pth'))
mobilenet = mobilenet.to(device)
mobilenet.eval()

# Define transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


background_paths = [
    f'../../../data/raw/patient_{i}/image1.jpeg' for i in range(1, 11)
]
background_images = []
for path in background_paths:
    img = Image.open(path).convert('RGB')
    img = transform(img).unsqueeze(0).to(device)
    background_images.append(img)
background = torch.cat(background_images, dim=0)

# Load the image to explain
image_path = '../../../data/raw/patient_1/image1.jpeg'
image = Image.open(image_path).convert('RGB')
image_tensor = transform(image).unsqueeze(0).to(device)

# Define a function to predict with MobileNet (required by SHAP)
def predict_fn(images):
    images = torch.from_numpy(images.transpose((0, 3, 1, 2))).float().to(device)
    with torch.no_grad():
        outputs = mobilenet(images)
        probs = torch.softmax(outputs, dim=1).cpu().numpy()
    return probs

# Initialize SHAP DeepExplainer
explainer = shap.DeepExplainer(
    model=mobilenet,
    data=background
)

# Compute SHAP values
shap_values = explainer.shap_values(image_tensor)

# Plot SHAP values for the 'Suspicious' class (class 1)
plt.figure()
shap.image_plot(
    shap_values=shap_values[1],  # SHAP values for class 1 (Suspicious)
    pixel_values=image_tensor.cpu().numpy().transpose((0, 2, 3, 1)),
    show=False
)
plt.savefig('../../../results/shap_explanation.png')
plt.close()

print("SHAP explanation saved to '../../../results/shap_explanation.png'")