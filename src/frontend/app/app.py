from flask import Flask, request, render_template
import torch
import torch.nn as nn
from torchvision.models import mobilenet_v2
import segmentation_models_pytorch as smp
import torchvision.transforms as transforms
from PIL import Image
import io
import base64
import numpy as np
import shap

app = Flask(__name__)

# Define device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define transforms for MobileNet and U-Net
mobilenet_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

unet_transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load MobileNet model
mobilenet = mobilenet_v2(pretrained=False)
mobilenet.classifier[1] = nn.Linear(mobilenet.classifier[1].in_features, 2)
mobilenet.load_state_dict(torch.load('../../models/mobilenet_classifier.pth'))
mobilenet = mobilenet.to(device)
mobilenet.eval()

# Load U-Net model
unet = smp.Unet(encoder_name='resnet34', encoder_weights=None, in_channels=3, classes=1)
unet.load_state_dict(torch.load('../../models/unet_segmenter.pth'))
unet = unet.to(device)
unet.eval()

# Prepare background dataset for SHAP (load 10 images)
background_paths = [
    f'../../data/raw/patient_{i}/image1.jpeg' for i in range(1, 11)
]
background_images = []
for path in background_paths:
    img = Image.open(path).convert('RGB')
    img = mobilenet_transform(img).unsqueeze(0).to(device)
    background_images.append(img)
background = torch.cat(background_images, dim=0)

# Initialize SHAP DeepExplainer
def predict_fn(images):
    images = torch.from_numpy(images.transpose((0, 3, 1, 2))).float().to(device)
    with torch.no_grad():
        outputs = mobilenet(images)
        probs = torch.softmax(outputs, dim=1).cpu().numpy()
    return probs

explainer = shap.DeepExplainer(model=mobilenet, data=background)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return render_template('index.html', error='No image uploaded')

    file = request.files['image']
    try:
        image = Image.open(file.stream).convert('RGB')
    except Exception as e:
        return render_template('index.html', error='Invalid image file')

    # Classification with MobileNet
    mobilenet_input = mobilenet_transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        mobilenet_output = mobilenet(mobilenet_input)
        _, predicted = torch.max(mobilenet_output, 1)
        classification = 'Suspicious' if predicted.item() == 1 else 'Non-Suspicious'
        probability = torch.softmax(mobilenet_output, dim=1)[0][predicted.item()].item()

    # SHAP explanation
    shap_values = explainer.shap_values(mobilenet_input)
    # Convert SHAP values to an image
    shap_numpy = shap_values[1]  # For 'Suspicious' class
    pixel_values = mobilenet_input.cpu().numpy().transpose((0, 2, 3, 1))
    # Create a plot and save to bytes
    import matplotlib.pyplot as plt
    plt.figure()
    shap.image_plot(shap_values=shap_numpy, pixel_values=pixel_values, show=False)
    shap_io = io.BytesIO()
    plt.savefig(shap_io, format='png', bbox_inches='tight')
    plt.close()
    shap_base64 = base64.b64encode(shap_io.getvalue()).decode('utf-8')

    # Segmentation with U-Net (only if suspicious)
    mask_base64 = None
    if predicted.item() == 1:
        unet_input = unet_transform(image).unsqueeze(0).to(device)
        with torch.no_grad():
            unet_output = unet(unet_input)
            mask = torch.sigmoid(unet_output) > 0.5
            mask = mask.cpu().numpy().squeeze()
            mask_image = Image.fromarray((mask * 255).astype('uint8'))
            mask_io = io.BytesIO()
            mask_image.save(mask_io, format='PNG')
            mask_base64 = base64.b64encode(mask_io.getvalue()).decode('utf-8')

    # Encode the original image
    image_io = io.BytesIO()
    image.save(image_io, format='PNG')
    image_base64 = base64.b64encode(image_io.getvalue()).decode('utf-8')

    return render_template('result.html',
                          classification=classification,
                          probability=probability,
                          mask_base64=mask_base64,
                          image_base64=image_base64,
                          shap_base64=shap_base64)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
