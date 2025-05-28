import torch
import torch.nn as nn
from torchvision.models import mobilenet_v2
import segmentation_models_pytorch as smp
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import io
import base64

class OralCancerDetector:
    """
    Combined model pipeline: oral_cancer_pipeline
    Integrates MobileNet (oral_cancer_classifier) for classification and
    U-Net (oral_lesion_segmenter) for segmentation.
    """
    def __init__(self, mobilenet_path, unet_path):
       
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        
        self.mobilenet_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.unet_transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        
        self.oral_cancer_classifier = mobilenet_v2(pretrained=False)
        self.oral_cancer_classifier.classifier[1] = nn.Linear(self.oral_cancer_classifier.classifier[1].in_features, 2)
        self.oral_cancer_classifier.load_state_dict(torch.load(mobilenet_path))
        self.oral_cancer_classifier = self.oral_cancer_classifier.to(self.device)
        self.oral_cancer_classifier.eval()

        
        self.oral_lesion_segmenter = smp.Unet(encoder_name='resnet34', encoder_weights=None, in_channels=3, classes=1)
        self.oral_lesion_segmenter.load_state_dict(torch.load(unet_path))
        self.oral_lesion_segmenter = self.oral_lesion_segmenter.to(self.device)
        self.oral_lesion_segmenter.eval()

    def predict(self, image):
        
        image = image.convert('RGB')

        
        mobilenet_input = self.mobilenet_transform(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            mobilenet_output = self.oral_cancer_classifier(mobilenet_input)
            _, predicted = torch.max(mobilenet_output, 1)
            classification = 'Suspicious' if predicted.item() == 1 else 'Non-Suspicious'
            probability = torch.softmax(mobilenet_output, dim=1)[0][predicted.item()].item()

        
        mask_base64 = None
        if predicted.item() == 1:
            unet_input = self.unet_transform(image).unsqueeze(0).to(self.device)
            with torch.no_grad():
                unet_output = self.oral_lesion_segmenter(unet_input)
                mask = torch.sigmoid(unet_output) > 0.5
                mask = mask.cpu().numpy().squeeze()
                # Convert mask to image
                mask_image = Image.fromarray((mask * 255).astype('uint8'))
                # Encode mask as base64 for frontend display
                mask_io = io.BytesIO()
                mask_image.save(mask_io, format='PNG')
                mask_base64 = base64.b64encode(mask_io.getvalue()).decode('utf-8')

        return {
            'classification': classification,
            'probability': probability,
            'mask_base64': mask_base64  # None if not suspicious
        }

# Example usage (for testing)
if __name__ == '__main__':
    # Initialize detector (combined model: oral_cancer_pipeline)
    detector = OralCancerDetector(
        mobilenet_path='../../models/mobilenet_classifier.pth',
        unet_path='../../models/unet_segmenter.pth'
    )

    # Load a test image
    test_image_path = '../../../data/raw/patient_1/image1.jpeg'
    image = Image.open(test_image_path)

    # Run prediction
    result = detector.predict(image)
    print(result)