import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from lime import lime_image
from skimage.segmentation import slic
import cv2
import logging
import time
import os

# Configuration
CONFIG = {
    'RESULTS_DIR': '/home/user/Desktop/oral_cancer/gdc_oral_cancer_dataset/oral-cancer-detection/src/models/Classification/lime_results',
    'IMAGE_SIZE': 300,
    'DEVICE': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    'MIN_MUCOSAL_AREA': 400,
    'BYPASS_TOOL_MASK': True,
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

def detect_intraoral_roi(image):
    start = time.time()
    image_np = np.array(image)
    image_hsv = cv2.cvtColor(image_np, cv2.COLOR_RGB2HSV)
    timestamp = time.time()
    debug_dir = os.path.join(CONFIG['RESULTS_DIR'], 'debug')

    # Mucosa: Pink/red hues, exclude tongue
    lower_mucosa = np.array([0, 30, 20])
    upper_mucosa = np.array([25, 200, 255])
    lower_mucosa2 = np.array([155, 30, 20])
    upper_mucosa2 = np.array([180, 200, 255])
    mucosa_mask = cv2.bitwise_or(
        cv2.inRange(image_hsv, lower_mucosa, upper_mucosa),
        cv2.inRange(image_hsv, lower_mucosa2, upper_mucosa2)
    )
    cv2.imwrite(os.path.join(debug_dir, f'mucosa_mask_initial_{timestamp}.png'), mucosa_mask)

    # Exclude skin (fingers)
    lower_skin = np.array([0, 15, 80])
    upper_skin = np.array([25, 255, 255])
    skin_mask = cv2.inRange(image_hsv, lower_skin, upper_skin)
    skin_mask = cv2.dilate(skin_mask, np.ones((7, 7), np.uint8), iterations=2)
    mucosa_mask = cv2.bitwise_and(mucosa_mask, cv2.bitwise_not(skin_mask))
    cv2.imwrite(os.path.join(debug_dir, f'mucosa_mask_post_skin_{timestamp}.png'), mucosa_mask)

    # Exclude teeth
    _, _, v_channel = cv2.split(image_hsv)
    brightness_mask = v_channel < 180
    mucosa_mask = cv2.bitwise_and(mucosa_mask, mucosa_mask, mask=brightness_mask.astype(np.uint8))
    cv2.imwrite(os.path.join(debug_dir, f'mucosa_mask_post_teeth_{timestamp}.png'), mucosa_mask)

    # Texture filter
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    edge_magnitude = cv2.Laplacian(gray, cv2.CV_64F, ksize=3)
    edge_magnitude = cv2.normalize(np.abs(edge_magnitude), None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    texture_mask = edge_magnitude > 15
    mucosa_mask_pre_texture = mucosa_mask.copy()
    mucosa_mask = cv2.bitwise_and(mucosa_mask, mucosa_mask, mask=texture_mask.astype(np.uint8))
    cv2.imwrite(os.path.join(debug_dir, f'mucosa_mask_post_texture_{timestamp}.png'), mucosa_mask)
    if np.sum(mucosa_mask > 0) == 0:
        logging.warning(f"Texture filter zeroed out mucosa mask at {timestamp}, reverting")
        mucosa_mask = mucosa_mask_pre_texture

    # Tool mask
    tool_mask = np.zeros_like(mucosa_mask)
    if not CONFIG['BYPASS_TOOL_MASK']:
        tool_mask = cv2.bitwise_or(
            cv2.inRange(image_hsv, np.array([0, 70, 0]), np.array([180, 255, 130])),
            cv2.inRange(image_hsv, np.array([0, 0, 140]), np.array([180, 60, 255]))
        )
        tool_mask = cv2.dilate(tool_mask, np.ones((5, 5), np.uint8), iterations=2)
        mucosa_mask = cv2.bitwise_and(mucosa_mask, cv2.bitwise_not(tool_mask))
    cv2.imwrite(os.path.join(debug_dir, f'tool_mask_{timestamp}.png'), tool_mask)

    # Fallback if mucosa mask is empty
    if np.sum(mucosa_mask > 0) == 0:
        logging.warning(f"Empty mucosa mask at {timestamp}, using fallback mask")
        mucosa_mask = np.ones_like(mucosa_mask, dtype=np.uint8) * 255

    mucosa_coverage = np.sum(mucosa_mask > 0) / mucosa_mask.size
    logging.info(f"Mucosa mask coverage: {mucosa_coverage:.3f}, saved as mucosa_mask_final_{timestamp}.png")

    kernel = np.ones((5, 5), np.uint8)
    mucosa_mask = cv2.dilate(mucosa_mask, kernel, iterations=2)
    mucosa_mask = cv2.erode(mucosa_mask, kernel, iterations=1)
    cv2.imwrite(os.path.join(debug_dir, f'mucosa_mask_final_{timestamp}.png'), mucosa_mask)

    contours, _ = cv2.findContours(mucosa_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    is_mouth_open = False
    roi_mask = mucosa_mask
    bbox = None
    img_center = (image_np.shape[1] / 2, image_np.shape[0] / 2)
    if contours:
        valid_contours = []
        for c in contours:
            area = cv2.contourArea(c)
            if area > CONFIG['MIN_MUCOSAL_AREA']:
                x, y, w, h = cv2.boundingRect(c)
                contour_center = (x + w / 2, y + h / 2)
                distance_to_center = np.sqrt((contour_center[0] - img_center[0])**2 + (contour_center[1] - img_center[1])**2)
                perimeter = cv2.arcLength(c, True)
                circularity = (4 * np.pi * area) / (perimeter**2) if perimeter > 0 else 0
                if distance_to_center < min(image_np.shape[:2]) / 2.5 and 0.3 < circularity < 1.0:
                    valid_contours.append((c, area, x, y, w, h))
        if valid_contours:
            _, contour_area, x, y, w, h = max(valid_contours, key=lambda x: x[1])
            is_mouth_open = True
            x, y, w, h = max(0, x), max(0, y), min(w, image_np.shape[1] - x), min(h, image_np.shape[0] - y)
            bbox = (x, y, w, h)
            logging.info(f"Detected ROI bbox: {bbox}, area: {contour_area}")
        else:
            logging.info("No valid contours found")

    if not is_mouth_open:
        crop_size = min(image_np.shape[:2]) // 3
        cy, cx = image_np.shape[0] // 2, image_np.shape[1] // 2
        x = max(0, cx - crop_size)
        y = max(0, cy - crop_size)
        w = min(crop_size * 2, image_np.shape[1] - x)
        h = min(crop_size * 2, image_np.shape[0] - y)
        bbox = (x, y, w, h)
        roi_mask = np.zeros_like(mucosa_mask)
        roi_mask[y:y+h, x:x+w] = 255
        logging.info(f"Fallback central crop bbox: {bbox}")

    roi_image = Image.fromarray(image_np).resize((CONFIG['IMAGE_SIZE'], CONFIG['IMAGE_SIZE']))
    roi_mask = cv2.resize(roi_mask, (CONFIG['IMAGE_SIZE'], CONFIG['IMAGE_SIZE']), interpolation=cv2.INTER_NEAREST)
    logging.info(f"ROI detection time: {time.time() - start:.2f}s")
    return roi_image, roi_mask, bbox, is_mouth_open, tool_mask

class LIMEExplainer:
    def __init__(self, model, device=CONFIG['DEVICE']):
        self.model = model
        self.device = device
        self.mean = np.array([0.5, 0.5, 0.5])
        self.std = np.array([0.2, 0.2, 0.2])
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.2, 0.2, 0.2])
        ])

    def detect_reflections(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(blurred, 200, 255, cv2.THRESH_BINARY)
        kernel = np.ones((3, 3), np.uint8)
        thresh = cv2.dilate(thresh, kernel, iterations=1)
        reflection_mask = thresh == 0
        debug_dir = os.path.join(CONFIG['RESULTS_DIR'], 'debug')
        timestamp = time.time()
        cv2.imwrite(os.path.join(debug_dir, f'reflection_mask_{timestamp}.png'), reflection_mask.astype(np.uint8) * 255)
        reflection_coverage = np.sum(reflection_mask) / reflection_mask.size
        logging.info(f"Reflection mask coverage: {reflection_coverage:.3f}")
        if np.sum(reflection_mask) == 0:
            logging.warning(f"Empty reflection mask at {timestamp}, using fallback")
            reflection_mask = np.ones_like(reflection_mask, dtype=bool)
        return reflection_mask

    def enhanced_segmentation(self, image, n_segments=200):
        roi_image, roi_mask, _, _, _ = detect_intraoral_roi(Image.fromarray(image.astype(np.uint8)))
        reflection_mask = self.detect_reflections(image)
        combined_mask = np.logical_and(roi_mask > 0, reflection_mask)
        debug_dir = os.path.join(CONFIG['RESULTS_DIR'], 'debug')
        timestamp = time.time()
        cv2.imwrite(os.path.join(debug_dir, f'combined_mask_{timestamp}.png'), combined_mask.astype(np.uint8) * 255)
        combined_coverage = np.sum(combined_mask) / combined_mask.size
        logging.info(f"Combined mask coverage: {combined_coverage:.3f}")
        if np.sum(combined_mask) == 0:
            logging.warning(f"Empty combined mask at {timestamp}, using fallback")
            combined_mask = np.ones_like(combined_mask, dtype=bool)
        try:
            segments = slic(
                image,
                n_segments=n_segments,
                compactness=10,
                start_label=1,
                mask=combined_mask,
                enforce_connectivity=True,
                sigma=0.5
            )
        except Exception as e:
            logging.error(f"SLIC failed: {str(e)}, using default segmentation")
            segments = slic(
                image,
                n_segments=n_segments,
                compactness=10,
                start_label=1,
                enforce_connectivity=True,
                sigma=0.5
            )
        return segments

    def predict_fn(self, images):
        self.model.eval()
        predictions = []
        with torch.no_grad():
            for img in images:
                if img.max() > 1.0:
                    img = img / 255.0
                normalized = (img - self.mean) / self.std
                tensor = torch.FloatTensor(normalized).permute(2, 0, 1).unsqueeze(0).to(self.device)
                output = self.model(tensor)
                prob = torch.nn.functional.softmax(output, dim=1)
                predictions.append(prob.cpu().numpy()[0])
        return np.array(predictions)

    def explain_image(self, image_path, n_segments=200, brightness_adjust=1.0):
        original_image = Image.open(image_path).convert('RGB')
        img_array = np.array(original_image)
        img_array = np.clip(img_array * brightness_adjust, 0, 255).astype(np.uint8)
        adjusted_image = Image.fromarray(img_array)
        resized_image = adjusted_image.resize((CONFIG['IMAGE_SIZE'], CONFIG['IMAGE_SIZE']))
        img_array = np.array(resized_image)
        roi_image, roi_mask, bbox, is_mouth_open, tool_mask = detect_intraoral_roi(original_image)

        img_tensor = self.transform(resized_image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            output = self.model(img_tensor)
            probs = torch.nn.functional.softmax(output, dim=1)
            predicted_class = torch.argmax(probs, dim=1).item()
            confidence = probs.max().item()
            non_suspicious_prob = probs[0][0].item()
            suspicious_prob = probs[0][1].item()

        try:
            explainer = lime_image.LimeImageExplainer()
            explanation = explainer.explain_instance(
                img_array,
                self.predict_fn,
                top_labels=2,
                hide_color=0,
                num_samples=1500,
                segmentation_fn=lambda x: self.enhanced_segmentation(x, n_segments)
            )
            segments = self.enhanced_segmentation(img_array, n_segments)
            feature_importance = explanation.local_exp[predicted_class]
        except Exception as e:
            logging.error(f"LIME failed: {str(e)}, using fallback")
            segments = np.zeros((img_array.shape[0], img_array.shape[1]), dtype=np.int32)
            feature_importance = []
            explanation = type('Explanation', (), {'local_exp': {predicted_class: []}})

        positive_segments = [(seg_id, imp) for seg_id, imp in feature_importance if imp > 0]
        positive_segments.sort(key=lambda x: x[1], reverse=True)
        negative_segments = [(seg_id, imp) for seg_id, imp in feature_importance if imp < 0]
        negative_segments.sort(key=lambda x: x[1])

        cancer_type = self.determine_cancer_type(suspicious_prob, predicted_class)
        risk_level = self.determine_risk_level(suspicious_prob, predicted_class)

        return {
            'image': resized_image,
            'img_array': img_array,
            'roi_image': roi_image,
            'roi_mask': roi_mask,
            'bbox': bbox,
            'is_mouth_open': is_mouth_open,
            'tool_mask': tool_mask,
            'predicted_class': predicted_class,
            'confidence': confidence,
            'non_suspicious_prob': non_suspicious_prob,
            'suspicious_prob': suspicious_prob,
            'segments': segments,
            'positive_segments': positive_segments,
            'negative_segments': negative_segments,
            'all_segments': feature_importance,
            'explanation': explanation,
            'cancer_type': cancer_type,
            'risk_level': risk_level,
            'brightness_adjust': brightness_adjust
        }

    def determine_cancer_type(self, suspicious_prob, predicted_class):
        if predicted_class == 0:
            return CANCER_TYPES[8]
        else:
            if suspicious_prob > 0.9:
                return CANCER_TYPES[3]
            elif suspicious_prob > 0.8:
                return CANCER_TYPES[5]
            elif suspicious_prob > 0.7:
                return CANCER_TYPES[1]
            elif suspicious_prob > 0.6:
                return CANCER_TYPES[7]
            else:
                return CANCER_TYPES[9]

    def determine_risk_level(self, suspicious_prob, predicted_class):
        if predicted_class == 0:
            return "Low Risk"
        else:
            if suspicious_prob > 0.9:
                return "Very High Risk"
            elif suspicious_prob > 0.8:
                return "High Risk"
            elif suspicious_prob > 0.7:
                return "Moderate Risk"
            elif suspicious_prob > 0.6:
                return "Moderate-Low Risk"
            else:
                return "Low Risk"