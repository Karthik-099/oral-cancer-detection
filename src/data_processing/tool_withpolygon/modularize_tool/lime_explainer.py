import os
import lime
import lime.lime_image
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms
import cv2
import logging
import time
from skimage.segmentation import slic
from config import CONFIG, CANCER_TYPES

class LIMEExplainer:
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.mean = np.array([0.5, 0.5, 0.5])
        self.std = np.array([0.2, 0.2, 0.2])
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.2, 0.2, 0.2])
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
        from utils import detect_intraoral_roi
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
        predictions = np.array(predictions)
        logging.info(f"predict_fn output shape: {predictions.shape}, Prob range: min={np.min(predictions):.3f}, max={np.max(predictions):.3f}")
        return predictions

    def explain_image(self, image_path, n_segments=200):
        start_time = time.time()
        logging.info(f"Starting LIME explanation for {image_path} with {n_segments} segments")
        try:
            from utils import detect_intraoral_roi
            original_image = Image.open(image_path).convert('RGB')
            img_array = np.array(original_image)
            resized_image = original_image.resize((CONFIG['IMAGE_SIZE'], CONFIG['IMAGE_SIZE']))
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

            explainer = lime.lime_image.LimeImageExplainer()
            explanation = explainer.explain_instance(
                img_array,
                self.predict_fn,
                top_labels=2,
                hide_color=0,
                num_samples=1500,
                segmentation_fn=lambda x: self.enhanced_segmentation(x, n_segments)
            )
            segments = self.enhanced_segmentation(img_array, n_segments)
            logging.info(f"LIME explanation generated. Top labels: {explanation.top_labels}, "
                        f"local_pred type: {type(explanation.local_pred)}, "
                        f"local_pred shape: {explanation.local_pred.shape if isinstance(explanation.local_pred, np.ndarray) else 'N/A'}, "
                        f"local_exp keys: {list(explanation.local_exp.keys())}")

            feature_importance = explanation.local_exp.get(predicted_class, [])
            logging.info(f"Processing local_exp for class {predicted_class}: {feature_importance}")
            positive_segments = []
            negative_segments = []
            for item in feature_importance:
                try:
                    if not isinstance(item, (tuple, list)) or len(item) != 2:
                        logging.warning(f"Invalid item in local_exp: {item}, type: {type(item)}, length: {len(item) if isinstance(item, (tuple, list)) else 'N/A'}")
                        continue
                    seg_id, importance = item
                    if not isinstance(seg_id, (int, np.integer)) or not isinstance(importance, (int, float, np.floating)):
                        logging.warning(f"Invalid segment data: seg_id={seg_id} (type: {type(seg_id)}), importance={importance} (type: {type(importance)})")
                        continue
                    if importance > 0:
                        positive_segments.append((seg_id, importance))
                    elif importance < 0:
                        negative_segments.append((seg_id, importance))
                except ValueError as e:
                    logging.warning(f"Failed to unpack item in local_exp: {item}, error: {str(e)}")
                    continue
            positive_segments.sort(key=lambda x: x[1], reverse=True)
            negative_segments.sort(key=lambda x: x[1])
            logging.info(f"Found {len(positive_segments)} positive segments and {len(negative_segments)} negative segments for class {predicted_class}")

            cancer_type = self.determine_cancer_type(suspicious_prob, predicted_class)
            risk_level = self.determine_risk_level(suspicious_prob, predicted_class)

            output = {
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
                'risk_level': risk_level
            }
            logging.info(f"LIME explanation took {time.time() - start_time:.2f} seconds")
            return output
        except Exception as e:
            logging.error(f"LIME explanation failed: {str(e)}")
            segments = np.zeros((img_array.shape[0], img_array.shape[1]), dtype=np.int32)
            return {
                'image': resized_image,
                'img_array': img_array,
                'roi_image': img_array,
                'roi_mask': np.ones_like(img_array[:,:,0]),
                'bbox': None,
                'is_mouth_open': False,
                'tool_mask': np.zeros_like(img_array[:,:,0]),
                'predicted_class': 0,
                'confidence': 0.0,
                'non_suspicious_prob': 1.0,
                'suspicious_prob': 0.0,
                'segments': segments,
                'positive_segments': [],
                'negative_segments': [],
                'all_segments': [],
                'explanation': None,
                'cancer_type': 'Unknown',
                'risk_level': 'Unknown'
            }

    def determine_cancer_type(self, suspicious_prob, predicted_class):
        if predicted_class == 0:
            return 'Normal'
        else:
            if suspicious_prob > 0.9:
                return CANCER_TYPES[3]
            elif suspicious_prob > 0.8:
                return CANCER_TYPES[4]
            elif suspicious_prob > 0.7:
                return CANCER_TYPES[1]
            elif suspicious_prob > 0.6:
                return CANCER_TYPES[2]
            else:
                return CANCER_TYPES[5]

    def determine_risk_level(self, suspicious_prob, predicted_class):
        if predicted_class == 0:
            return "Low Risk"
        else:
            return "High Risk" if suspicious_prob > 0.7 else "Low Risk"