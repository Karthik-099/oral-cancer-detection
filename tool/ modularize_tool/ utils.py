import cv2
import numpy as np
import os
import logging
import time
from PIL import Image
from config import CONFIG, RESULTS_DIR

def setup_logging():
    logging.basicConfig(
        filename=os.path.join(RESULTS_DIR, 'debug.log'),
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

def normalize_path(image_path):
    norm_path = image_path.lower().replace('\\', '/').strip('/')
    data_dir_prefix = CONFIG['DATA_DIR'].lower().replace('\\', '/').strip('/')
    if norm_path.startswith(data_dir_prefix):
        norm_path = norm_path[len(data_dir_prefix):].lstrip('/')
    parts = norm_path.split('/')
    if len(parts) > 2:
        norm_path = '/'.join(parts[-2:])
    norm_path = os.path.splitext(norm_path)[0] + '.jpeg'
    logging.info(f"Normalized path: {image_path} -> {norm_path}")
    return norm_path

def detect_intraoral_roi(image):
    start = time.time()
    image_np = np.array(image)
    image_hsv = cv2.cvtColor(image_np, cv2.COLOR_RGB2HSV)
    timestamp = time.time()
    debug_dir = os.path.join(CONFIG['RESULTS_DIR'], 'debug')

    lower_mucosa = np.array([0, 30, 20])
    upper_mucosa = np.array([25, 200, 255])
    lower_mucosa2 = np.array([155, 30, 20])
    upper_mucosa2 = np.array([180, 200, 255])
    mucosa_mask = cv2.bitwise_or(
        cv2.inRange(image_hsv, lower_mucosa, upper_mucosa),
        cv2.inRange(image_hsv, lower_mucosa2, upper_mucosa2)
    )
    cv2.imwrite(os.path.join(debug_dir, f'mucosa_mask_initial_{timestamp}.png'), mucosa_mask)

    lower_skin = np.array([0, 15, 80])
    upper_skin = np.array([25, 255, 255])
    skin_mask = cv2.inRange(image_hsv, lower_skin, upper_skin)
    skin_mask = cv2.dilate(skin_mask, np.ones((7, 7), np.uint8), iterations=2)
    mucosa_mask = cv2.bitwise_and(mucosa_mask, cv2.bitwise_not(skin_mask))
    cv2.imwrite(os.path.join(debug_dir, f'mucosa_mask_post_skin_{timestamp}.png'), mucosa_mask)

    _, _, v_channel = cv2.split(image_hsv)
    brightness_mask = v_channel < 180
    mucosa_mask = cv2.bitwise_and(mucosa_mask, mucosa_mask, mask=brightness_mask.astype(np.uint8))
    cv2.imwrite(os.path.join(debug_dir, f'mucosa_mask_post_teeth_{timestamp}.png'), mucosa_mask)

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

    tool_mask = np.zeros_like(mucosa_mask)
    if not CONFIG['BYPASS_TOOL_MASK']:
        tool_mask = cv2.bitwise_or(
            cv2.inRange(image_hsv, np.array([0, 70, 0]), np.array([180, 255, 130])),
            cv2.inRange(image_hsv, np.array([0, 0, 140]), np.array([180, 60, 255]))
        )
        tool_mask = cv2.dilate(tool_mask, np.ones((5, 5), np.uint8), iterations=2)
        mucosa_mask = cv2.bitwise_and(mucosa_mask, cv2.bitwise_not(tool_mask))
    cv2.imwrite(os.path.join(debug_dir, f'tool_mask_{timestamp}.png'), tool_mask)

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