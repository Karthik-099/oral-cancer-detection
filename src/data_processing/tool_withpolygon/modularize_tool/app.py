import streamlit as st
import torch
import numpy as np
from PIL import Image, ImageDraw
import cv2
import os
import json
from datetime import datetime
import logging
import scipy.ndimage as ndimage
import matplotlib.pyplot as plt
import io
from streamlit_drawable_canvas import st_canvas
from zipfile import ZipFile
from config import CONFIG, CANCER_TYPES
from models import EnhancedEfficientNetB4
from gradcam import GradCAM
from lime_explainer import LIMEExplainer
from utils import detect_intraoral_roi
import torchvision.transforms as transforms

# Setup logging (minimized for clean UI)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load model
@st.cache_resource
def load_model():
    model = EnhancedEfficientNetB4(num_classes=CONFIG['NUM_CLASSES'])
    checkpoint = torch.load(CONFIG['MODEL_PATH'], map_location=CONFIG['DEVICE'], weights_only=False)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model = model.to(CONFIG['DEVICE'])
    model.eval()
    explainer = LIMEExplainer(model, CONFIG['DEVICE'])
    return model, explainer

model, explainer = load_model()

# Streamlit app
st.title("Oral Cancer Detection Tool")

# Initialize session state
if 'results' not in st.session_state:
    st.session_state.results = {}
if 'feedback' not in st.session_state:
    st.session_state.feedback = {}
if 'image_paths' not in st.session_state:
    st.session_state.image_paths = {}
if 'analysis_done' not in st.session_state:
    st.session_state.analysis_done = False
if 'knob_settings' not in st.session_state:
    st.session_state.knob_settings = {'n_segments': 200, 'top_contributors': 100, 'show_numbers': True, 'show_boundaries': True}
if 'canvas_key_counter' not in st.session_state:
    st.session_state.canvas_key_counter = 0

# Patient ID input
patient_id = st.text_input("Patient ID (e.g., patient_1)", value="patient_1")

# Image upload (multiple files)
uploaded_files = st.file_uploader("Upload images for {}".format(patient_id), type=["jpg", "jpeg", "png", "bmp"], accept_multiple_files=True)

# Save uploaded files temporarily
if uploaded_files:
    for file in uploaded_files:
        image_path = f"temp_{file.name}"
        with open(image_path, "wb") as f:
            f.write(file.getvalue())
        st.session_state.image_paths[file.name] = image_path
        if file.name not in st.session_state.results:
            st.session_state.results[file.name] = None
        if file.name not in st.session_state.feedback:
            st.session_state.feedback[file.name] = {'polygons': [], 'comments': '', 'cancer_type': ''}

# Parameters
col_param1, col_param2 = st.columns(2)
with col_param1:
    n_segments = st.slider("Segments", min_value=100, max_value=300, value=200, key="segments")
with col_param2:
    top_contributors = st.slider("Top Contributors (%)", min_value=10, max_value=100, value=100, key="top_contributors")
col_check1, col_check2 = st.columns(2)
with col_check1:
    show_numbers = st.checkbox("Show Numbers", value=True, key="show_numbers")
with col_check2:
    show_boundaries = st.checkbox("Show Boundaries", value=True, key="show_boundaries")

# Check if knob settings changed
knobs_changed = (
    n_segments != st.session_state.knob_settings['n_segments'] or
    top_contributors != st.session_state.knob_settings['top_contributors'] or
    show_numbers != st.session_state.knob_settings['show_numbers'] or
    show_boundaries != st.session_state.knob_settings['show_boundaries']
)

# Update knob settings
st.session_state.knob_settings = {
    'n_segments': n_segments,
    'top_contributors': top_contributors,
    'show_numbers': show_numbers,
    'show_boundaries': show_boundaries
}

# Analyze all images
if uploaded_files and st.button("Analyze All Images"):
    st.session_state.analysis_done = False
    with st.spinner("Analyzing images..."):
        try:
            for file_name, image_path in st.session_state.image_paths.items():
                if st.session_state.results[file_name] is None or n_segments != st.session_state.knob_settings['n_segments']:
                    # Run LIME explanation only if not cached or segments changed
                    current_data = explainer.explain_image(image_path, n_segments=n_segments)
                    current_data['image'] = Image.open(image_path).convert('RGB')  # Store PIL image
                    # Run Grad-CAM
                    transform = transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.2, 0.2, 0.2])
                    ])
                    img = Image.open(image_path).convert('RGB')
                    roi_image, roi_mask, _, is_mouth_open, _ = detect_intraoral_roi(img)
                    current_data['roi_image'] = roi_image
                    current_data['roi_mask'] = roi_mask
                    current_data['is_mouth_open'] = is_mouth_open
                    input_tensor = transform(roi_image).unsqueeze(0).to(CONFIG['DEVICE'])
                    grad_cam = GradCAM(model)
                    heatmap, gradcam_class, _, gradcam_confidence = grad_cam.generate(input_tensor, roi_mask)
                    current_data['gradcam_heatmap'] = heatmap
                    current_data['gradcam_class'] = gradcam_class
                    current_data['gradcam_confidence'] = gradcam_confidence
                    st.session_state.results[file_name] = current_data
            st.session_state.analysis_done = True
            st.success("Analysis completed for all images.")
        except Exception as e:
            st.error(f"Analysis failed: {str(e)}")
            logging.error(f"Analysis failed: {str(e)}")

# Display results for each image
for file_name, current_data in st.session_state.results.items():
    if current_data:
        st.subheader(f"Results for {file_name} (Patient: {patient_id})")
        pred_class = current_data['predicted_class']
        class_name = ['Non-Suspicious', 'Suspicious'][pred_class]
        confidence = current_data['confidence']
        non_suspicious_prob = current_data['non_suspicious_prob']
        suspicious_prob = current_data['suspicious_prob']
        cancer_type = current_data['cancer_type']
        risk_level = current_data['risk_level']
        num_segments = len(current_data['positive_segments'])
        num_to_show = max(1, int(num_segments * (top_contributors / 100)))
        st.write(f"Class: {class_name}")
        st.write(f"Confidence: {confidence:.3f}")
        st.write(f"Probabilities: Non-Suspicious={non_suspicious_prob:.3f}, Suspicious={suspicious_prob:.3f}")
        st.write(f"Cancer Type: {cancer_type}")
        st.write(f"Risk Level: {risk_level}")
        st.write(f"Segments: {num_to_show}/{num_segments} displayed")

        # Visualizations
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Original Image")
            st.image(current_data['image'], use_column_width=True)

        with col2:
            st.subheader("LIME Segmentation")
            fig_lime, ax_lime = plt.subplots()
            img_array = current_data['img_array']
            segments = current_data['segments']
            positive_segments = current_data['positive_segments']
            segments_to_show = positive_segments[:num_to_show]
            display_image = img_array.copy().astype(np.float32) / 255.0
            heatmap = np.zeros_like(display_image[:,:,0])
            if segments_to_show:
                importances = [imp for _, imp in segments_to_show]
                if importances:
                    min_imp, max_imp = min(importances), max(importances)
                    imp_range = max_imp - min_imp if max_imp != min_imp else 1
                    for i, (seg_id, importance) in enumerate(segments_to_show):
                        mask = segments == seg_id
                        norm_imp = (importance - min_imp) / imp_range if imp_range > 0 else 1
                        heatmap[mask] = norm_imp
                        if show_numbers:
                            y_coords, x_coords = np.where(mask)
                            if len(y_coords) > 0:
                                centroid_y, centroid_x = int(np.mean(y_coords)), int(np.mean(x_coords))
                                ax_lime.text(centroid_x, centroid_y, f'S{i}',
                                             fontsize=8, fontweight='bold', color='white',
                                             ha='center', va='center',
                                             bbox=dict(facecolor='black', alpha=0.9))
                    heatmap_color = plt.cm.hot(heatmap)[:,:,:3]
                    blended = 0.7 * display_image + 0.3 * heatmap_color
                    if show_boundaries:
                        boundary_image = np.zeros_like(blended)
                        for seg_id, _ in segments_to_show:
                            mask = segments == seg_id
                            edges = ndimage.binary_erosion(mask) ^ mask
                            boundary_image[edges] = [1, 1, 1]
                        boundary_mask = np.any(boundary_image > 0, axis=2)
                        blended[boundary_mask] = boundary_image[boundary_mask]
                    display_image = blended
            ax_lime.imshow(display_image)
            title_text = f'LIME Heatmap: {class_name} ({confidence:.3f})\n{cancer_type} | {risk_level}\n({num_to_show}/{num_segments} segments)'
            ax_lime.set_title(title_text, fontsize=10)
            ax_lime.axis('off')
            st.pyplot(fig_lime)

        col3, col4 = st.columns(2)
        with col3:
            st.subheader("Grad-CAM Heatmap")
            fig_gradcam, ax_gradcam = plt.subplots()
            heatmap = current_data['gradcam_heatmap']
            heatmap = cv2.resize(heatmap, (CONFIG['IMAGE_SIZE'], CONFIG['IMAGE_SIZE']))
            heatmap = np.uint8(255 * heatmap)
            heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
            heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
            roi_array = np.array(current_data['roi_image'])
            overlay = roi_array * 0.6 + heatmap * 0.4
            ax_gradcam.imshow(overlay.astype(np.uint8))
            gradcam_class_name = 'Suspicious' if current_data['gradcam_class'] == 1 else 'Non-Suspicious'
            title_text = (
                f'Grad-CAM Heatmap\nPrediction: {gradcam_class_name} '
                f'(Confidence: {current_data["gradcam_confidence"]:.3f})\n'
                f'Mouth Status: {"Open" if current_data["is_mouth_open"] else "Closed"}'
            )
            ax_gradcam.set_title(title_text, fontsize=10)
            ax_gradcam.axis('off')
            st.pyplot(fig_gradcam)

        with col4:
            st.subheader("LIME on Grad-CAM Overlay")
            fig_lime_gradcam, ax_lime_gradcam = plt.subplots()
            display_image = overlay.astype(np.float32) / 255.0
            if segments_to_show:
                if show_boundaries:
                    boundary_image = np.zeros_like(display_image)
                    for seg_id, _ in segments_to_show:
                        mask = segments == seg_id
                        edges = ndimage.binary_erosion(mask) ^ mask
                        boundary_image[edges] = [1, 1, 1]
                    boundary_mask = np.any(boundary_image > 0, axis=2)
                    display_image[boundary_mask] = boundary_image[boundary_mask]
                if show_numbers:
                    for i, (seg_id, _) in enumerate(segments_to_show):
                        mask = segments == seg_id
                        y_coords, x_coords = np.where(mask)
                        if len(y_coords) > 0:
                            centroid_y, centroid_x = int(np.mean(y_coords)), int(np.mean(x_coords))
                            ax_lime_gradcam.text(centroid_x, centroid_y, f'S{i}',
                                                 fontsize=8, fontweight='bold', color='white',
                                                 ha='center', va='center',
                                                 bbox=dict(facecolor='black', alpha=0.9))
            ax_lime_gradcam.imshow(display_image)
            title_text = (
                f'LIME on Grad-CAM Overlay\nPrediction: {class_name} ({confidence:.3f})\n'
                f'{cancer_type}\nSegments: {num_to_show}/{num_segments}'
            )
            ax_lime_gradcam.set_title(title_text, fontsize=10)
            ax_lime_gradcam.axis('off')
            st.pyplot(fig_lime_gradcam)

        # Contribution Analysis
        st.subheader("Contribution Analysis")
        fig_contrib, ax_contrib = plt.subplots()
        positive_segments = current_data['positive_segments']
        if positive_segments:
            importances = [imp for _, imp in positive_segments[:num_to_show]]
            segment_labels = [f'S{i}' for i in range(len(importances))]
            bars = ax_contrib.bar(segment_labels, importances, color='#D3D3D3')
            ax_contrib.set_ylabel('Contribution Score')
            ax_contrib.set_xlabel('Segments')
            title_text = f'Segment Contribution Analysis\nPrediction: {class_name} (Confidence: {confidence:.3f})\nTop {num_to_show}/{num_segments} Segments'
            ax_contrib.set_title(title_text, fontsize=10)
            ax_contrib.set_ylim(0, max(importances) * 1.1 if importances else 1.0)
            for bar in bars:
                height = bar.get_height()
                ax_contrib.text(
                    bar.get_x() + bar.get_width() / 2, height,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=8
                )
            plt.tight_layout()
            st.pyplot(fig_contrib)
        else:
            ax_contrib.text(0.5, 0.5, 'No positive contributors found', ha='center', va='center', fontsize=10)
            st.pyplot(fig_contrib)

        # Doctor Feedback
        st.subheader(f"Doctor Feedback for {file_name}")
        cancer_types = list(CANCER_TYPES.values())
        selected_cancer_type = st.selectbox(
            f"Select Cancer Type for {file_name}",
            [""] + cancer_types,
            key=f"cancer_type_{file_name}"
        )
        
        st.subheader("Polygon Annotation")
        # Drawable canvas for polygon annotation
        canvas_result = st_canvas(
            fill_color="rgba(255, 165, 0, 0.3)",  # Semi-transparent orange for polygon fill
            stroke_width=2,
            stroke_color="#FF0000",  # Red for polygon lines
            background_image=current_data['image'],
            update_streamlit=True,
            height=400,
            drawing_mode="polygon",  # Enable polygon mode
            point_display_radius=5,  # Make points visible
            key=f"canvas_{file_name}"
        )
        doctor_comments = st.text_area(f"Doctor Comments for {file_name}", key=f"comments_{file_name}")
        if st.button(f"Save Feedback for {file_name}", key=f"save_feedback_{file_name}"):
            if canvas_result.json_data and canvas_result.json_data.get("objects"):
                st.session_state.feedback[file_name]['polygons'] = canvas_result.json_data["objects"]
                st.session_state.feedback[file_name]['comments'] = doctor_comments
                st.session_state.feedback[file_name]['cancer_type'] = selected_cancer_type
                st.success(f"Feedback saved for {file_name}. Polygons: {len(canvas_result.json_data['objects'])}")
            else:
                st.warning(f"No polygons drawn on the canvas for {file_name}.")

# Download All Results button
if st.session_state.results and any(st.session_state.results.values()) and st.button("Download All Results"):
    zip_buffer = io.BytesIO()
    with ZipFile(zip_buffer, "w") as zip_file:
        for file_name, current_data in st.session_state.results.items():
            if current_data:
                base_name = f"{patient_id}_{file_name.replace('.', '_')}"
                # Save original image
                img_buffer = io.BytesIO()
                current_data['image'].save(img_buffer, format="PNG")
                zip_file.writestr(f"{patient_id}/{base_name}_original.png", img_buffer.getvalue())

                # Save LIME heatmap
                fig_lime, ax_lime = plt.subplots()
                img_array = current_data['img_array']
                segments = current_data['segments']
                positive_segments = current_data['positive_segments']
                segments_to_show = positive_segments[:num_to_show]
                display_image = img_array.copy().astype(np.float32) / 255.0
                heatmap = np.zeros_like(display_image[:,:,0])
                if segments_to_show:
                    importances = [imp for _, imp in segments_to_show]
                    if importances:
                        min_imp, max_imp = min(importances), max(importances)
                        imp_range = max_imp - min_imp if max_imp != min_imp else 1
                        for i, (seg_id, importance) in enumerate(segments_to_show):
                            mask = segments == seg_id
                            norm_imp = (importance - min_imp) / imp_range if imp_range > 0 else 1
                            heatmap[mask] = norm_imp
                            if show_numbers:
                                y_coords, x_coords = np.where(mask)
                                if len(y_coords) > 0:
                                    centroid_y, centroid_x = int(np.mean(y_coords)), int(np.mean(x_coords))
                                    ax_lime.text(centroid_x, centroid_y, f'S{i}',
                                                 fontsize=8, fontweight='bold', color='white',
                                                 ha='center', va='center',
                                                 bbox=dict(facecolor='black', alpha=0.9))
                        heatmap_color = plt.cm.hot(heatmap)[:,:,:3]
                        blended = 0.7 * display_image + 0.3 * heatmap_color 
                        if show_boundaries:
                            boundary_image = np.zeros_like(blended)
                            for seg_id, _ in segments_to_show:
                                mask = segments == seg_id
                                edges = ndimage.binary_erosion(mask) ^ mask
                                boundary_image[edges] = [1, 1, 1]
                            boundary_mask = np.any(boundary_image > 0, axis=2)
                            blended[boundary_mask] = boundary_image[boundary_mask]
                        display_image = blended
                ax_lime.imshow(display_image)
                ax_lime.set_title(
                    f'LIME Heatmap: {class_name} ({confidence:.3f})\n{cancer_type} | {risk_level}\n({num_to_show}/{num_segments} segments)',
                    fontsize=10
                )
                ax_lime.axis('off')
                lime_buffer = io.BytesIO()
                fig_lime.savefig(lime_buffer, format="PNG", bbox_inches='tight')
                zip_file.writestr(f"{patient_id}/{base_name}_lime.png", lime_buffer.getvalue())
                plt.close(fig_lime)

                # Save Grad-CAM heatmap
                fig_gradcam, ax_gradcam = plt.subplots()
                heatmap = current_data['gradcam_heatmap']
                heatmap = cv2.resize(heatmap, (CONFIG['IMAGE_SIZE'], CONFIG['IMAGE_SIZE']))
                heatmap = np.uint8(255 * heatmap)
                heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
                heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
                roi_array = np.array(current_data['roi_image'])
                overlay = roi_array * 0.6 + heatmap * 0.4
                ax_gradcam.imshow(overlay.astype(np.uint8))
                gradcam_class_name = 'Suspicious' if current_data['gradcam_class'] == 1 else 'Non-Suspicious'
                ax_gradcam.set_title(
                    f'Grad-CAM Heatmap\nPrediction: {gradcam_class_name} (Confidence: {current_data["gradcam_confidence"]:.3f})\n'
                    f'Mouth Status: {"Open" if current_data["is_mouth_open"] else "Closed"}',
                    fontsize=10
                )
                ax_gradcam.axis('off')
                gradcam_buffer = io.BytesIO()
                fig_gradcam.savefig(gradcam_buffer, format="PNG", bbox_inches='tight')
                zip_file.writestr(f"{patient_id}/{base_name}_gradcam.png", gradcam_buffer.getvalue())
                plt.close(fig_gradcam)

                # Save LIME on Grad-CAM overlay
                fig_lime_gradcam, ax_lime_gradcam = plt.subplots()
                display_image = overlay.astype(np.float32) / 255.0
                if segments_to_show:
                    if show_boundaries:
                        boundary_image = np.zeros_like(display_image)
                        for seg_id, _ in segments_to_show:
                            mask = segments == seg_id
                            edges = ndimage.binary_erosion(mask) ^ mask
                            boundary_image[edges] = [1, 1, 1]
                        boundary_mask = np.any(boundary_image > 0, axis=2)
                        display_image[boundary_mask] = boundary_image[boundary_mask]
                    if show_numbers:
                        for i, (seg_id, _) in enumerate(segments_to_show):
                            mask = segments == seg_id
                            y_coords, x_coords = np.where(mask)
                            if len(y_coords) > 0:
                                centroid_y, centroid_x = int(np.mean(y_coords)), int(np.mean(x_coords))
                                ax_lime_gradcam.text(centroid_x, centroid_y, f'S{i}',
                                                     fontsize=8, fontweight='bold', color='white',
                                                     ha='center', va='center',
                                                     bbox=dict(facecolor='black', alpha=0.9))
                ax_lime_gradcam.imshow(display_image)
                ax_lime_gradcam.set_title(
                    f'LIME on Grad-CAM Overlay\nPrediction: {class_name} ({confidence:.3f})\n'
                    f'{cancer_type}\nSegments: {num_to_show}/{num_segments}',
                    fontsize=10
                )
                ax_lime_gradcam.axis('off')
                lime_gradcam_buffer = io.BytesIO()
                fig_lime_gradcam.savefig(lime_gradcam_buffer, format="PNG", bbox_inches='tight')
                zip_file.writestr(f"{patient_id}/{base_name}_lime_gradcam.png", lime_gradcam_buffer.getvalue())
                plt.close(fig_lime_gradcam)

                # Save Contribution Analysis
                fig_contrib, ax_contrib = plt.subplots()
                positive_segments = current_data['positive_segments']
                if positive_segments:
                    importances = [imp for _, imp in positive_segments[:num_to_show]]
                    segment_labels = [f'S{i}' for i in range(len(importances))]
                    bars = ax_contrib.bar(segment_labels, importances, color='#D3D3D3')
                    ax_contrib.set_ylabel('Contribution Score')
                    ax_contrib.set_xlabel('Segments')
                    ax_contrib.set_title(
                        f'Segment Contribution Analysis\nPrediction: {class_name} (Confidence: {confidence:.3f})\n'
                        f'Top {num_to_show}/{num_segments} Segments',
                        fontsize=10
                    )
                    ax_contrib.set_ylim(0, max(importances) * 1.1 if importances else 1.0)
                    for bar in bars:
                        height = bar.get_height()
                        ax_contrib.text(
                            bar.get_x() + bar.get_width() / 2, height,
                            f'{height:.3f}', ha='center', va='bottom', fontsize=8
                        )
                    plt.tight_layout()
                else:
                    ax_contrib.text(0.5, 0.5, 'No positive contributors found', ha='center', va='center', fontsize=10)
                contrib_buffer = io.BytesIO()
                fig_contrib.savefig(contrib_buffer, format="PNG", bbox_inches='tight')
                zip_file.writestr(f"{patient_id}/{base_name}_contrib.png", contrib_buffer.getvalue())
                plt.close(fig_contrib)

                # Save Feedback Image
                feedback_img = current_data['image'].copy()
                draw = ImageDraw.Draw(feedback_img)
                for polygon in st.session_state.feedback[file_name]['polygons']:
                    if polygon['type'] == 'path':
                        path = polygon['path']
                        for i in range(len(path) - 1):
                            draw.line([tuple(path[i]), tuple(path[i + 1])], fill='red', width=2)
                        if len(path) > 2:
                            draw.line([tuple(path[-1]), tuple(path[0])], fill='red', width=2)
                        for point in path:
                            draw.ellipse([point[0] - 5, point[1] - 5, point[0] + 5, point[1] + 5], fill='red')
                        if st.session_state.feedback[file_name]['cancer_type']:
                            centroid_x = np.mean([p[0] for p in path]) if path else 0
                            centroid_y = np.mean([p[1] for p in path]) if path else 0
                            draw.text((centroid_x, centroid_y), st.session_state.feedback[file_name]['cancer_type'], fill='blue')
                feedback_buffer = io.BytesIO()
                feedback_img.save(feedback_buffer, format="PNG")
                zip_file.writestr(f"{patient_id}/{base_name}_feedback.png", feedback_buffer.getvalue())

                # Save JSON metadata
                metadata = {
                    'image_name': file_name,
                    'patient_id': patient_id,
                    'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S"),
                    'predicted_class': current_data['predicted_class'],
                    'confidence': current_data['confidence'],
                    'non_suspicious_prob': current_data['non_suspicious_prob'],
                    'suspicious_prob': current_data['suspicious_prob'],
                    'cancer_type': current_data['cancer_type'],
                    'risk_level': current_data['risk_level'],
                    'polygons': st.session_state.feedback[file_name]['polygons'],
                    'doctor_comments': st.session_state.feedback[file_name]['comments'],
                    'doctor_cancer_type': st.session_state.feedback[file_name]['cancer_type'],
                    'segments_count': len(current_data['positive_segments'])
                }
                zip_file.writestr(f"{patient_id}/{base_name}_metadata.json", json.dumps(metadata, indent=4))

    zip_buffer.seek(0)
    st.download_button(
        label=f"Download {patient_id}.zip",
        data=zip_buffer,
        file_name=f"{patient_id}.zip",
        mime="application/zip"
    )


def cleanup_temp_files():
    for file_name, image_path in st.session_state.image_paths.items():
        if os.path.exists(image_path):
            os.remove(image_path)
    st.session_state.image_paths = {}
    st.session_state.results = {}
    st.session_state.feedback = {}
    st.session_state.analysis_done = False
    st.session_state.canvas_key_counter += 1
    map.function(calculate=set.add)