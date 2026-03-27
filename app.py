import os
import streamlit as st
import torch
import torch.nn.functional as F
import numpy as np
import cv2
import timm
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image

# ==============================================================================
# AI Healthcare Hackathon 2026 — Presentation UI (V5 ULTRA)
# ==============================================================================
# "The interface must clearly display the statement: 
#  For research and demonstration purposes only. Not for clinical use."
# ==============================================================================

# --- Helper Functions ---
def robust_resize(img, sz, is_mask=False, return_meta=False):
    """Aspect-ratio preserving padding (Matches V5 Training)"""
    h, w = img.shape[:2]
    scale = sz / max(h, w)
    new_h, new_w = int(h * scale), int(w * scale)
    interp = cv2.INTER_NEAREST if is_mask else cv2.INTER_LINEAR
    img = cv2.resize(img, (new_w, new_h), interpolation=interp)
    
    pad_h = (sz - new_h) // 2
    pad_w = (sz - new_w) // 2
    img = cv2.copyMakeBorder(img, pad_h, sz - new_h - pad_h, pad_w, sz - new_w - pad_w, 
                            cv2.BORDER_CONSTANT, value=0)
    if return_meta:
        return img, {"pad_h": pad_h, "pad_w": pad_w, "new_h": new_h, "new_w": new_w}
    return img

def restore_original_mask(prob_mask, orig_h, orig_w, resize_meta):
    y0 = resize_meta["pad_h"]
    x0 = resize_meta["pad_w"]
    y1 = y0 + resize_meta["new_h"]
    x1 = x0 + resize_meta["new_w"]
    cropped = prob_mask[y0:y1, x0:x1]
    return cv2.resize(cropped, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
@st.cache_resource
def load_models(cls_path, seg_path, device):
    """Loads classification and segmentation models safely"""
    cls_model, seg_model = None, None
    cls_ckpt, seg_ckpt = None, None

    # Load Classification
    if os.path.exists(cls_path):
        try:
            cls_ckpt = torch.load(cls_path, map_location=device, weights_only=False)
            cls_model = timm.create_model(
                cls_ckpt.get("model_name", "tf_efficientnet_b2.ns_jft_in1k"), 
                pretrained=False, 
                num_classes=cls_ckpt.get("num_classes", 12)
            )
            cls_model.load_state_dict(cls_ckpt["model_state_dict"])
            cls_model.to(device).eval()
        except Exception as e:
            st.error(f"Classification Model Error: {e}")
    
    # Load Segmentation
    if os.path.exists(seg_path):
        try:
            seg_ckpt = torch.load(seg_path, map_location=device, weights_only=False)
            seg_model = smp.UnetPlusPlus(
                encoder_name=seg_ckpt.get("encoder", "efficientnet-b2"),
                encoder_weights=None,
                in_channels=3, 
                classes=1, 
                activation=None,
            )
            seg_model.load_state_dict(seg_ckpt["model_state_dict"])
            seg_model.to(device).eval()
            
            # Extract metadata and store in model object temporarily for inference
            seg_model.img_size = seg_ckpt.get("img_size", 224)
            seg_model.best_threshold = seg_ckpt.get("best_threshold", 0.5)
            
        except Exception as e:
            st.error(f"Segmentation Model Error: {e}")
            
    return cls_model, cls_ckpt, seg_model, seg_ckpt

# get_base_transforms removed — was dead code using A.Resize (squash) instead of robust_resize

@torch.no_grad()
def run_classification(model, img_np, img_size, device):
    img_resized = robust_resize(img_np, img_size)
    base_tfm = A.Compose([
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

    all_probs = []
    tta_images = [
        img_resized,
        np.fliplr(img_resized).copy(),
        np.flipud(img_resized).copy(),
        np.rot90(img_resized, 1).copy(),
    ]
    for aug_img in tta_images:
        tensor = base_tfm(image=aug_img)["image"].unsqueeze(0).to(device)
        with torch.autocast(device_type="cuda" if "cuda" in device else "cpu", enabled=False):
            logits = model(tensor)
        all_probs.append(F.softmax(logits, dim=1))

    probs = torch.stack(all_probs).mean(dim=0).squeeze().cpu().numpy()
    pred_class = np.argmax(probs)
    return pred_class, probs

def postprocess_mask(mask_binary):
    mask = mask_binary.astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels > 1:
        largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        mask = (labels == largest).astype(np.uint8)
    return mask

@torch.no_grad()
def run_segmentation(model, img_np, img_size, device):
    h_orig, w_orig = img_np.shape[:2]
    img_resized, resize_meta = robust_resize(img_np, img_size, return_meta=True)
    
    base_tfm = A.Compose([
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

    def predict_single(image):
        tensor = base_tfm(image=image)["image"].unsqueeze(0).to(device)
        with torch.autocast(device_type="cuda" if "cuda" in device else "cpu", enabled=False):
            logits = model(tensor)
        return torch.sigmoid(logits.float()).squeeze().cpu().numpy().astype(np.float32)

    preds = [
        predict_single(img_resized),
        np.fliplr(predict_single(np.fliplr(img_resized).copy())),
        np.flipud(predict_single(np.flipud(img_resized).copy())),
        np.rot90(predict_single(np.rot90(img_resized, 1).copy()), -1),
    ]

    prob_mask = np.mean(preds, axis=0).astype(np.float32)
    prob_mask = restore_original_mask(prob_mask, h_orig, w_orig, resize_meta)
    
    threshold = getattr(model, "best_threshold", 0.5)
    binary_mask = postprocess_mask((prob_mask > threshold).astype(np.uint8))
    return binary_mask

def draw_overlay(image_np, mask_np):
    """Draws a red highlight over the segmented area"""
    overlay = image_np.copy()
    color_mask = np.zeros_like(image_np)
    color_mask[mask_np == 1] = [255, 0, 0] # Red mask
    cv2.addWeighted(color_mask, 0.5, overlay, 1.0, 0, overlay)
    
    # Optional: draw contours
    contours, _ = cv2.findContours(mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (255, 255, 0), 2) # Yellow border
    return overlay

# --- UI Setup ---
st.set_page_config(page_title="Biopsy AI Analysis", page_icon="🧬", layout="wide")

# DISCLAIMER (MANDATORY BY GUIDELINES)
st.warning("⚠️ **For research and demonstration purposes only. Not for clinical use.**")

st.title("🔬 AI Healthcare Biopsy Analyzer")
st.markdown("Automated image classification and region segmentation using deep learning.")

# Sidebar - Settings
with st.sidebar:
    st.header("⚙️ Configuration")
    cls_model_path = st.text_input("Classification Model Path", "models/classification/best_model.pth")
    seg_model_path = st.text_input("Segmentation Model Path", "models/segmentation/best_model.pth")
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    st.info(f"Using compute device: **{device.upper()}**")

# Load Models
cls_model, cls_ckpt, seg_model, seg_ckpt = load_models(cls_model_path, seg_model_path, device)

# Main App
uploaded_file = st.file_uploader("Upload a Biopsy Image (PNG/JPG)", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Read Image
    image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Original Image")
        st.image(image, use_column_width=True)
        
    with col2:
        st.subheader("AI Analysis Results")
        
        # 1. Classification
        if cls_model is not None:
            with st.spinner("Running Classification..."):
                cls_size = cls_ckpt.get("img_size", 512)
                pred_class, probs = run_classification(cls_model, image_np, cls_size, device)
                confidence = float(probs[pred_class])
                
                if confidence < 0.40:
                    st.warning("⚠️ **Low Confidence (OOD)**: This image might not be a standard biopsy or belongs to an unknown class.")
                else:
                    st.success(f"**Predicted Disease Class:** Class {pred_class}")
                    
                st.progress(confidence, text=f"Confidence Score: {confidence*100:.1f}%")
                
                # Confidence Breakdown
                import pandas as pd
                prob_df = pd.DataFrame({
                    "Class": [f"Class {i}" for i in range(len(probs))],
                    "Probability": probs
                })
                st.bar_chart(prob_df.set_index("Class"))
        else:
            st.error("Classification model not loaded.")
            
        # 2. Segmentation
        if seg_model is not None:
            with st.spinner("Running ROI Segmentation..."):
                seg_size = seg_ckpt.get("img_size", 512)
                mask = run_segmentation(seg_model, image_np, seg_size, device)
                overlay_img = draw_overlay(image_np, mask)
                st.image(overlay_img, caption="Segmented Region of Interest", use_column_width=True)
                
                # Stats
                pixel_count = np.sum(mask)
                total_pixels = mask.size
                st.info(f"**Lesion Area:** {pixel_count} pixels ({pixel_count/total_pixels*100:.2f}% of image)")
        else:
            st.error("Segmentation model not loaded.")
            
else:
    st.info("Please upload an image to begin the analysis.")
