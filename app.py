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
# AI Healthcare Hackathon 2026 — Presentation UI
# ==============================================================================
# Guidelines Rule 11 & 12 Implementation
# "The interface must clearly display the statement: 
#  For research and demonstration purposes only. Not for clinical use."
# ==============================================================================

# --- Helper Functions ---
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
                cls_ckpt.get("model_name", "tf_efficientnet_b5_ns"), 
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
                encoder_name=seg_ckpt.get("encoder", "timm-efficientnet-b5"),
                encoder_weights=None,
                in_channels=3, 
                classes=1, 
                activation=None,
            )
            seg_model.load_state_dict(seg_ckpt["model_state_dict"])
            seg_model.to(device).eval()
        except Exception as e:
            st.error(f"Segmentation Model Error: {e}")
            
    return cls_model, cls_ckpt, seg_model, seg_ckpt

def get_base_transforms(img_size):
    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

@torch.no_grad()
def run_classification(model, img_np, img_size, device):
    tfm = get_base_transforms(img_size)
    tensor = tfm(image=img_np)["image"].unsqueeze(0).to(device)
    with torch.autocast(device_type="cuda" if "cuda" in device else "cpu", enabled=False):
        logits = model(tensor)
    probs = F.softmax(logits, dim=1).squeeze().cpu().numpy()
    pred_class = np.argmax(probs)
    return pred_class, probs

@torch.no_grad()
def run_segmentation(model, img_np, img_size, device):
    h_orig, w_orig = img_np.shape[:2]
    tfm = get_base_transforms(img_size)
    tensor = tfm(image=img_np)["image"].unsqueeze(0).to(device)
    with torch.autocast(device_type="cuda" if "cuda" in device else "cpu", enabled=False):
        logits = model(tensor)
    prob_mask = torch.sigmoid(logits.float()).squeeze().cpu().numpy()
    
    binary_mask = (prob_mask > 0.5).astype(np.uint8)
    binary_mask = cv2.resize(binary_mask, (w_orig, h_orig), interpolation=cv2.INTER_NEAREST)
    
    # Morphology
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
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
    cls_model_path = st.text_input("Classification Model Path", "models/classification_best.pth")
    seg_model_path = st.text_input("Segmentation Model Path", "models/segmentation_best.pth")
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
                st.success(f"**Predicted Disease Class:** {pred_class}")
                # Optional: Show confidence bar
                st.progress(float(probs[pred_class]), text=f"Confidence: {probs[pred_class]*100:.1f}%")
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
