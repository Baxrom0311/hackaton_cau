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

# --- UI Setup & Premium CSS ---
st.set_page_config(page_title="Biopsy AI Analysis", page_icon="🧬", layout="wide", initial_sidebar_state="collapsed")

# Inject Custom Premium CSS (Glassmorphism, Animations, Modern Fonts)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    /* Main Background Gradient */
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 100%);
        color: #f8fafc;
    }
    
    /* Glassmorphic Container */
    .glass-container {
        background: rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border-radius: 20px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 30px;
        box-shadow: 0 4px 30px rgba(0, 0, 0, 0.3);
        margin-bottom: 20px;
        animation: fadeIn 0.8s ease-out;
    }
    
    /* Typography & Glow */
    .title-glow {
        font-size: 3rem;
        font-weight: 800;
        background: -webkit-linear-gradient(45deg, #38bdf8, #818cf8, #c084fc);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 5px;
        text-shadow: 0px 0px 20px rgba(129, 140, 248, 0.4);
    }
    .subtitle {
        text-align: center;
        color: #94a3b8;
        font-size: 1.1rem;
        margin-bottom: 30px;
        font-weight: 300;
    }
    
    /* Disclaimer Banner */
    .disclaimer {
        background: rgba(239, 68, 68, 0.1);
        border-left: 4px solid #ef4444;
        color: #fca5a5;
        padding: 12px 20px;
        border-radius: 8px;
        font-size: 0.9rem;
        font-weight: 600;
        text-align: center;
        margin-bottom: 30px;
    }
    
    /* Image Wrappers */
    .img-wrapper {
        border-radius: 12px;
        overflow: hidden;
        border: 2px solid rgba(255,255,255,0.05);
        box-shadow: 0 10px 25px rgba(0,0,0,0.5);
        transition: transform 0.3s ease;
    }
    .img-wrapper:hover {
        transform: scale(1.02);
        border-color: rgba(56, 189, 248, 0.5);
    }
    
    /* Metric Cards */
    .metric-card {
        background: rgba(15, 23, 42, 0.6);
        border: 1px solid rgba(56, 189, 248, 0.2);
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        margin-top: 15px;
    }
    .metric-value {
        font-size: 2.5rem;
        font-weight: 800;
        color: #38bdf8;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #cbd5e1;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* Custom Animations */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    /* Hide Streamlit components */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ─── App Header ───
st.markdown('<div class="disclaimer">⚠️ For research and demonstration purposes only. Not for clinical use.</div>', unsafe_allow_html=True)
st.markdown('<div class="title-glow">🔬 AI Biopsy Diagnostic System</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Next-Generation Classification & Segmentation powered by Deep Learning</div>', unsafe_allow_html=True)

# ─── Sidebar Settings (Hidden by default for clean UI) ───
with st.sidebar:
    st.header("⚙️ System Configuration")
    cls_model_path = st.text_input("Classification Model Path", "models/classification/best_model.pth")
    seg_model_path = st.text_input("Segmentation Model Path", "models/segmentation/best_model.pth")
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    st.info(f"Compute Engine: **{device.upper()}**")

# Load Models
cls_model, cls_ckpt, seg_model, seg_ckpt = load_models(cls_model_path, seg_model_path, device)

# ─── Main Interface ───
st.markdown('<div class="glass-container">', unsafe_allow_html=True)
uploaded_file = st.file_uploader("📥 Upload Biopsy Image (PNG/JPG)", type=["png", "jpg", "jpeg"])
st.markdown('</div>', unsafe_allow_html=True)

if uploaded_file is not None:
    # Read Image
    image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image)
    
    st.markdown('<div class="glass-container">', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1], gap="large")
    
    with col1:
        st.markdown("### 📸 Original Biopsy Image")
        st.image(image, use_column_width=True, clamp=True)
        
    with col2:
        st.markdown("### 🤖 Clinical AI Insights")
        
        # 1. Classification
        if cls_model is not None:
            with st.spinner("🔬 Running Deep Classification..."):
                cls_size = cls_ckpt.get("img_size", 224)
                pred_class, probs = run_classification(cls_model, image_np, cls_size, device)
                confidence = float(probs[pred_class])
                
                # Dynamic Metric Card
                color = "#22c55e" if confidence > 0.8 else ("#eab308" if confidence > 0.5 else "#ef4444")
                st.markdown(f"""
                <div class="metric-card" style="border-left: 4px solid {color};">
                    <div class="metric-label">Predicted Disease Class</div>
                    <div class="metric-value">Class {pred_class}</div>
                    <div class="metric-label" style="text-transform: none; margin-top: 5px; color: {color};">
                        Confidence: {(confidence * 100):.1f}%
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                if confidence < 0.40:
                    st.warning("⚠️ **Low Confidence**: Artifact detected or Out-of-Distribution sample.")
                
                with st.expander("📊 View Probability Distribution"):
                    prob_df = pd.DataFrame({"Probability": probs})
                    st.bar_chart(prob_df)
        else:
            st.error("Classification module offline.")
            
        # 2. Segmentation
        if seg_model is not None:
            st.markdown("---")
            with st.spinner("🔬 Mapping Region of Interest (ROI)..."):
                seg_size = seg_ckpt.get("img_size", 224)
                mask = run_segmentation(seg_model, image_np, seg_size, device)
                overlay_img = draw_overlay(image_np, mask)
                
                st.markdown("### 🎯 AI Segmentation Map")
                st.image(overlay_img, use_column_width=True)
                
                # Stats
                pixel_count = int(np.sum(mask))
                total_pixels = mask.size
                percentage = (pixel_count / total_pixels) * 100
                
                st.markdown(f"""
                <div style="display: flex; gap: 15px; margin-top: 15px;">
                    <div class="metric-card" style="flex: 1; padding: 10px;">
                        <div class="metric-label">Lesion Pixel Area</div>
                        <div style="font-size: 1.5rem; color: #a855f7; font-weight: bold;">{pixel_count:,} px</div>
                    </div>
                    <div class="metric-card" style="flex: 1; padding: 10px;">
                        <div class="metric-label">Tissue Proportion</div>
                        <div style="font-size: 1.5rem; color: #ec4899; font-weight: bold;">{percentage:.2f}%</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.error("Segmentation module offline.")
            
    st.markdown('</div>', unsafe_allow_html=True) # End glass container
else:
    st.markdown("""
    <div style="text-align: center; padding: 50px; color: #64748b;">
        <h2 style="font-weight: 300;">Ready for Analysis</h2>
        <p>Upload a biopsy image above to trigger the AI pipeline.</p>
    </div>
    """, unsafe_allow_html=True)

