import os
import streamlit as st
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import cv2
import timm
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image

# ==============================================================================
# AI Healthcare Hackathon 2026 — Presentation UI (PRO VERSION)
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

# --- MULTI-LANGUAGE DICTIONARY ---
TRANSLATIONS = {
    "🇺🇿 O'zbekcha": {
        "title": "🩺 Tibbiy AI Diagnostika Tizimi",
        "warn": "⚠️ Faqat tadqiqot va ko'rgazma maqsadlarida. Klinik foydalanish uchun taqiqlanadi.",
        "settings": "⚙️ Tizim Sozlamalari",
        "cls_model": "Klassifikatsiya Modeli Yo'li",
        "seg_model": "Segmentatsiya Modeli Yo'li",
        "tab1": "Kasallikni Aniqlash (Task A)",
        "tab2": "To'qimani Segmentlash (Task B)",
        "upload": "📥 Biopsiya rasmini yuklang (PNG/JPG)",
        "original": "📸 Asl Rasm",
        "results": "🤖 AI Tahlil Natijasi",
        "running_cls": "🔬 chuqur tahlil qilinmoqda...",
        "running_seg": "🔬 zararlangan hudud chizilmoqda...",
        "pred_class": "Bashorat Qilingan Sinf:",
        "conf": "Ishonchlilik Darajasi:",
        "low_conf": "⚠️ Past Ishonchlilik: Kadrda artefaktlar bo'lishi yoki noma'lum sinf bo'lishi mumkin.",
        "distribution": "📊 Ehtimolliklar Taqsimoti",
        "lesion_area": "Kasallangan Maydon",
        "tissue_prop": "Umumiy Kadrga Nisbatan",
        "upload_req": "Tahlilni boshlash uchun fayl yuklang.",
        "err_cls": "Klassifikatsiya modeli yuklanmadi.",
        "err_seg": "Segmentatsiya modeli yuklanmadi."
    },
    "🇬🇧 English": {
        "title": "🩺 AI Medical Diagnostic System",
        "warn": "⚠️ For research and demonstration purposes only. Not for clinical use.",
        "settings": "⚙️ System Configuration",
        "cls_model": "Classification Model Path",
        "seg_model": "Segmentation Model Path",
        "tab1": "Disease Classification (Task A)",
        "tab2": "Tissue Segmentation (Task B)",
        "upload": "📥 Upload Biopsy Image (PNG/JPG)",
        "original": "📸 Original Image",
        "results": "🤖 AI Analysis Insights",
        "running_cls": "🔬 Running Deep Classification...",
        "running_seg": "🔬 Mapping Region of Interest...",
        "pred_class": "Predicted Disease Class:",
        "conf": "Confidence Score:",
        "low_conf": "⚠️ Low Confidence: Artifact detected or Out-of-Distribution sample.",
        "distribution": "📊 Probability Distribution",
        "lesion_area": "Lesion Pixel Area",
        "tissue_prop": "Tissue Proportion",
        "upload_req": "Please upload an image to begin the analysis.",
        "err_cls": "Classification module offline.",
        "err_seg": "Segmentation module offline."
    },
    "🇷🇺 Русский": {
        "title": "🩺 ИИ Система Медицинской Диагностики",
        "warn": "⚠️ Только для исследований и демонстрации. Не для клинического использования.",
        "settings": "⚙️ Настройки Системы",
        "cls_model": "Путь к модели классификации",
        "seg_model": "Путь к модели сегментации",
        "tab1": "Классификация Заболеваний (Task A)",
        "tab2": "Сегментация Тканей (Task B)",
        "upload": "📥 Загрузите снимок биопсии (PNG/JPG)",
        "original": "📸 Исходный снимок",
        "results": "🤖 Результаты ИИ-анализа",
        "running_cls": "🔬 Выполнение глубокой классификации...",
        "running_seg": "🔬 Картографирование зараженной области...",
        "pred_class": "Прогнозируемый Класс:",
        "conf": "Уровень Уверенности:",
        "low_conf": "⚠️ Низкая уверенность: обнаружен артефакт или неизвестный класс.",
        "distribution": "📊 Распределение Вероятностей",
        "lesion_area": "Площадь Поражения",
        "tissue_prop": "Доля от общего кадра",
        "upload_req": "Пожалуйста, загрузите изображение для начала.",
        "err_cls": "Модуль классификации отключен.",
        "err_seg": "Модуль сегментации отключен."
    }
}

# --- UI Setup & Premium CSS ---
st.set_page_config(page_title="AI Medical Diagnostics", page_icon="🩺", layout="wide", initial_sidebar_state="expanded")

# Inject Clean, Clinical CSS Theme
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;500;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Roboto', sans-serif;
    }
    
    /* Clinical Minimalist Theme */
    .stApp {
        background-color: #f8fafc;
        color: #0f172a;
    }
    
    /* Clean Cards */
    .clinical-card {
        background: #ffffff;
        border-radius: 12px;
        border: 1px solid #e2e8f0;
        padding: 25px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.03);
        margin-bottom: 20px;
    }
    
    /* Header Styling */
    .main-title {
        color: #0f172a;
        font-weight: 700;
        font-size: 2.5rem;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    
    /* Disclaimer Banner */
    .disclaimer {
        background-color: #fef2f2;
        border-left: 4px solid #ef4444;
        color: #991b1b;
        padding: 12px 20px;
        border-radius: 6px;
        font-size: 0.95rem;
        font-weight: 500;
        text-align: center;
        margin-bottom: 25px;
    }
    
    /* Metric Cards */
    .metric-box {
        background: #f1f5f9;
        border-radius: 8px;
        padding: 15px;
        text-align: center;
        border-top: 4px solid #3b82f6;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #1e293b;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #64748b;
        text-transform: uppercase;
        font-weight: 500;
    }
    
    /* Custom Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 20px;
        background-color: #ffffff;
        padding: 10px 20px 0px 20px;
        border-radius: 12px 12px 0 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.02);
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: transparent;
        border-radius: 4px 4px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
        color: #64748b;
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        color: #2563eb !important;
        border-bottom: 3px solid #2563eb !important;
    }
</style>
""", unsafe_allow_html=True)

# ─── Sidebar Settings & Languages ───
with st.sidebar:
    selected_lang = st.selectbox("🌐 Language / Til / Язык", list(TRANSLATIONS.keys()))
    t = TRANSLATIONS[selected_lang]
    
    st.markdown("---")
    st.header(t["settings"])
    cls_model_path = st.text_input(t["cls_model"], "models/classification/best_model.pth")
    seg_model_path = st.text_input(t["seg_model"], "models/segmentation/best_model.pth")
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    st.info(f"Compute Engine: **{device.upper()}**")

# Load Models
cls_model, cls_ckpt, seg_model, seg_ckpt = load_models(cls_model_path, seg_model_path, device)

# ─── App Header ───
st.markdown(f'<div class="disclaimer">{t["warn"]}</div>', unsafe_allow_html=True)
st.markdown(f'<div class="main-title">{t["title"]}</div>', unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

# ─── Main Interface ───
st.markdown('<div class="clinical-card">', unsafe_allow_html=True)
uploaded_file = st.file_uploader(t["upload"], type=["png", "jpg", "jpeg"])
st.markdown('</div>', unsafe_allow_html=True)

if uploaded_file is not None:
    # Read Image
    image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image)
    
    # ─── TABS LAYOUT ───
    tab1, tab2 = st.tabs([t["tab1"], t["tab2"]])
    
    # ==========================================
    # TAB 1: CLASSIFICATION (Task A)
    # ==========================================
    with tab1:
        st.markdown('<div class="clinical-card">', unsafe_allow_html=True)
        colA1, colA2 = st.columns([1, 1], gap="large")
        
        with colA1:
            st.markdown(f"### {t['original']}")
            st.image(image, use_container_width=True, clamp=True)
            
        with colA2:
            st.markdown(f"### {t['results']}")
            if cls_model is not None:
                with st.spinner(t["running_cls"]):
                    cls_size = cls_ckpt.get("img_size", 224)
                    pred_class, probs = run_classification(cls_model, image_np, cls_size, device)
                    confidence = float(probs[pred_class])
                    
                    # Highlight color based on confidence
                    color = "#22c55e" if confidence > 0.8 else ("#eab308" if confidence > 0.5 else "#ef4444")
                    
                    st.markdown(f"""
                    <div class="metric-box" style="border-top-color: {color};">
                        <div class="metric-label">{t['pred_class']}</div>
                        <div class="metric-value">Class {pred_class}</div>
                        <div class="metric-label" style="text-transform: none; margin-top: 8px; font-size: 1rem; color: #475569;">
                            {t['conf']} <strong style="color: {color};">{(confidence * 100):.1f}%</strong>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    if confidence < 0.40:
                        st.warning(t["low_conf"])
                    
                    st.markdown("<br>", unsafe_allow_html=True)
                    with st.expander(t["distribution"]):
                        prob_df = pd.DataFrame({"Probability": probs})
                        st.bar_chart(prob_df)
            else:
                st.error(t["err_cls"])
        st.markdown('</div>', unsafe_allow_html=True)

    # ==========================================
    # TAB 2: SEGMENTATION (Task B)
    # ==========================================
    with tab2:
        st.markdown('<div class="clinical-card">', unsafe_allow_html=True)
        colB1, colB2 = st.columns([1, 1], gap="large")
        
        with colB1:
            st.markdown(f"### {t['original']}")
            st.image(image, use_container_width=True, clamp=True)
            
        with colB2:
            st.markdown(f"### {t['results']}")
            if seg_model is not None:
                with st.spinner(t["running_seg"]):
                    seg_size = seg_ckpt.get("img_size", 224)
                    mask = run_segmentation(seg_model, image_np, seg_size, device)
                    overlay_img = draw_overlay(image_np, mask)
                    
                    st.image(overlay_img, use_container_width=True)
                    
                    # Stats calculation
                    pixel_count = int(np.sum(mask))
                    total_pixels = mask.size
                    percentage = (pixel_count / total_pixels) * 100
                    
                    st.markdown(f"""
                    <div style="display: flex; gap: 15px; margin-top: 15px;">
                        <div class="metric-box" style="flex: 1; border-top-color: #8b5cf6;">
                            <div class="metric-label">{t['lesion_area']}</div>
                            <div class="metric-value" style="font-size: 1.5rem; color: #7c3aed;">{pixel_count:,} px</div>
                        </div>
                        <div class="metric-box" style="flex: 1; border-top-color: #ec4899;">
                            <div class="metric-label">{t['tissue_prop']}</div>
                            <div class="metric-value" style="font-size: 1.5rem; color: #db2777;">{percentage:.2f}%</div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.error(t["err_seg"])
        st.markdown('</div>', unsafe_allow_html=True)

else:
    st.markdown(f"""
    <div style="text-align: center; padding: 60px; color: #94a3b8; background: #ffffff; border-radius: 12px; border: 1px dashed #cbd5e1;">
        <h2 style="font-weight: 300; margin-bottom: 10px;">{t['title']}</h2>
        <p>{t['upload_req']}</p>
    </div>
    """, unsafe_allow_html=True)

