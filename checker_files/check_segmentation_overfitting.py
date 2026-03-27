# ============================================================
# Segmentation Overfitting yoki Underfitting Tekshiruvchisi
# ============================================================

import os
import torch
import numpy as np
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
from tqdm import tqdm
import cv2

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models/segmentation/best_model.pth")

# Avtomatik ravishda rasmlar papkasini topish
def find_image_dir():
    candidates = [
        os.path.join(BASE_DIR, "dataset/Segmentation/training/images"),
        os.path.join(BASE_DIR, "dataset/Segmentation/validation/images"),
        os.path.join(BASE_DIR, "dataset/Segmentation/testing/images"),
        os.path.join(BASE_DIR, "Segmentation/training/images"),
        os.path.join(BASE_DIR, "Segmentation/validation/images"),
        os.path.join(BASE_DIR, "Segmentation/testing/images"),
    ]
    for c in candidates:
        if os.path.exists(c): return c
    return None

def find_mask_dir():
    candidates = [
        os.path.join(BASE_DIR, "dataset/Segmentation/validation/masks"),
        os.path.join(BASE_DIR, "dataset/Segmentation/training/masks"),
        os.path.join(BASE_DIR, "Segmentation/validation/masks"),
        os.path.join(BASE_DIR, "Segmentation/training/masks"),
    ]
    for c in candidates:
        if os.path.exists(c): return c
    return None

TRAIN_IMG_DIR = find_image_dir()
TRAIN_MSK_DIR = find_mask_dir()

def robust_resize(img, sz, is_mask=False):
    """Aspect-ratio preserving padding (Ultra Quality)"""
    h, w = img.shape[:2]
    scale = sz / max(h, w)
    new_h, new_w = int(h * scale), int(w * scale)
    interp = cv2.INTER_NEAREST if is_mask else cv2.INTER_LINEAR
    img = cv2.resize(img, (new_w, new_h), interpolation=interp)
    
    pad_h = (sz - new_h) // 2
    pad_w = (sz - new_w) // 2
    img = cv2.copyMakeBorder(img, pad_h, sz - new_h - pad_h, pad_w, sz - new_w - pad_w, 
                            cv2.BORDER_CONSTANT, value=0)
    return img

device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

def check_segmentation_overfitting():
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Model topilmadi: {MODEL_PATH}")
        return
        
    print(f"🔍 {MODEL_PATH} tekshirilmoqda...")
    ckpt = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    
    # Robust Loading (Supports full checkpoint AND legacy state_dict)
    is_full_ckpt = "model_state_dict" in ckpt
    
    val_iou = ckpt.get("val_iou", None)
    epoch_saved = ckpt.get("epoch", "Noma'lum")
    img_size = ckpt.get("img_size", 512)
    encoder_name = ckpt.get("encoder", "timm-efficientnet-b5")
    
    # State dict correction
    state_dict = ckpt["model_state_dict"] if is_full_ckpt else ckpt
    
    print(f"📦 Model ichidagi ma'lumotlar:")
    print(f"  • Saqlangan Epoch: {epoch_saved}")
    if val_iou is not None:
        print(f"  • Validation IoU (Colab dagi): {val_iou*100:.2f}%")
    else:
        print(f"  • Validation IoU (Colab dagi): NOMA'LUM (Eski format)")
    
    if not (os.path.exists(TRAIN_IMG_DIR) and os.path.exists(TRAIN_MSK_DIR)):
        print(f"\n❌ '{TRAIN_IMG_DIR}' topilmadi kompyuteringizda.")
        print(f"💡 HUKM: Colabdagi konsolda 'Train IoU' nechi chiqqaniga qarang.")
        v_disp = f"{val_iou*100:.2f}%" if val_iou is not None else "80-85%"
        print(f"  - Agar Train IoU = 95% va Val IoU = {v_disp} bo'lsa -> OVERFITTING")
        print(f"  - Agar Train IoU = {v_disp} ga yaqin bo'lsa -> MODEL SOG'LOM!")
        return

    model = smp.UnetPlusPlus(
        encoder_name=encoder_name,
        encoder_weights=None,
        in_channels=3, classes=1, activation=None,
    )
    model.load_state_dict(state_dict)
    model.to(device).eval()

    tfm = A.Compose([
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

    # Rasm va maskani ID orqali bog'lash (extentsion va 'x' suffixga qaramasdan)
    processed_count = 0
    ious = []
    
    # Rasmlar ro'yxatini olish
    img_files = [f for f in os.listdir(TRAIN_IMG_DIR) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
    
    for f in tqdm(img_files, desc="IoU ni o'lchash", leave=False):
        img_id = os.path.splitext(f)[0]
        img_path = os.path.join(TRAIN_IMG_DIR, f)
        
        # Maskani qidirish: ID.png yoki IDx.png
        msk_candidates = [
            os.path.join(TRAIN_MSK_DIR, f"{img_id}.png"),
            os.path.join(TRAIN_MSK_DIR, f"{img_id}x.png"),
        ]
        
        msk_path = None
        for c in msk_candidates:
            if os.path.exists(c):
                msk_path = c
                break
        
        if msk_path is None:
            continue
            
        try:
            # Rasm va maskani o'qish (ROBUST RESIZE bilan)
            img_bgr = cv2.imread(img_path)
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            mask_raw = cv2.imread(msk_path, cv2.IMREAD_GRAYSCALE)
            
            # 🛡️ Robust Preprocessing (Training bilan bir xil)
            img_processed = robust_resize(img_rgb, img_size, is_mask=False)
            mask_true = robust_resize(mask_raw, img_size, is_mask=True)
            mask_true = (mask_true > 127).astype(np.float32)
            
            tensor = tfm(image=img_processed)["image"].unsqueeze(0).to(device)
            with torch.no_grad():
                logits = model(tensor)
                pred_mask = torch.sigmoid(logits.float()).squeeze().cpu().numpy() > 0.5
                    
            intersection = np.logical_and(mask_true, pred_mask).sum()
            union = np.logical_or(mask_true, pred_mask).sum()
            if union == 0:
                iou = 1.0 if intersection == 0 else 0.0
            else:
                iou = intersection / union
            ious.append(iou)
            processed_count += 1
        except Exception:
            continue

        if processed_count >= 50: break

    if len(ious) == 0:
        print(f"❌ '{TRAIN_IMG_DIR}' va '{TRAIN_MSK_DIR}' orasida mos juftliklar topilmadi.")
        return

    train_iou = np.mean(ious)
    
    print("\n" + "="*50)
    print(f"📊 DIAGNOSTIKA NATIJASI ({processed_count} ta rasmda):")
    print(f"  • Train datasetidagi IoU (namuna): {train_iou*100:.2f}%")
    if val_iou is not None:
        print(f"  • Colabdagi Val IoU: {val_iou*100:.2f}%")
    else:
        print(f"  • Colabdagi Val IoU: NOMA'LUM (Eski model format)")
    print("="*50)

    if val_iou is None:
        print("💡 IZOH: Bu model eski formatda saqlangan. Aniq tashxis qo'yish uchun V5 ni o'qitib ko'ring.")
    elif train_iou > 0.95 and val_iou < 0.85:
        print("🚨 XULOSA: Segmentation Model OVERFITTING holatida!")
        print("💡 Yechim: GridDropout yoki ElasticTransform darajasini biroz oshiring.")
    else:
        print("✅ XULOSA: Model sog'lom va to'qimalarni ajoyib farqlayapti (Generalization zo'r)!")

if __name__ == "__main__":
    check_segmentation_overfitting()
