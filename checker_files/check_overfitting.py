# ============================================================
# Overfitting yoki Underfitting ekanligini Tekshiruvchi Skript
# ============================================================
# Qanday ishlashini bilish siri:
# 1. Agar Train_Acc = 99% va Val_Acc = 84% bo'lsa -> Bu OVERFITTING (Yodlab olish)
# 2. Agar Train_Acc = 85% va Val_Acc = 84% bo'lsa -> Bu UNDERFITTING (Model qiynalyapti, yodlolmayapti)
# ============================================================

import os
import torch
import torch.nn.functional as F
import timm
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
from tqdm import tqdm
import numpy as np

# Konfiguratsiya
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models/classification/best_model.pth")

def find_train_dir():
    candidates = [
        os.path.join(BASE_DIR, "dataset/classification/train"),
        os.path.join(BASE_DIR, "classification/train"),
    ]
    for c in candidates:
        if os.path.exists(c): return c
    return candidates[0] # Default

TRAIN_DIR = find_train_dir()

device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

def robust_resize(img, sz):
    """Aspect-ratio preserving padding (Matches V5 Training)"""
    h, w = img.shape[:2]
    scale = sz / max(h, w)
    new_h, new_w = int(h * scale), int(w * scale)
    img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    pad_h = (sz - new_h) // 2
    pad_w = (sz - new_w) // 2
    img = cv2.copyMakeBorder(img, pad_h, sz - new_h - pad_h, pad_w, sz - new_w - pad_w,
                            cv2.BORDER_CONSTANT, value=0)
    return img

def get_transforms(img_size):
    return A.Compose([
        # Resize is handled by robust_resize (aspect-ratio preserving padding)
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

def check_overfitting():
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Model topilmadi: {MODEL_PATH}")
        return
    if not os.path.exists(TRAIN_DIR):
        print(f"❌ '{TRAIN_DIR}' topilmadi. Colabdagi Train va Val natijalari bo'yicha tahlil qiling.")
        return

    print(f"🔍 {MODEL_PATH} tekshirilmoqda...")
    ckpt = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    
    # Check if it's a full checkpoint or just state_dict
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
        model_name = ckpt.get("model_name", "tf_efficientnet_b2.ns_jft_in1k")
        num_classes = ckpt.get("num_classes", 12)
        img_size = ckpt.get("img_size", 224)
        val_acc = ckpt.get("val_acc", None) # None for legacy models
    else:
        state_dict = ckpt
        # Fallback to V5 defaults
        model_name = "tf_efficientnet_b2.ns_jft_in1k"
        num_classes = 12
        img_size = 224
        val_acc = None # Legacy mode

    model = timm.create_model(model_name, pretrained=False, num_classes=num_classes)
    model.load_state_dict(state_dict)
    model.to(device).eval()
    
    tfm = get_transforms(img_size)

    # Taxminan 200 ta Train rasmini tasodifiy tekshiramiz
    all_classes = sorted([d for d in os.listdir(TRAIN_DIR) if os.path.isdir(os.path.join(TRAIN_DIR, d))], key=lambda x: int(x))
    
    correct = 0
    total = 0
    confidences = []

    for cls_id in all_classes:
        d = os.path.join(TRAIN_DIR, cls_id)
        files = os.listdir(d)[:20] # har bir klassdan 20 tadan (jami 240 ta rasm)
        for f in tqdm(files, desc=f"Tekshirish (Class {cls_id})", leave=False):
            img = np.array(Image.open(os.path.join(d, f)).convert("RGB"))
            img = robust_resize(img, img_size)  # Match V5 training preprocessing
            tensor = tfm(image=img)["image"].unsqueeze(0).to(device)
            with torch.no_grad():
                with torch.autocast(device_type="cuda" if "cuda" in device else "cpu", enabled=False):
                    logits = model(tensor)
                    probs = F.softmax(logits, dim=1)
            
            pred = probs.argmax(dim=1).item()
            confidences.append(probs.max().item())
            if pred == int(cls_id):
                correct += 1
            total += 1

    train_acc = correct / total
    avg_conf = np.mean(confidences)
    
    print("\n" + "="*50)
    print(f"📊 DIAGNOSTIKA NATIJASI:")
    print(f"  • Train datasetidagi natija (namunaviy): {train_acc*100:.2f}%")
    if val_acc is not None:
        print(f"  • Colabdagi Val natijasi: {val_acc*100:.2f}%")
    else:
        print(f"  • Colabdagi Val natijasi: NOMA'LUM (Eski model format)")
    print(f"  • O'rtacha ishonch (Confidence): {avg_conf*100:.2f}%")
    print("="*50)

    if val_acc is None:
        print("💡 IZOH: Bu model eski formatda saqlangan. Aniq tashxis qo'yish uchun V5 ni o'qitib ko'ring.")
    elif train_acc > 0.95 and val_acc < 0.88:
        print("🚨 XULOSA: Model aniq OVERFITTING (Yodlab olish) holatida!")
        print("💡 Yechim: Augmentatsiyani (Dropout, Mixup) oshirish va Epoch ni qisqartirish kerak.")
    elif train_acc < 0.90:
        print("⚠️ XULOSA: Model UNDERFITTING holatida! (U haligacha rasmlarni tushunib yetmagan).")
        print("💡 Yechim: Augmentatsiya (Mixup 0.7) juda og'ir kelgani uchun ko'rsatkichlar past chiqmoqda. \nMixup ni 0.2 ga tushirish yaxshilaydi (V4 versiyadagi kabi).")
    else:
        print("✅ XULOSA: Model sog'lom va Generalization zo'r!")

if __name__ == "__main__":
    check_overfitting()
