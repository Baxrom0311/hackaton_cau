# ============================================================
# AI Healthcare Hackathon 2026 — Inference: Classification
# ============================================================
# Team: OxDEAD
# Model: EfficientNet-B2 (Noisy-Student) | IMG: 224x224
# Usage:
#   python OxDEADClass.py                          → Model haqida ma'lumot
#   python OxDEADClass.py --test_dir path/to/imgs  → Inference & Excel yaratish
# ============================================================

import os, sys, argparse
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import timm
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
from tqdm import tqdm

# ─── Auto-detect model file next to this script ─────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_FILE = os.path.join(SCRIPT_DIR, "OxDEADClassModel.pth")
TEAM_NAME = "OxDEAD"

def build_tta_images(img_resized):
    return [
        img_resized,
        np.fliplr(img_resized).copy(),
        np.flipud(img_resized).copy(),
        np.rot90(img_resized, 1).copy(),
    ]


def image_id_sort_key(value):
    value = str(value)
    return (0, int(value)) if value.isdigit() else (1, value)


# ─── Preprocessing ──────────────────────────────────────────────────────────
def robust_resize(img, sz):
    """Aspect-ratio preserving padding (Matches Training Pipeline)"""
    h, w = img.shape[:2]
    scale = sz / max(h, w)
    new_h, new_w = int(h * scale), int(w * scale)
    img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    
    pad_h = (sz - new_h) // 2
    pad_w = (sz - new_w) // 2
    img = cv2.copyMakeBorder(img, pad_h, sz - new_h - pad_h, pad_w, sz - new_w - pad_w, 
                            cv2.BORDER_CONSTANT, value=0)
    return img

# ─── TTA ────────────────────────────────────────────────────────────────────
@torch.no_grad()
def predict_tta(model, img_np, img_size, device):
    base_tfm = A.Compose([
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])
    img_resized = robust_resize(img_np, img_size)
    
    all_probs = []
    for aug_img in build_tta_images(img_resized):
        tensor = base_tfm(image=aug_img)["image"].unsqueeze(0).to(device)
        with torch.autocast(device_type="cuda" if "cuda" in device else "cpu", enabled=False):
            logits = model(tensor)
        all_probs.append(F.softmax(logits, dim=1))

    avg_probs = torch.stack(all_probs).mean(dim=0)
    return avg_probs.argmax(dim=1).item()

# ─── Model Info ─────────────────────────────────────────────────────────────
def show_model_info(ckpt):
    print("=" * 55)
    print("  🤖 OxDEAD Classification Model — Info")
    print("=" * 55)
    print(f"  Team:           {TEAM_NAME}")
    print(f"  Architecture:   {ckpt.get('model_name', 'N/A')}")
    print(f"  Image Size:     {ckpt.get('img_size', 'N/A')}x{ckpt.get('img_size', 'N/A')}")
    print(f"  Num Classes:    {ckpt.get('num_classes', 12)}")
    print(f"  Saved Epoch:    {ckpt.get('epoch', 'N/A')}")
    val_acc = ckpt.get('val_acc')
    print(f"  Val Accuracy:   {val_acc*100:.2f}%" if isinstance(val_acc, (int, float)) else "  Val Accuracy:   N/A")
    print(f"  Preprocessing:  Robust Padding (aspect-ratio preserving)")
    print(f"  TTA:            4x (Original + HFlip + VFlip + Rot90)")
    print(f"  Model File:     {MODEL_FILE}")
    print("=" * 55)
    print(f"\n  💡 Usage: python {os.path.basename(__file__)} --test_dir <path/to/test/images>")
    print(f"     Output: {TEAM_NAME} test_ground_truth.xlsx\n")

# ─── Main ───────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="OxDEAD Classification Inference")
    parser.add_argument("--test_dir", type=str, default=None, help="Directory containing test images")
    args = parser.parse_args()

    # Check model exists
    if not os.path.exists(MODEL_FILE):
        print(f"❌ Model topilmadi: {MODEL_FILE}")
        print(f"   OxDEADClassModel.pth faylini shu skript bilan bir papkaga qo'ying.")
        return

    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    ckpt = torch.load(MODEL_FILE, map_location=device, weights_only=False)

    # If no test_dir → show model info and exit
    if args.test_dir is None:
        show_model_info(ckpt)
        return

    # Validate test directory
    if not os.path.exists(args.test_dir):
        print(f"❌ Test papkasi topilmadi: {args.test_dir}")
        return

    # Load model
    num_classes = ckpt.get("num_classes", 12)
    model_name = ckpt.get("model_name", "tf_efficientnet_b2.ns_jft_in1k")
    img_size = ckpt.get("img_size", 224)

    print(f"🚀 Loading {model_name} on {device.upper()}...")
    model = timm.create_model(model_name, pretrained=False, num_classes=num_classes)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    
    val_acc = ckpt.get('val_acc')
    val_text = f"{val_acc*100:.2f}%" if isinstance(val_acc, (int, float)) else "N/A"
    print(f"✅ Model loaded (Epoch: {ckpt.get('epoch', '?')}, Acc: {val_text})")

    # Run inference
    files = sorted([f for f in os.listdir(args.test_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))])
    print(f"📸 Test images: {len(files)} | Preprocessing: Robust Padding ({img_size}x{img_size})")

    all_ids, all_preds = [], []
    for f in tqdm(files, desc="Predicting"):
        path = os.path.join(args.test_dir, f)
        try:
            img = np.array(Image.open(path).convert("RGB"))
            pred = predict_tta(model, img, img_size, device)
            if not 0 <= pred < num_classes:
                raise ValueError(f"Predicted label {pred} is outside 0..{num_classes - 1}")
        except Exception as e:
            raise RuntimeError(f"Error processing {f}: {e}") from e
        all_ids.append(os.path.splitext(f)[0])
        all_preds.append(pred)

    # Save Excel
    output_file = f"{TEAM_NAME} test_ground_truth.xlsx"
    rows = sorted(zip(all_ids, all_preds), key=lambda row: image_id_sort_key(row[0]))
    df = pd.DataFrame(rows, columns=["Image_ID", "Label"])
    df.to_excel(output_file, index=False)
    print(f"\n✅ Natija saqlandi: {output_file} ({len(df)} ta bashorat)")

if __name__ == "__main__":
    main()
