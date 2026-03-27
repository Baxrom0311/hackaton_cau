# ============================================================
# AI Healthcare Hackathon 2026 — Inference: Classification
# ============================================================
# ✅ Loads best model dynamically from args
# ✅ Applies Test-Time Augmentation (4x)
# ✅ Generates `[TeamName] test_ground_truth.xlsx`
# ============================================================

import os, argparse
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

# ─── Preprocessing ──────────────────────────────────────────────────────────
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

# ─── TTA ────────────────────────────────────────────────────────────────────
def get_tta_transforms():
    """Returns basic augmentations for TTA (Resize is handled by robust_resize)"""
    return [
        None, # Original
        A.HorizontalFlip(p=1.0),
        A.VerticalFlip(p=1.0),
        A.RandomRotate90(p=1.0)
    ]

@torch.no_grad()
def predict_tta(model, img_np, img_size, device):
    all_probs = []
    
    # Base normalization/tensorization
    base_tfm = A.Compose([
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

    # 1. Resize once using robust logic
    img_resized = robust_resize(img_np, img_size)
    
    # 2. Apply TTA
    tta_ops = get_tta_transforms()
    for op in tta_ops:
        aug_img = op(image=img_resized)["image"] if op else img_resized
        tensor = base_tfm(image=aug_img)["image"].unsqueeze(0).to(device)
        
        with torch.autocast(device_type="cuda" if "cuda" in device else "cpu", enabled=False):
            logits = model(tensor)
        probs = F.softmax(logits, dim=1)
        all_probs.append(probs)

    avg_probs = torch.stack(all_probs).mean(dim=0)
    return avg_probs.argmax(dim=1).item()

# ─── Main ───────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Classification Inference Script")
    parser.add_argument("--test_dir", type=str, required=True, help="Directory containing test images")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained PyTorch model (.pth)")
    parser.add_argument("--team", type=str, default="Baxrom", help="Your team name for the output Excel file")
    args = parser.parse_args()

    if not os.path.exists(args.model_path):
        print(f"❌ Model topilmadi: {args.model_path}")
        return
    if not os.path.exists(args.test_dir):
        print(f"❌ Test papkasi topilmadi: {args.test_dir}")
        return

    output_file = f"{args.team} test_ground_truth.xlsx"
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
        
    print(f"🚀 Loading Classification Model on {device}: {args.model_path}")
    ckpt = torch.load(args.model_path, map_location=device, weights_only=False)
    num_classes = ckpt.get("num_classes", 12)
    val_acc = ckpt.get("val_acc")

    model = timm.create_model(ckpt.get("model_name", "tf_efficientnet_b2.ns_jft_in1k"), pretrained=False, num_classes=num_classes)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    val_acc_text = f"{val_acc:.4f}" if isinstance(val_acc, (int, float)) else "N/A"
    print(f"✅ Model loaded (Epoch: {ckpt.get('epoch', '?')}, Acc: {val_acc_text})")

    img_size = ckpt.get("img_size", 224)
    files = sorted([f for f in os.listdir(args.test_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))])
    print(f"📸 Test images found: {len(files)} | Preprocessing: Robust Padding ({img_size}x{img_size})")

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

    df = pd.DataFrame({"Image_ID": all_ids, "Label": all_preds})
    df = df.sort_values("Image_ID").reset_index(drop=True)
    df.to_excel(output_file, index=False)
    print(f"\n✅ Submission tayyor: {output_file} ({len(df)} ta bashorat)")

if __name__ == "__main__":
    main()
