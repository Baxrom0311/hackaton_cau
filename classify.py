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
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
from tqdm import tqdm

# ─── TTA ────────────────────────────────────────────────────────────────────
def get_tta_transforms(img_size):
    base = [
        A.Resize(img_size, img_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ]
    return [
        A.Compose(base),
        A.Compose([A.HorizontalFlip(p=1.0)] + base),
        A.Compose([A.VerticalFlip(p=1.0)] + base),
        A.Compose([A.RandomRotate90(p=1.0)] + base),
    ]

@torch.no_grad()
def predict_tta(model, img_np, tta_tfms, device):
    all_probs = []
    for tfm in tta_tfms:
        tensor = tfm(image=img_np)["image"].unsqueeze(0).to(device)
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
    parser.add_argument("--team", type=str, default="TeamName", help="Your team name for the output Excel file")
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
    
    model = timm.create_model(ckpt.get("model_name", "tf_efficientnet_b5_ns"), pretrained=False, num_classes=ckpt.get("num_classes", 12))
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    print(f"✅ Model loaded (Epoch: {ckpt.get('epoch', '?')}, Acc: {ckpt.get('val_acc', '?'):.4f})")

    tta_tfms = get_tta_transforms(ckpt.get("img_size", 512))
    files = sorted([f for f in os.listdir(args.test_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))])
    print(f"📸 Test images found: {len(files)}")

    all_ids, all_preds = [], []
    for f in tqdm(files, desc="Predicting"):
        path = os.path.join(args.test_dir, f)
        img = np.array(Image.open(path).convert("RGB"))
        pred = predict_tta(model, img, tta_tfms, device)

        all_ids.append(os.path.splitext(f)[0])
        all_preds.append(pred)

    df = pd.DataFrame({"Image_ID": all_ids, "Label": all_preds})
    df = df.sort_values("Image_ID").reset_index(drop=True)
    df.to_excel(output_file, index=False)
    print(f"\n✅ Submission tayyor: {output_file} ({len(df)} ta bashorat)")

if __name__ == "__main__":
    main()
