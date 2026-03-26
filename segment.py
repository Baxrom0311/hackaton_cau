# ============================================================
# AI Healthcare Hackathon 2026 — Inference: Segmentation
# ============================================================
# ✅ Loads best model dynamically from args
# ✅ Applies Test-Time Augmentation (4x)
# ✅ Generates `[TeamName]` folder with predicted masks
# ============================================================

import os, argparse
import torch
import numpy as np
import cv2
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
from tqdm import tqdm

# ─── Post-proc & TTA ────────────────────────────────────────────────────────
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
def predict_mask_tta(model, img_np, img_size, device):
    h_orig, w_orig = img_np.shape[:2]
    normalize = A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

    def predict_single(image):
        t = normalize(image=image)["image"].unsqueeze(0).to(device)
        with torch.autocast(device_type="cuda" if "cuda" in device else "cpu", enabled=False):
            logits = model(t)
        return torch.sigmoid(logits.float()).squeeze(0).squeeze(0).cpu().numpy()

    preds = []
    preds.append(predict_single(img_np))
    preds.append(np.fliplr(predict_single(np.fliplr(img_np).copy())))
    preds.append(np.flipud(predict_single(np.flipud(img_np).copy())))
    preds.append(np.rot90(predict_single(np.rot90(img_np, 1).copy()), -1))

    avg_pred = np.mean(preds, axis=0)
    binary_mask = (avg_pred > 0.5).astype(np.uint8)
    binary_mask = cv2.resize(binary_mask, (w_orig, h_orig), interpolation=cv2.INTER_NEAREST)
    binary_mask = postprocess_mask(binary_mask)
    return binary_mask * 255

# ─── Main ───────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Segmentation Inference Script")
    parser.add_argument("--test_dir", type=str, required=True, help="Directory containing test images")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained PyTorch model (.pth)")
    parser.add_argument("--team", type=str, default="TeamName", help="Your team name for the output folder")
    args = parser.parse_args()

    if not os.path.exists(args.model_path):
        print(f"❌ Model topilmadi: {args.model_path}")
        return
    if not os.path.exists(args.test_dir):
        print(f"❌ Test papkasi topilmadi: {args.test_dir}")
        return

    output_dir = args.team
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
        
    print(f"🚀 Loading Segmentation Model on {device}: {args.model_path}")
    ckpt = torch.load(args.model_path, map_location=device, weights_only=False)
    
    model = smp.UnetPlusPlus(
        encoder_name=ckpt.get("encoder", "timm-efficientnet-b5"),
        encoder_weights=None,
        in_channels=3, classes=1, activation=None,
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    print(f"✅ Model loaded (Epoch: {ckpt.get('epoch', '?')}, IoU: {ckpt.get('val_iou', '?'):.4f})")

    os.makedirs(output_dir, exist_ok=True)
    files = sorted([f for f in os.listdir(args.test_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))])
    print(f"📸 Test images found: {len(files)}")

    for f in tqdm(files, desc="Predicting masks"):
        path = os.path.join(args.test_dir, f)
        img = np.array(Image.open(path).convert("RGB"))
        mask = predict_mask_tta(model, img, ckpt.get("img_size", 512), device)

        out_name = os.path.splitext(f)[0] + ".png"
        Image.fromarray(mask).save(os.path.join(output_dir, out_name))

    print(f"\n✅ Submission tayyor: {output_dir}/ papkasida ({len(files)} ta fayl)")

if __name__ == "__main__":
    main()
