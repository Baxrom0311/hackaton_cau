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

# ─── Preprocessing ──────────────────────────────────────────────────────────
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
def predict_mask_tta(model, img_np, img_size, device, threshold=0.5):
    h_orig, w_orig = img_np.shape[:2]
    
    # 1. Resize once using robust logic
    img_resized, resize_meta = robust_resize(img_np, img_size, return_meta=True)
    
    base_tfm = A.Compose([
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

    def predict_single(image):
        t = base_tfm(image=image)["image"].unsqueeze(0).to(device)
        with torch.autocast(device_type="cuda" if "cuda" in device else "cpu", enabled=False):
            logits = model(t)
        return torch.sigmoid(logits.float()).squeeze(0).squeeze(0).cpu().numpy()

    preds = []
    # TTA: Original, Horz, Vert, Rot90
    preds.append(predict_single(img_resized))
    preds.append(np.fliplr(predict_single(np.fliplr(img_resized).copy())))
    preds.append(np.flipud(predict_single(np.flipud(img_resized).copy())))
    preds.append(np.rot90(predict_single(np.rot90(img_resized, 1).copy()), -1))

    avg_pred = np.mean(preds, axis=0).astype(np.float32)
    avg_pred = restore_original_mask(avg_pred, h_orig, w_orig, resize_meta)
    binary_mask = (avg_pred > threshold).astype(np.uint8)
    binary_mask = postprocess_mask(binary_mask)
    return binary_mask * 255

# ─── Main ───────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Segmentation Inference Script")
    parser.add_argument("--test_dir", type=str, required=True, help="Directory containing test images")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained PyTorch model (.pth)")
    parser.add_argument("--team", type=str, default="Baxrom", help="Your team name for the output folder")
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
        encoder_name=ckpt.get("encoder", "efficientnet-b2"),
        encoder_weights=None,
        in_channels=3, classes=1, activation=None,
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    val_iou = ckpt.get("val_iou")
    best_th = ckpt.get("best_threshold", 0.5)
    val_iou_text = f"{val_iou:.4f}" if isinstance(val_iou, (int, float)) else "N/A"
    print(f"✅ Model loaded (Epoch: {ckpt.get('epoch', '?')}, IoU: {val_iou_text}, Threshold: {best_th:.2f})")

    os.makedirs(output_dir, exist_ok=True)
    img_size = ckpt.get("img_size", 224)
    files = sorted([f for f in os.listdir(args.test_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))])
    print(f"📸 Test images found: {len(files)} | Preprocessing: Robust Padding ({img_size}x{img_size})")

    saved_masks = 0
    for f in tqdm(files, desc="Predicting masks"):
        path = os.path.join(args.test_dir, f)
        try:
            img = np.array(Image.open(path).convert("RGB"))
            mask = predict_mask_tta(model, img, img_size, device, threshold=best_th)
    
            out_name = os.path.splitext(f)[0] + ".png"
            Image.fromarray(mask).save(os.path.join(output_dir, out_name))
            saved_masks += 1
        except Exception as e:
            raise RuntimeError(f"Error processing {f}: {e}") from e

    print(f"\n✅ Submission tayyor: {output_dir}/ papkasida ({saved_masks} ta fayl)")

if __name__ == "__main__":
    main()
