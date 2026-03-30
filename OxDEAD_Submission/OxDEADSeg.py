# ============================================================
# AI Healthcare Hackathon 2026 — Inference: Segmentation
# ============================================================
# Team: OxDEAD
# Model: UNet++ + EfficientNet-B2 | IMG: 224x224
# Usage:
#   python OxDEADSeg.py                          → Model haqida ma'lumot
#   python OxDEADSeg.py --test_dir path/to/imgs  → Inference & masklar yaratish
# ============================================================

import os, sys, argparse
import torch
import numpy as np
import cv2
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
from tqdm import tqdm

# ─── Auto-detect model file next to this script ─────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_FILE = os.path.join(SCRIPT_DIR, "OxDEADSegModel.pth")
TEAM_NAME = "OxDEAD"

# ─── Preprocessing ──────────────────────────────────────────────────────────
def robust_resize(img, sz, is_mask=False, return_meta=False):
    """Aspect-ratio preserving padding (Matches Training Pipeline)"""
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


def load_segmentation_model(ckpt, device):
    attention_candidates = [ckpt.get("decoder_attention_type")]
    if attention_candidates[0] is None:
        attention_candidates.append("scse")

    last_error = None
    for attention_type in attention_candidates:
        model = smp.UnetPlusPlus(
            encoder_name=ckpt.get("encoder", "efficientnet-b2"),
            encoder_weights=None,
            in_channels=3,
            classes=1,
            activation=None,
            decoder_attention_type=attention_type,
        )
        try:
            model.load_state_dict(ckpt["model_state_dict"])
            model.to(device).eval()
            return model, attention_type
        except RuntimeError as exc:
            last_error = exc

    raise last_error

# ─── TTA Prediction ─────────────────────────────────────────────────────────
@torch.no_grad()
def predict_mask_tta(model, img_np, img_size, device, threshold=0.5):
    h_orig, w_orig = img_np.shape[:2]
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
    preds.append(predict_single(img_resized))
    preds.append(np.fliplr(predict_single(np.fliplr(img_resized).copy())))
    preds.append(np.flipud(predict_single(np.flipud(img_resized).copy())))
    preds.append(np.rot90(predict_single(np.rot90(img_resized, 1).copy()), -1))

    avg_pred = np.mean(preds, axis=0).astype(np.float32)
    avg_pred = restore_original_mask(avg_pred, h_orig, w_orig, resize_meta)
    binary_mask = (avg_pred > threshold).astype(np.uint8)
    binary_mask = postprocess_mask(binary_mask)
    return binary_mask * 255

# ─── Model Info ─────────────────────────────────────────────────────────────
def show_model_info(ckpt):
    print("=" * 55)
    print("  🔬 OxDEAD Segmentation Model — Info")
    print("=" * 55)
    print(f"  Team:           {TEAM_NAME}")
    print(f"  Architecture:   UNet++ (smp)")
    print(f"  Encoder:        {ckpt.get('encoder', 'N/A')}")
    print(f"  Image Size:     {ckpt.get('img_size', 'N/A')}x{ckpt.get('img_size', 'N/A')}")
    print(f"  Saved Epoch:    {ckpt.get('epoch', 'N/A')}")
    val_iou = ckpt.get('val_iou')
    print(f"  Val IoU:        {val_iou*100:.2f}%" if isinstance(val_iou, (int, float)) else "  Val IoU:        N/A")
    best_th = ckpt.get('best_threshold', 0.5)
    print(f"  Best Threshold: {best_th:.2f}")
    print(f"  Preprocessing:  Robust Padding (aspect-ratio preserving)")
    print(f"  TTA:            4x (Original + HFlip + VFlip + Rot90)")
    print(f"  Postprocess:    Morphological Closing + Largest Component")
    print(f"  Model File:     {MODEL_FILE}")
    print("=" * 55)
    print(f"\n  💡 Usage: python {os.path.basename(__file__)} --test_dir <path/to/test/images>")
    print(f"     Output: {TEAM_NAME}/ folder with predicted PNG masks\n")

# ─── Main ───────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="OxDEAD Segmentation Inference")
    parser.add_argument("--test_dir", type=str, default=None, help="Directory containing test images")
    args = parser.parse_args()

    # Check model exists
    if not os.path.exists(MODEL_FILE):
        print(f"❌ Model topilmadi: {MODEL_FILE}")
        print(f"   OxDEADSegModel.pth faylini shu skript bilan bir papkaga qo'ying.")
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
    encoder_name = ckpt.get("encoder", "efficientnet-b2")
    img_size = ckpt.get("img_size", 224)
    if "best_threshold" not in ckpt:
        print("⚠️ best_threshold checkpointda yo'q, default 0.50 ishlatiladi.")
    best_th = ckpt.get("best_threshold", 0.5)

    print(f"🚀 Loading UNet++ ({encoder_name}) on {device.upper()}...")
    model, attention_type = load_segmentation_model(ckpt, device)

    val_iou = ckpt.get('val_iou')
    iou_text = f"{val_iou*100:.2f}%" if isinstance(val_iou, (int, float)) else "N/A"
    attention_text = attention_type or "none"
    print(
        f"✅ Model loaded (Epoch: {ckpt.get('epoch', '?')}, IoU: {iou_text}, "
        f"Threshold: {best_th:.2f}, Attention: {attention_text})"
    )

    # Run inference
    output_dir = TEAM_NAME
    os.makedirs(output_dir, exist_ok=True)
    files = sorted([f for f in os.listdir(args.test_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))])
    print(f"📸 Test images: {len(files)} | Preprocessing: Robust Padding ({img_size}x{img_size})")

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

    print(f"\n✅ Natija saqlandi: {output_dir}/ ({saved_masks} ta mask)")

if __name__ == "__main__":
    main()
