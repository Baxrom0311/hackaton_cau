"""
AI in Healthcare Hackathon 2026 — Segmentation Inference Script
===============================================================
Loads the trained segmentation model and generates binary masks.

Usage:
    python segment.py <test_images_dir> <model_path> <output_dir>
"""

import argparse
import os

import albumentations as A
import cv2
import numpy as np
import segmentation_models_pytorch as smp
import torch
from albumentations.pytorch import ToTensorV2
from PIL import Image
from tqdm import tqdm


def robust_resize(img, sz, is_mask=False, return_meta=False):
    h, w = img.shape[:2]
    scale = sz / max(h, w)
    new_h, new_w = int(h * scale), int(w * scale)
    interp = cv2.INTER_NEAREST if is_mask else cv2.INTER_LINEAR
    img = cv2.resize(img, (new_w, new_h), interpolation=interp)
    pad_h = (sz - new_h) // 2
    pad_w = (sz - new_w) // 2
    img = cv2.copyMakeBorder(
        img,
        pad_h,
        sz - new_h - pad_h,
        pad_w,
        sz - new_w - pad_w,
        cv2.BORDER_CONSTANT,
        value=0,
    )
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


def load_segmentation_model(checkpoint, device):
    attention_candidates = [checkpoint.get("decoder_attention_type")]
    if attention_candidates[0] is None:
        attention_candidates.append("scse")

    last_error = None
    for attention_type in attention_candidates:
        model = smp.UnetPlusPlus(
            encoder_name=checkpoint.get("encoder", "efficientnet-b2"),
            encoder_weights=None,
            in_channels=3,
            classes=1,
            activation=None,
            decoder_attention_type=attention_type,
        )
        try:
            model.load_state_dict(checkpoint["model_state_dict"])
            model.to(device).eval()
            return model, attention_type
        except RuntimeError as exc:
            last_error = exc

    raise last_error


@torch.no_grad()
def predict_mask_tta(model, img_np, img_size, device, threshold=0.5):
    orig_h, orig_w = img_np.shape[:2]
    img_resized, resize_meta = robust_resize(img_np, img_size, return_meta=True)
    base_tfm = A.Compose([
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

    def predict_single(image):
        tensor = base_tfm(image=image)["image"].unsqueeze(0).to(device)
        with torch.autocast(device_type="cuda" if "cuda" in device else "cpu", enabled=False):
            logits = model(tensor)
        return torch.sigmoid(logits.float()).squeeze(0).squeeze(0).cpu().numpy()

    preds = [
        predict_single(img_resized),
        np.fliplr(predict_single(np.fliplr(img_resized).copy())),
        np.flipud(predict_single(np.flipud(img_resized).copy())),
        np.rot90(predict_single(np.rot90(img_resized, 1).copy()), -1),
    ]
    avg_pred = np.mean(preds, axis=0).astype(np.float32)
    avg_pred = restore_original_mask(avg_pred, orig_h, orig_w, resize_meta)
    binary_mask = (avg_pred > threshold).astype(np.uint8)
    binary_mask = postprocess_mask(binary_mask)
    return binary_mask * 255


def main():
    parser = argparse.ArgumentParser(description="Bundled segmentation inference script")
    parser.add_argument("test_dir", help="Directory containing test images")
    parser.add_argument("model_path", help="Path to the trained segmentation checkpoint")
    parser.add_argument("output_dir", help="Directory where masks will be saved")
    args = parser.parse_args()

    if not os.path.isdir(args.test_dir):
        raise FileNotFoundError(f"Test dir not found: {args.test_dir}")
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model not found: {args.model_path}")

    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    checkpoint = torch.load(args.model_path, map_location=device, weights_only=False)
    img_size = int(checkpoint.get("img_size", 224))
    if "best_threshold" not in checkpoint:
        print("⚠️ best_threshold checkpointda yo'q, default 0.50 ishlatiladi.")
    best_threshold = float(checkpoint.get("best_threshold", 0.5))

    model, attention_type = load_segmentation_model(checkpoint, device)

    test_images = sorted(
        f for f in os.listdir(args.test_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    )

    print(f"Device: {device}")
    print(f"Total test images: {len(test_images)}")
    print(f"Preprocessing: robust letterbox ({img_size}x{img_size}) + 4x TTA")
    print(f"Decoder attention: {attention_type or 'none'}")

    saved_masks = 0
    for image_name in tqdm(test_images, desc="Generating masks"):
        img_path = os.path.join(args.test_dir, image_name)
        image = np.array(Image.open(img_path).convert("RGB"))
        mask = predict_mask_tta(model, image, img_size, device, threshold=best_threshold)
        out_name = os.path.splitext(image_name)[0] + ".png"
        Image.fromarray(mask, mode="L").save(os.path.join(args.output_dir, out_name))
        saved_masks += 1

    print(f"\n✅ All {saved_masks} masks saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
