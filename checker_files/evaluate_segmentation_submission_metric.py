#!/usr/bin/env python3

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

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_MODEL_PATH = os.path.join(BASE_DIR, "models", "segmentation", "best_model.pth")


def resolve_validation_dirs(img_dir, mask_dir):
    if img_dir and mask_dir:
        return img_dir, mask_dir

    candidates = [
        (
            os.path.join(BASE_DIR, "dataset", "Segmentation", "validation", "images"),
            os.path.join(BASE_DIR, "dataset", "Segmentation", "validation", "masks"),
        ),
        (
            os.path.join(BASE_DIR, "Segmentation", "validation", "images"),
            os.path.join(BASE_DIR, "Segmentation", "validation", "masks"),
        ),
    ]
    for candidate_img_dir, candidate_mask_dir in candidates:
        if os.path.isdir(candidate_img_dir) and os.path.isdir(candidate_mask_dir):
            return candidate_img_dir, candidate_mask_dir
    checked = "\n".join(f"  - {img}\n  - {mask}" for img, mask in candidates)
    raise FileNotFoundError(f"Validation images/masks split not found. Checked:\n{checked}")


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
        ).to(device)
        try:
            model.load_state_dict(ckpt["model_state_dict"])
            model.eval()
            return model, attention_type
        except RuntimeError as exc:
            last_error = exc

    raise last_error


@torch.no_grad()
def predict_probability_map(model, img_np, img_size, device):
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
        return torch.sigmoid(logits.float()).squeeze(0).squeeze(0).cpu().numpy()

    preds = [
        predict_single(img_resized),
        np.fliplr(predict_single(np.fliplr(img_resized).copy())),
        np.flipud(predict_single(np.flipud(img_resized).copy())),
        np.rot90(predict_single(np.rot90(img_resized, 1).copy()), -1),
    ]
    avg_pred = np.mean(preds, axis=0).astype(np.float32)
    return restore_original_mask(avg_pred, h_orig, w_orig, resize_meta)


def find_mask_path(mask_dir, image_name):
    stem = os.path.splitext(image_name)[0]
    candidates = [
        os.path.join(mask_dir, f"{stem}.png"),
        os.path.join(mask_dir, f"{stem}x.png"),
    ]
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    return None


def mean_iou(prob_maps, true_masks, threshold):
    scores = []
    for prob_map, true_mask in zip(prob_maps, true_masks):
        pred_mask = postprocess_mask((prob_map > threshold).astype(np.uint8))
        intersection = np.logical_and(pred_mask, true_mask).sum()
        union = np.logical_or(pred_mask, true_mask).sum()
        iou = 1.0 if union == 0 and intersection == 0 else (intersection / max(union, 1))
        scores.append(iou)
    return float(np.mean(scores)) if scores else 0.0


def main():
    parser = argparse.ArgumentParser(description="Evaluate segmentation checkpoint with submission-time preprocessing.")
    parser.add_argument("--model_path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--img_dir", default=None)
    parser.add_argument("--mask_dir", default=None)
    parser.add_argument("--write-threshold", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model not found: {args.model_path}")

    img_dir, mask_dir = resolve_validation_dirs(args.img_dir, args.mask_dir)
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    ckpt = torch.load(args.model_path, map_location=device, weights_only=False)
    model, attention_type = load_segmentation_model(ckpt, device)

    img_size = int(ckpt.get("img_size", 224))
    checkpoint_threshold = float(ckpt.get("best_threshold", 0.5))
    image_files = sorted([f for f in os.listdir(img_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))])
    if args.limit > 0:
        image_files = image_files[:args.limit]

    prob_maps, true_masks = [], []
    for image_name in tqdm(image_files, desc="Evaluating masks"):
        mask_path = find_mask_path(mask_dir, image_name)
        if mask_path is None:
            continue
        image = np.array(Image.open(os.path.join(img_dir, image_name)).convert("RGB"))
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        prob_maps.append(predict_probability_map(model, image, img_size, device))
        true_masks.append((mask > 127).astype(np.uint8))

    if not prob_maps:
        raise RuntimeError(f"No paired validation samples found in {img_dir} and {mask_dir}")

    thresholds = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]
    best_threshold = checkpoint_threshold
    best_iou = -1.0

    print(f"🔍 Model: {args.model_path}")
    print(f"📁 Validation images: {img_dir}")
    print(f"📁 Validation masks:  {mask_dir}")
    print(f"📦 Samples: {len(prob_maps)}")
    print(f"🧩 Decoder attention: {attention_type or 'none'}")

    for threshold in thresholds:
        score = mean_iou(prob_maps, true_masks, threshold)
        suffix = " (checkpoint)" if abs(threshold - checkpoint_threshold) < 1e-9 else ""
        print(f"  Threshold {threshold:.2f} -> IoU {score * 100:.2f}%{suffix}")
        if score > best_iou:
            best_iou = score
            best_threshold = threshold

    print("\n" + "=" * 60)
    print(f"Best submission-matched IoU: {best_iou * 100:.2f}% @ threshold {best_threshold:.2f}")
    print("=" * 60)

    if args.write_threshold:
        ckpt["best_threshold"] = best_threshold
        ckpt["threshold_iou"] = best_iou
        torch.save(ckpt, args.model_path)
        print(f"✅ Wrote best_threshold={best_threshold:.2f} into {args.model_path}")


if __name__ == "__main__":
    main()
