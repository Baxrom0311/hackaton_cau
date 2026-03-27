"""
AI in Healthcare Hackathon 2026 — Segmentation Inference Script
===============================================================
Loads the trained segmentation model and generates binary masks.

Usage:
    python segment.py <test_images_dir> <model_path> <output_dir>

Example:
    python segment.py Segmentation/testing/images models/segmentation/best_model.pth output_masks/
"""

import os
import sys
import torch
import numpy as np
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
from tqdm import tqdm
import segmentation_models_pytorch as smp

def robust_resize(img, sz, is_mask=False, return_meta=False):
    h, w = img.shape[:2]
    scale = sz / max(h, w)
    new_h, new_w = int(h * scale), int(w * scale)
    interp = cv2.INTER_NEAREST if is_mask else cv2.INTER_LINEAR
    img = cv2.resize(img, (new_w, new_h), interpolation=interp)
    pad_h = (sz - new_h) // 2
    pad_w = (sz - new_w) // 2
    img = cv2.copyMakeBorder(img, pad_h, sz - new_h - pad_h, pad_w, sz - new_w - pad_w, cv2.BORDER_CONSTANT, value=0)
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

def get_transforms():
    return A.Compose([
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])


@torch.no_grad()
def main():
    if len(sys.argv) < 4:
        print("Usage: python segment.py <test_images_dir> <model_path> <output_dir>")
        sys.exit(1)

    test_dir = sys.argv[1]
    model_path = sys.argv[2]
    output_dir = sys.argv[3]
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

    os.makedirs(output_dir, exist_ok=True)

    print(f"Device: {device}")
    print(f"Test dir: {test_dir}")
    print(f"Model: {model_path}")
    print(f"Output dir: {output_dir}")

    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    encoder = checkpoint["encoder"]
    img_size = checkpoint["img_size"]

    # Build model
    model = smp.UnetPlusPlus(
        encoder_name=encoder,
        encoder_weights=None,
        in_channels=3,
        classes=1,
        activation=None,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    transform = get_transforms()

    # Get test images
    test_images = sorted([
        f for f in os.listdir(test_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ])

    print(f"Total test images: {len(test_images)}")

    saved_masks = 0
    for fname in tqdm(test_images, desc="Generating masks"):
        # Read original image (keep original size)
        img_path = os.path.join(test_dir, fname)
        original_img = Image.open(img_path).convert("RGB")
        original_w, original_h = original_img.size

        img_np = np.array(original_img)

        # Transform for model input
        img_resized, resize_meta = robust_resize(img_np, img_size, return_meta=True)
        transformed = transform(image=img_resized)
        input_tensor = transformed["image"].unsqueeze(0).to(device)

        # Predict
        output = model(input_tensor)
        pred = torch.sigmoid(output).squeeze().cpu().numpy().astype(np.float32)
        pred = restore_original_mask(pred, original_h, original_w, resize_meta)

        # Threshold to binary
        binary_mask = (pred > 0.5).astype(np.uint8) * 255

        mask_pil = Image.fromarray(binary_mask, mode="L")

        # Save with same filename (PNG)
        out_name = os.path.splitext(fname)[0] + ".png"
        mask_pil.save(os.path.join(output_dir, out_name))
        saved_masks += 1

    print(f"\n✅ All {saved_masks} masks saved to: {output_dir}")


if __name__ == "__main__":
    main()
