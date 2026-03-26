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
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
from tqdm import tqdm
import segmentation_models_pytorch as smp


def get_transforms(img_size):
    return A.Compose([
        A.Resize(img_size, img_size),
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

    transform = get_transforms(img_size)

    # Get test images
    test_images = sorted([
        f for f in os.listdir(test_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ])

    print(f"Total test images: {len(test_images)}")

    for fname in tqdm(test_images, desc="Generating masks"):
        # Read original image (keep original size)
        img_path = os.path.join(test_dir, fname)
        original_img = Image.open(img_path).convert("RGB")
        original_w, original_h = original_img.size

        img_np = np.array(original_img)

        # Transform for model input
        transformed = transform(image=img_np)
        input_tensor = transformed["image"].unsqueeze(0).to(device)

        # Predict
        output = model(input_tensor)
        pred = torch.sigmoid(output).squeeze().cpu().numpy()

        # Threshold to binary
        binary_mask = (pred > 0.5).astype(np.uint8) * 255

        # Resize mask back to original image size
        mask_pil = Image.fromarray(binary_mask, mode="L")
        mask_pil = mask_pil.resize((original_w, original_h), Image.NEAREST)

        # Save with same filename (PNG)
        out_name = os.path.splitext(fname)[0] + ".png"
        mask_pil.save(os.path.join(output_dir, out_name))

    print(f"\n✅ All {len(test_images)} masks saved to: {output_dir}")


if __name__ == "__main__":
    main()
