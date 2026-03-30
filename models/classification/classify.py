"""
AI in Healthcare Hackathon 2026 — Classification Inference Script
=================================================================
Loads the trained model and generates predictions for test images.
Outputs an Excel file with Image_ID and Label columns.

Usage:
    python classify.py <test_images_dir> <model_path>

Example:
    python classify.py classification/test models/classification/best_model.pth
"""

import os
import sys
import torch
import timm
import numpy as np
import pandas as pd
import cv2
import albumentations as A
import torch.nn.functional as F
from albumentations.pytorch import ToTensorV2
from PIL import Image
from tqdm import tqdm

def robust_resize(img, sz):
    h, w = img.shape[:2]
    scale = sz / max(h, w)
    new_h, new_w = int(h * scale), int(w * scale)
    img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    pad_h = (sz - new_h) // 2
    pad_w = (sz - new_w) // 2
    img = cv2.copyMakeBorder(img, pad_h, sz - new_h - pad_h, pad_w, sz - new_w - pad_w, cv2.BORDER_CONSTANT, value=0)
    return img


def get_test_transforms():
    return A.Compose([
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])


@torch.no_grad()
def predict_tta(model, image, img_size, device, transform):
    resized = robust_resize(image, img_size)
    tta_images = [
        resized,
        np.fliplr(resized).copy(),
        np.flipud(resized).copy(),
        np.rot90(resized, 1).copy(),
    ]
    probs = []
    for aug_img in tta_images:
        tensor = transform(image=aug_img)["image"].unsqueeze(0).to(device)
        logits = model(tensor)
        probs.append(F.softmax(logits, dim=1))
    return torch.stack(probs).mean(dim=0).argmax(dim=1).item()


def image_id_sort_key(value):
    value = str(value)
    return (0, int(value)) if value.isdigit() else (1, value)


@torch.no_grad()
def main():
    if len(sys.argv) < 3:
        print("Usage: python classify.py <test_images_dir> <model_path>")
        sys.exit(1)

    test_dir = sys.argv[1]
    model_path = sys.argv[2]
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

    print(f"Device: {device}")
    print(f"Test dir: {test_dir}")
    print(f"Model: {model_path}")

    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model_name = checkpoint["model_name"]
    num_classes = checkpoint["num_classes"]
    img_size = checkpoint["img_size"]

    # Build model and load weights
    model = timm.create_model(model_name, pretrained=False, num_classes=num_classes)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    transform = get_test_transforms()
    image_files = sorted(
        f for f in os.listdir(test_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    )
    print(f"Total test images: {len(image_files)}")

    # Inference
    all_ids = []
    all_preds = []

    for image_name in tqdm(image_files, desc="Predicting"):
        image_path = os.path.join(test_dir, image_name)
        image = np.array(Image.open(image_path).convert("RGB"))
        pred = predict_tta(model, image, img_size, device, transform)
        all_ids.append(os.path.splitext(image_name)[0])
        all_preds.append(pred)

    # Save to Excel
    rows = sorted(zip(all_ids, all_preds), key=lambda row: image_id_sort_key(row[0]))
    df = pd.DataFrame(rows, columns=["Image_ID", "Label"])

    output_file = "test_ground_truth.xlsx"
    df.to_excel(output_file, index=False)
    print(f"\n✅ Predictions saved to: {output_file}")
    print(f"   Total predictions: {len(df)}")
    print(f"   Label distribution:")
    for label in sorted(df["Label"].unique()):
        count = (df["Label"] == label).sum()
        print(f"     Class {label:>2d}: {count}")


if __name__ == "__main__":
    main()
