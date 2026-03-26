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
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm


class TestDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.image_files = sorted([
            f for f in os.listdir(image_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        fname = self.image_files[idx]
        img = np.array(Image.open(os.path.join(self.image_dir, fname)).convert("RGB"))
        image_id = os.path.splitext(fname)[0]

        if self.transform:
            img = self.transform(image=img)["image"]

        return img, image_id


def get_test_transforms(img_size):
    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])


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

    # Dataset & Loader
    test_dataset = TestDataset(test_dir, transform=get_test_transforms(img_size))
    test_loader = DataLoader(
        test_dataset, batch_size=64,
        shuffle=False, num_workers=4, pin_memory=True
    )

    print(f"Total test images: {len(test_dataset)}")

    # Inference
    all_ids = []
    all_preds = []

    for images, image_ids in tqdm(test_loader, desc="Predicting"):
        images = images.to(device)
        outputs = model(images)
        preds = outputs.argmax(dim=1).cpu().numpy()

        all_ids.extend(image_ids)
        all_preds.extend(preds)

    # Save to Excel
    df = pd.DataFrame({
        "Image_ID": [int(x) for x in all_ids],
        "Label": [int(x) for x in all_preds]
    })
    df = df.sort_values("Image_ID").reset_index(drop=True)

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
