import os
import shutil
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

EXPECTED_CLASSIFICATION_ROWS = 1276
EXPECTED_SEGMENTATION_MASKS = 200
ALLOWED_LABELS = set(range(12))

def validate_excel(excel_path):
    df = pd.read_excel(excel_path)
    required_cols = {"Image_ID", "Label"}
    if set(df.columns) != required_cols:
        raise ValueError(f"Excel columns must be exactly {sorted(required_cols)}; got {list(df.columns)}")
    if len(df) != EXPECTED_CLASSIFICATION_ROWS:
        raise ValueError(f"Expected {EXPECTED_CLASSIFICATION_ROWS} classification rows, got {len(df)}")
    if df["Image_ID"].duplicated().any():
        raise ValueError("Excel contains duplicate Image_ID values")
    invalid_labels = sorted(set(df["Label"].tolist()) - ALLOWED_LABELS)
    if invalid_labels:
        raise ValueError(f"Excel contains invalid labels: {invalid_labels}")

def validate_masks_dir(masks_dir):
    mask_paths = sorted(Path(masks_dir).glob("*.png"))
    if len(mask_paths) != EXPECTED_SEGMENTATION_MASKS:
        raise ValueError(f"Expected {EXPECTED_SEGMENTATION_MASKS} PNG masks, got {len(mask_paths)}")
    for mask_path in mask_paths:
        mask = np.array(Image.open(mask_path))
        unique_values = set(np.unique(mask).tolist())
        if not unique_values.issubset({0, 255}):
            raise ValueError(f"Mask {mask_path.name} is not binary PNG; found values {sorted(unique_values)}")

def main():
    parser = argparse.ArgumentParser(description="Prepare Hackathon Submission Folder")
    parser.add_argument("--team", type=str, required=True, help="Your Team Name")
    parser.add_argument("--cls_model", type=str, required=True, help="Path to best classification model")
    parser.add_argument("--seg_model", type=str, required=True, help="Path to best segmentation model")
    parser.add_argument("--excel_path", type=str, required=True, help="Path to generated results Excel")
    parser.add_argument("--masks_dir", type=str, required=True, help="Path to predicted masks folder")
    args = parser.parse_args()

    validate_excel(args.excel_path)
    validate_masks_dir(args.masks_dir)

    team_root = args.team
    if os.path.exists(team_root):
        shutil.rmtree(team_root)
    os.makedirs(team_root)

    # 1. Excel File
    shutil.copy(args.excel_path, os.path.join(team_root, f"{args.team} test_ground_truth.xlsx"))

    # 2. Masks Folder
    shutil.copytree(args.masks_dir, os.path.join(team_root, args.team))

    # 3. Models & Scripts Folder
    models_dir = os.path.join(team_root, "models")
    os.makedirs(os.path.join(models_dir, "classification"))
    os.makedirs(os.path.join(models_dir, "segmentation"))

    # Copy Scripts
    shutil.copy("classify.py", os.path.join(models_dir, "classification", "classify.py"))
    shutil.copy("segment.py", os.path.join(models_dir, "segmentation", "segment.py"))
    
    # Copy Models
    shutil.copy(args.cls_model, os.path.join(models_dir, "classification", "best_model.pth"))
    shutil.copy(args.seg_model, os.path.join(models_dir, "segmentation", "best_model.pth"))

    # Copy Requirements
    if os.path.exists("requirements.txt"):
        shutil.copy("requirements.txt", os.path.join(models_dir, "classification", "requirements.txt"))
        shutil.copy("requirements.txt", os.path.join(models_dir, "segmentation", "requirements.txt"))

    print(f"✅ Submission folder '{team_root}' created successfully!")
    print(f"📦 ZIP this folder and upload to Google Form.")

if __name__ == "__main__":
    main()
