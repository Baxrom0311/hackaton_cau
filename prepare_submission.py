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

    team_root = f"{args.team}_Submission"
    if os.path.exists(team_root):
        shutil.rmtree(team_root)
    os.makedirs(team_root)

    # 1. Excel File
    excel_dest = os.path.join(team_root, f"{args.team} test_ground_truth.xlsx")
    shutil.copy(args.excel_path, excel_dest)

    # 2. Masks Folder
    masks_folder_name = f"{args.team} masks"
    dest_masks_dir = os.path.join(team_root, masks_folder_name)
    if os.path.abspath(args.masks_dir) != os.path.abspath(dest_masks_dir):
        shutil.copytree(args.masks_dir, dest_masks_dir)
    
    # Compress masks folder to TeamName masks.zip
    shutil.make_archive(dest_masks_dir, 'zip', dest_masks_dir)
    shutil.rmtree(dest_masks_dir) # Remove the unzipped folder to leave only the .zip

    # 3. Code Python Scripts
    class_py_dest = os.path.join(team_root, f"{args.team}Class.py")
    seg_py_dest = os.path.join(team_root, f"{args.team}Seg.py")
    shutil.copy("classify.py", class_py_dest)
    shutil.copy("segment.py", seg_py_dest)

    # 4. Models
    cls_ext = os.path.splitext(args.cls_model)[1]
    seg_ext = os.path.splitext(args.seg_model)[1]
    class_model_dest = os.path.join(team_root, f"{args.team}ClassModel{cls_ext}")
    seg_model_dest = os.path.join(team_root, f"{args.team}SegModel{seg_ext}")
    
    shutil.copy(args.cls_model, class_model_dest)
    shutil.copy(args.seg_model, seg_model_dest)

    print(f"\n✅ SUBMISSION FOLDER '{team_root}' TAYYORLANDI!")
    print(f"Ushbu papka ichida roppa-rosa Google form suragan 6 ta fayl joylashgan:")
    print(f" 1. {os.path.basename(excel_dest)}")
    print(f" 2. {masks_folder_name}.zip")
    print(f" 3. {os.path.basename(class_py_dest)}")
    print(f" 4. {os.path.basename(seg_py_dest)}")
    print(f" 5. {os.path.basename(class_model_dest)}")
    print(f" 6. {os.path.basename(seg_model_dest)}")
    print("\n📦 Bularni to'g'ridan to'g'ri bittalab Google Form'ga yuklang!")

if __name__ == "__main__":
    main()
