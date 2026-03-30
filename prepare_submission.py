import argparse
import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

EXPECTED_CLASSIFICATION_ROWS = 1276
EXPECTED_SEGMENTATION_MASKS = 200
ALLOWED_LABELS = set(range(12))
CLASSIFICATION_REQUIREMENTS = """torch>=2.0
timm>=0.9
albumentations>=1.3
pandas>=2.0
openpyxl>=3.1
opencv-python>=4.8
numpy>=1.24
Pillow>=10.0
tqdm>=4.65
"""
SEGMENTATION_REQUIREMENTS = """torch>=2.0
segmentation-models-pytorch>=0.3
timm>=0.9
albumentations>=1.3
opencv-python>=4.8
numpy>=1.24
Pillow>=10.0
tqdm>=4.65
"""
BASE_DIR = Path(__file__).resolve().parent


def resolve_existing_dir(candidates, description):
    for candidate in candidates:
        if candidate and os.path.isdir(candidate):
            return Path(candidate).resolve()
    checked = "\n".join(f"  - {candidate}" for candidate in candidates if candidate)
    raise FileNotFoundError(f"{description} not found. Checked:\n{checked}")


def resolve_classification_test_dir(explicit_path):
    candidates = [explicit_path] if explicit_path else []
    candidates.extend(
        [
            BASE_DIR / "dataset" / "classification" / "test",
            BASE_DIR / "classification" / "test",
        ]
    )
    return resolve_existing_dir(candidates, "Classification test directory")


def resolve_segmentation_test_dir(explicit_path):
    candidates = [explicit_path] if explicit_path else []
    candidates.extend(
        [
            BASE_DIR / "dataset" / "Segmentation" / "testing" / "images",
            BASE_DIR / "dataset" / "segmentation" / "testing" / "images",
            BASE_DIR / "Segmentation" / "testing" / "images",
            BASE_DIR / "segmentation" / "testing" / "images",
        ]
    )
    return resolve_existing_dir(candidates, "Segmentation testing images directory")


def image_stems(image_dir):
    return sorted(
        os.path.splitext(path.name)[0]
        for path in Path(image_dir).iterdir()
        if path.suffix.lower() in {".png", ".jpg", ".jpeg"}
    )


def validate_excel(excel_path, classification_test_dir):
    df = pd.read_excel(excel_path)
    required_cols = {"Image_ID", "Label"}
    if set(df.columns) != required_cols:
        raise ValueError(f"Excel columns must be exactly {sorted(required_cols)}; got {list(df.columns)}")
    if df["Image_ID"].isna().any():
        raise ValueError("Excel contains empty Image_ID values")
    if df["Image_ID"].duplicated().any():
        raise ValueError("Excel contains duplicate Image_ID values")

    excel_ids = [str(x) for x in df["Image_ID"].tolist()]
    invalid_labels = sorted(set(int(x) for x in df["Label"].tolist()) - ALLOWED_LABELS)
    if invalid_labels:
        raise ValueError(f"Excel contains invalid labels: {invalid_labels}")

    expected_ids = image_stems(classification_test_dir)
    expected_id_set = set(expected_ids)
    excel_id_set = set(excel_ids)
    if excel_id_set != expected_id_set:
        missing = sorted(expected_id_set - excel_id_set)[:10]
        extra = sorted(excel_id_set - expected_id_set)[:10]
        raise ValueError(
            "Excel Image_ID values do not exactly match classification test images. "
            f"Missing sample: {missing} | Extra sample: {extra}"
        )
    if len(df) != len(expected_ids):
        raise ValueError(f"Expected {len(expected_ids)} classification rows, got {len(df)}")
    if len(df) != EXPECTED_CLASSIFICATION_ROWS:
        raise ValueError(f"Expected {EXPECTED_CLASSIFICATION_ROWS} classification rows, got {len(df)}")


def validate_masks_dir(masks_dir, segmentation_test_dir):
    mask_paths = sorted(Path(masks_dir).glob("*.png"))
    if len(mask_paths) != EXPECTED_SEGMENTATION_MASKS:
        raise ValueError(f"Expected {EXPECTED_SEGMENTATION_MASKS} PNG masks, got {len(mask_paths)}")

    expected_images = sorted(
        path for path in Path(segmentation_test_dir).iterdir()
        if path.suffix.lower() in {".png", ".jpg", ".jpeg"}
    )
    expected_names = {f"{image_path.stem}.png" for image_path in expected_images}
    actual_names = {mask_path.name for mask_path in mask_paths}
    if actual_names != expected_names:
        missing = sorted(expected_names - actual_names)[:10]
        extra = sorted(actual_names - expected_names)[:10]
        raise ValueError(
            "Mask filenames do not exactly match segmentation test image IDs. "
            f"Missing sample: {missing} | Extra sample: {extra}"
        )

    image_sizes = {}
    for image_path in expected_images:
        with Image.open(image_path) as image:
            image_sizes[image_path.stem] = image.size

    for mask_path in mask_paths:
        with Image.open(mask_path) as mask_image:
            mask = np.array(mask_image)
            if mask_image.size != image_sizes[mask_path.stem]:
                raise ValueError(
                    f"Mask {mask_path.name} size {mask_image.size} does not match source image size {image_sizes[mask_path.stem]}"
                )
        unique_values = set(np.unique(mask).tolist())
        if not unique_values.issubset({0, 255}):
            raise ValueError(f"Mask {mask_path.name} is not binary PNG; found values {sorted(unique_values)}")


def ensure_clean_dir(path):
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def write_requirements(path, content):
    path.write_text(content, encoding="utf-8")


def validate_checkpoint(model_path, task_name):
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    if "model_state_dict" not in checkpoint:
        raise ValueError(f"{task_name} checkpoint missing model_state_dict: {model_path}")
    if task_name == "Segmentation" and "best_threshold" not in checkpoint:
        raise ValueError(f"{task_name} checkpoint missing best_threshold: {model_path}")


def main():
    parser = argparse.ArgumentParser(description="Prepare guide-compliant hackathon submission folder")
    parser.add_argument("--team", type=str, required=True, help="Your Team Name")
    parser.add_argument("--cls_model", type=str, required=True, help="Path to best classification model")
    parser.add_argument("--seg_model", type=str, required=True, help="Path to best segmentation model")
    parser.add_argument("--excel_path", type=str, required=True, help="Path to generated results Excel")
    parser.add_argument("--masks_dir", type=str, required=True, help="Path to predicted masks folder")
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Build directory that will contain the official team folder (default: <team>_OfficialBuild)",
    )
    parser.add_argument("--cls_test_dir", type=str, default=None, help="Optional classification test directory override")
    parser.add_argument("--seg_test_dir", type=str, default=None, help="Optional segmentation test directory override")
    args = parser.parse_args()

    cls_model = Path(args.cls_model).resolve()
    seg_model = Path(args.seg_model).resolve()
    excel_path = Path(args.excel_path).resolve()
    masks_dir = Path(args.masks_dir).resolve()

    for required_path, label in [
        (cls_model, "Classification model"),
        (seg_model, "Segmentation model"),
        (excel_path, "Excel file"),
        (masks_dir, "Masks directory"),
    ]:
        if not required_path.exists():
            raise FileNotFoundError(f"{label} not found: {required_path}")

    cls_test_dir = resolve_classification_test_dir(args.cls_test_dir)
    seg_test_dir = resolve_segmentation_test_dir(args.seg_test_dir)

    validate_checkpoint(cls_model, "Classification")
    validate_checkpoint(seg_model, "Segmentation")
    validate_excel(excel_path, cls_test_dir)
    validate_masks_dir(masks_dir, seg_test_dir)

    build_root = Path(args.output_dir or f"{args.team}_OfficialBuild").resolve()
    team_root = build_root / args.team
    if team_root == masks_dir or team_root in masks_dir.parents:
        raise ValueError(
            f"Output folder {team_root} overlaps with masks source {masks_dir}. "
            "Use --output_dir to choose a different build directory."
        )

    ensure_clean_dir(team_root)

    excel_dest = team_root / f"{args.team} test_ground_truth.xlsx"
    masks_dest = team_root / args.team
    cls_dest_dir = team_root / "models" / "classification"
    seg_dest_dir = team_root / "models" / "segmentation"
    cls_dest_dir.mkdir(parents=True, exist_ok=True)
    seg_dest_dir.mkdir(parents=True, exist_ok=True)

    shutil.copy2(excel_path, excel_dest)
    masks_dest.mkdir(parents=True, exist_ok=True)
    for mask_path in sorted(masks_dir.glob("*.png")):
        shutil.copy2(mask_path, masks_dest / mask_path.name)
    shutil.copy2(BASE_DIR / "classify.py", cls_dest_dir / "classify.py")
    shutil.copy2(cls_model, cls_dest_dir / cls_model.name)
    write_requirements(cls_dest_dir / "requirements.txt", CLASSIFICATION_REQUIREMENTS)

    shutil.copy2(BASE_DIR / "segment.py", seg_dest_dir / "segment.py")
    shutil.copy2(seg_model, seg_dest_dir / seg_model.name)
    write_requirements(seg_dest_dir / "requirements.txt", SEGMENTATION_REQUIREMENTS)

    print(f"\n✅ Official submission folder ready: {team_root}")
    print("Contained structure:")
    print(f"  - {excel_dest.name}")
    print(f"  - {args.team}/ ({len(list(masks_dest.glob('*.png')))} masks)")
    print(f"  - models/classification/classify.py")
    print(f"  - models/classification/{cls_model.name}")
    print(f"  - models/classification/requirements.txt")
    print(f"  - models/segmentation/segment.py")
    print(f"  - models/segmentation/{seg_model.name}")
    print(f"  - models/segmentation/requirements.txt")


if __name__ == "__main__":
    main()
