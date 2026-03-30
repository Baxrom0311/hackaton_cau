#!/usr/bin/env python3

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

EXPECTED_CLASSIFICATION_ROWS = 1276
EXPECTED_SEGMENTATION_MASKS = 200
ALLOWED_LABELS = set(range(12))
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_SUFFIXES = {".pth", ".pt", ".ckpt", ".bin"}


def resolve_existing_dir(candidates, description):
    for candidate in candidates:
        if candidate and os.path.isdir(candidate):
            return Path(candidate).resolve()
    checked = "\n".join(f"  - {candidate}" for candidate in candidates if candidate)
    raise FileNotFoundError(f"{description} not found. Checked:\n{checked}")


def resolve_submission_root(submission_dir, team):
    candidate = Path(submission_dir).resolve()
    if (candidate / f"{team} test_ground_truth.xlsx").exists():
        return candidate
    nested = candidate / team
    if (nested / f"{team} test_ground_truth.xlsx").exists():
        return nested
    return candidate


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


def find_model_files(model_dir):
    return sorted(
        path for path in Path(model_dir).iterdir()
        if path.is_file() and path.suffix.lower() in MODEL_SUFFIXES
    )


def validate_excel(excel_path, classification_test_dir, errors):
    print(f"\n📊 Excel: {excel_path.name}")
    df = pd.read_excel(excel_path)
    print(f"   Rows: {len(df)}")
    print(f"   Columns: {list(df.columns)}")

    required_cols = {"Image_ID", "Label"}
    if set(df.columns) != required_cols:
        errors.append(f"Excel columns must be exactly {sorted(required_cols)}; got {list(df.columns)}")
        return
    if df["Image_ID"].isna().any():
        errors.append("Excel contains empty Image_ID values")
    if df["Image_ID"].duplicated().any():
        errors.append("Excel contains duplicate Image_ID values")

    try:
        labels = [int(x) for x in df["Label"].tolist()]
    except Exception as exc:
        errors.append(f"Excel labels are not all integers: {exc}")
        labels = []
    invalid_labels = sorted(set(labels) - ALLOWED_LABELS)
    if invalid_labels:
        errors.append(f"Excel contains invalid labels: {invalid_labels}")

    expected_ids = image_stems(classification_test_dir)
    excel_ids = [str(x) for x in df["Image_ID"].tolist()]
    if set(excel_ids) != set(expected_ids):
        missing = sorted(set(expected_ids) - set(excel_ids))[:10]
        extra = sorted(set(excel_ids) - set(expected_ids))[:10]
        errors.append(
            "Excel Image_ID values do not exactly match classification test images. "
            f"Missing sample: {missing} | Extra sample: {extra}"
        )
    if len(df) != len(expected_ids):
        errors.append(f"Excel row count {len(df)} does not match local classification test count {len(expected_ids)}")
    if len(df) != EXPECTED_CLASSIFICATION_ROWS:
        errors.append(f"Excel row count {len(df)} does not match expected {EXPECTED_CLASSIFICATION_ROWS}")


def validate_masks(mask_dir, segmentation_test_dir, errors):
    print(f"\n🗂️ Masks folder: {mask_dir.name}")
    mask_paths = sorted(mask_dir.glob("*.png"))
    extra_files = sorted(path.name for path in mask_dir.iterdir() if path.is_file() and path.suffix.lower() != ".png")
    print(f"   PNG masks: {len(mask_paths)}")
    if len(mask_paths) != EXPECTED_SEGMENTATION_MASKS:
        errors.append(f"Segmentation folder has {len(mask_paths)} masks, expected {EXPECTED_SEGMENTATION_MASKS}")
    if extra_files:
        errors.append(f"Masks folder contains non-PNG files: {extra_files[:10]}")

    test_images = sorted(
        path for path in Path(segmentation_test_dir).iterdir()
        if path.suffix.lower() in {".png", ".jpg", ".jpeg"}
    )
    expected_names = {f"{image_path.stem}.png" for image_path in test_images}
    actual_names = {mask_path.name for mask_path in mask_paths}
    if actual_names != expected_names:
        missing = sorted(expected_names - actual_names)[:10]
        extra = sorted(actual_names - expected_names)[:10]
        errors.append(
            "Mask filenames do not exactly match segmentation test image IDs. "
            f"Missing sample: {missing} | Extra sample: {extra}"
        )

    expected_sizes = {}
    for image_path in test_images:
        with Image.open(image_path) as image:
            expected_sizes[image_path.stem] = image.size

    for mask_path in mask_paths:
        with Image.open(mask_path) as mask_image:
            mask_array = np.array(mask_image)
            if mask_image.size != expected_sizes.get(mask_path.stem):
                errors.append(
                    f"Mask {mask_path.name} size {mask_image.size} does not match source image size {expected_sizes.get(mask_path.stem)}"
                )
            unique_values = set(np.unique(mask_array).tolist())
            if not unique_values.issubset({0, 255}):
                errors.append(f"Mask {mask_path.name} is not binary; found values {sorted(unique_values)}")


def validate_model_bundle(task_name, model_dir, required_script, errors):
    print(f"\n🧠 {task_name}: {model_dir}")
    script_path = model_dir / required_script
    req_path = model_dir / "requirements.txt"
    if not script_path.exists():
        errors.append(f"Missing script: {script_path}")
    if not req_path.exists():
        errors.append(f"Missing requirements.txt: {req_path}")

    model_files = find_model_files(model_dir)
    if not model_files:
        errors.append(f"No saved model file found in {model_dir}")
        return None

    print(f"   Model files: {[path.name for path in model_files]}")
    return model_files[0]


def inspect_checkpoint(model_path, task_name, errors):
    ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
    if "model_state_dict" not in ckpt:
        errors.append(f"{task_name} checkpoint missing model_state_dict: {model_path}")
    if task_name == "Segmentation" and "best_threshold" not in ckpt:
        errors.append(f"Segmentation checkpoint missing best_threshold: {model_path}")
    return ckpt


def main():
    parser = argparse.ArgumentParser(description="Validate official hackathon submission folder")
    parser.add_argument("--team", default="OxDEAD", help="Team name")
    parser.add_argument("--submission_dir", default=None, help="Path to the team submission folder or its parent")
    parser.add_argument("--cls_test_dir", default=None, help="Optional classification test directory override")
    parser.add_argument("--seg_test_dir", default=None, help="Optional segmentation test directory override")
    args = parser.parse_args()

    submission_input = Path(args.submission_dir or (BASE_DIR / f"{args.team}_OfficialBuild")).resolve()
    submission_root = resolve_submission_root(submission_input, args.team)
    if not submission_root.exists():
        raise FileNotFoundError(f"Submission directory not found: {submission_root}")

    cls_test_dir = resolve_classification_test_dir(args.cls_test_dir)
    seg_test_dir = resolve_segmentation_test_dir(args.seg_test_dir)
    errors = []

    print("=" * 60)
    print("  OFFICIAL SUBMISSION CHECK")
    print("=" * 60)
    print(f"Submission root: {submission_root}")
    print(f"Classification test dir: {cls_test_dir}")
    print(f"Segmentation test dir:   {seg_test_dir}")

    excel_path = submission_root / f"{args.team} test_ground_truth.xlsx"
    masks_dir = submission_root / args.team
    cls_dir = submission_root / "models" / "classification"
    seg_dir = submission_root / "models" / "segmentation"

    if not excel_path.exists():
        errors.append(f"Missing Excel file: {excel_path}")
    else:
        validate_excel(excel_path, cls_test_dir, errors)

    if not masks_dir.exists():
        errors.append(f"Missing masks folder: {masks_dir}")
    else:
        validate_masks(masks_dir, seg_test_dir, errors)

    cls_model = validate_model_bundle("Classification", cls_dir, "classify.py", errors) if cls_dir.exists() else None
    if not cls_dir.exists():
        errors.append(f"Missing classification model folder: {cls_dir}")

    seg_model = validate_model_bundle("Segmentation", seg_dir, "segment.py", errors) if seg_dir.exists() else None
    if not seg_dir.exists():
        errors.append(f"Missing segmentation model folder: {seg_dir}")

    if cls_model is not None:
        inspect_checkpoint(cls_model, "Classification", errors)
    if seg_model is not None:
        inspect_checkpoint(seg_model, "Segmentation", errors)

    print(f"\n{'=' * 60}")
    if errors:
        print(f"❌ {len(errors)} issue(s) found:")
        for error in errors:
            print(f" - {error}")
        raise SystemExit(1)
    print("✅ Submission folder matches the local official-format checks.")


if __name__ == "__main__":
    main()
