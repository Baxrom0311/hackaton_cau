#!/usr/bin/env python3

import argparse
import json
import os
from collections import Counter, defaultdict

import numpy as np
from PIL import Image, ImageStat

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_TRAIN_DIR = os.path.join(BASE_DIR, "dataset", "classification", "train")


def list_items(train_dir):
    items = []
    for label in sorted((entry for entry in os.listdir(train_dir) if entry.isdigit()), key=int):
        cls_dir = os.path.join(train_dir, label)
        for image_name in sorted(os.listdir(cls_dir)):
            if image_name.lower().endswith((".png", ".jpg", ".jpeg")):
                items.append((int(label), os.path.join(cls_dir, image_name)))
    return items


def compute_stats(path):
    with Image.open(path).convert("RGB") as image:
        width, height = image.size
        stat = ImageStat.Stat(image)
        brightness = sum(stat.mean) / 3.0
        contrast = sum(stat.stddev) / 3.0
        gray = np.asarray(image.convert("L"), dtype=np.float32)

    lap = (
        -4.0 * gray[1:-1, 1:-1]
        + gray[:-2, 1:-1]
        + gray[2:, 1:-1]
        + gray[1:-1, :-2]
        + gray[1:-1, 2:]
    )
    blur_proxy = float(lap.var())
    return {
        "brightness": float(brightness),
        "contrast": float(contrast),
        "blur_proxy": blur_proxy,
        "width": int(width),
        "height": int(height),
    }


def build_feature_table(items):
    rows = []
    for label, path in items:
        stats = compute_stats(path)
        stats.update(
            {
                "label": label,
                "path": path,
            }
        )
        rows.append(stats)
    return rows


def summarize(rows):
    per_class = defaultdict(list)
    for row in rows:
        per_class[row["label"]].append(row)

    print("=" * 70)
    print("Classification train dataset analysis")
    print("=" * 70)
    print(f"Total images: {len(rows)}")
    print("Class counts:")
    for label in sorted(per_class):
        print(f"  Class {label}: {len(per_class[label])}")

    widths = [row["width"] for row in rows]
    heights = [row["height"] for row in rows]
    ratios = [max(row["width"], row["height"]) / max(1, min(row["width"], row["height"])) for row in rows]
    print("Geometry:")
    print(f"  Width  min/median/max: {min(widths)} / {int(np.median(widths))} / {max(widths)}")
    print(f"  Height min/median/max: {min(heights)} / {int(np.median(heights))} / {max(heights)}")
    print(f"  Aspect ratio median/max: {np.median(ratios):.3f} / {max(ratios):.3f}")

    print("Per-class image style stats:")
    for label in sorted(per_class):
        cls_rows = per_class[label]
        brightness = [row["brightness"] for row in cls_rows]
        contrast = [row["contrast"] for row in cls_rows]
        blur = [row["blur_proxy"] for row in cls_rows]
        print(
            f"  Class {label}: "
            f"brightness={np.mean(brightness):.1f}, "
            f"contrast={np.mean(contrast):.1f}, "
            f"blur={np.median(blur):.1f}"
        )


def exact_duplicates(items):
    import hashlib

    by_hash = defaultdict(list)
    for label, path in items:
        with open(path, "rb") as handle:
            digest = hashlib.md5(handle.read()).hexdigest()
        by_hash[digest].append((label, path))
    return [cluster for cluster in by_hash.values() if len(cluster) > 1]


def attach_style_buckets(rows):
    feature_names = ["brightness", "contrast", "blur_proxy"]
    bounds = {}
    for name in feature_names:
        values = np.array([row[name] for row in rows], dtype=np.float32)
        bounds[name] = np.quantile(values, [1.0 / 3.0, 2.0 / 3.0]).tolist()

    for row in rows:
        parts = []
        for name in feature_names:
            bucket = int(np.searchsorted(bounds[name], row[name], side="right"))
            row[f"{name}_bucket"] = bucket
            parts.append(str(bucket))
        row["style_bucket"] = "-".join(parts)
    return bounds


def build_hard_manifest(rows, train_dir, output_path, val_fraction):
    feature_names = ["brightness", "contrast", "blur_proxy"]
    global_mean = {
        name: float(np.mean([row[name] for row in rows]))
        for name in feature_names
    }
    global_std = {
        name: float(np.std([row[name] for row in rows]) + 1e-6)
        for name in feature_names
    }

    by_class = defaultdict(list)
    for row in rows:
        by_class[row["label"]].append(row)

    train_rows = []
    val_rows = []
    holdout_summary = {}

    for label in sorted(by_class):
        cls_rows = by_class[label]
        target = max(1, int(round(len(cls_rows) * val_fraction)))
        cls_mean = {
            name: float(np.mean([row[name] for row in cls_rows]))
            for name in feature_names
        }

        grouped = defaultdict(list)
        for row in cls_rows:
            grouped[row["style_bucket"]].append(row)

        bucket_infos = []
        for bucket_name, bucket_rows in grouped.items():
            centroid = {
                name: float(np.mean([row[name] for row in bucket_rows]))
                for name in feature_names
            }
            distance = 0.0
            for name in feature_names:
                distance += abs(centroid[name] - cls_mean[name]) / global_std[name]
            bucket_infos.append(
                {
                    "bucket": bucket_name,
                    "rows": bucket_rows,
                    "count": len(bucket_rows),
                    "distance": distance,
                }
            )

        bucket_infos.sort(key=lambda row: (-row["distance"], -row["count"], row["bucket"]))

        picked = []
        picked_count = 0
        for info in bucket_infos:
            if picked_count >= target:
                break
            picked.append(info)
            picked_count += info["count"]

        picked_buckets = {info["bucket"] for info in picked}
        holdout_summary[label] = [(info["bucket"], info["count"], round(info["distance"], 3)) for info in picked]
        for row in cls_rows:
            entry = {
                "path": os.path.relpath(row["path"], train_dir),
                "label": int(label),
                "style_bucket": row["style_bucket"],
            }
            if row["style_bucket"] in picked_buckets:
                val_rows.append(entry)
            else:
                train_rows.append(entry)

    payload = {
        "root_dir": train_dir,
        "method": "style_outlier_holdout_v1",
        "val_fraction": val_fraction,
        "train_count": len(train_rows),
        "val_count": len(val_rows),
        "style_bucket_features": feature_names,
        "train": train_rows,
        "val": val_rows,
        "holdout_summary": holdout_summary,
    }

    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    print(f"Hard split manifest written: {output_path}")
    print(f"  Train count: {len(train_rows)}")
    print(f"  Val count:   {len(val_rows)}")
    print("  Holdout buckets per class:")
    for label in sorted(holdout_summary):
        print(f"    Class {label}: {holdout_summary[label]}")


def main():
    parser = argparse.ArgumentParser(description="Analyze classification train dataset and optionally build a harder validation manifest.")
    parser.add_argument("--train_dir", default=DEFAULT_TRAIN_DIR)
    parser.add_argument("--write_hard_manifest", default=None)
    parser.add_argument("--val_fraction", type=float, default=0.10)
    args = parser.parse_args()

    if not os.path.isdir(args.train_dir):
        raise FileNotFoundError(f"Train dir not found: {args.train_dir}")

    items = list_items(args.train_dir)
    rows = build_feature_table(items)
    summarize(rows)

    duplicates = exact_duplicates(items)
    print("Exact duplicate clusters:")
    print(f"  Cluster count: {len(duplicates)}")
    print(f"  Duplicate images total: {sum(len(cluster) for cluster in duplicates)}")
    for cluster in duplicates[:10]:
        print(f"  {cluster}")

    bounds = attach_style_buckets(rows)
    bucket_counts = Counter(row["style_bucket"] for row in rows)
    print("Style bucket quantile bounds:")
    for feature_name, feature_bounds in bounds.items():
        print(f"  {feature_name}: {feature_bounds}")
    print("Top style buckets:")
    for bucket_name, count in bucket_counts.most_common(10):
        print(f"  {bucket_name}: {count}")

    if args.write_hard_manifest:
        build_hard_manifest(rows, args.train_dir, args.write_hard_manifest, args.val_fraction)


if __name__ == "__main__":
    main()
