#!/usr/bin/env python3

import argparse
import json
import os
import random
from collections import Counter, defaultdict

import albumentations as A
import cv2
import numpy as np
import timm
import torch
import torch.nn.functional as F
from albumentations.pytorch import ToTensorV2
from PIL import Image
from tqdm import tqdm

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_MODEL_PATH = os.path.join(BASE_DIR, "models", "classification", "best_model.pth")
DEFAULT_TRAIN_DIR = os.path.join(BASE_DIR, "dataset", "classification", "train")


def robust_resize(img, sz):
    h, w = img.shape[:2]
    scale = sz / max(h, w)
    new_h, new_w = int(h * scale), int(w * scale)
    img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    pad_h = (sz - new_h) // 2
    pad_w = (sz - new_w) // 2
    return cv2.copyMakeBorder(
        img,
        pad_h,
        sz - new_h - pad_h,
        pad_w,
        sz - new_w - pad_w,
        cv2.BORDER_CONSTANT,
        value=0,
    )


def collect_items(root_dir):
    items = []
    for label in sorted((entry for entry in os.listdir(root_dir) if entry.isdigit()), key=int):
        cls_dir = os.path.join(root_dir, label)
        for img_name in sorted(os.listdir(cls_dir)):
            if img_name.lower().endswith((".png", ".jpg", ".jpeg")):
                items.append((os.path.join(cls_dir, img_name), int(label)))
    return items


def build_split(root_dir, val_fraction, seed):
    rng = random.Random(seed)
    by_class = defaultdict(list)
    for item in collect_items(root_dir):
        by_class[item[1]].append(item)

    train_items, val_items = [], []
    for cls_id in sorted(by_class):
        cls_items = list(by_class[cls_id])
        rng.shuffle(cls_items)
        split = int((1.0 - val_fraction) * len(cls_items))
        train_items.extend(cls_items[:split])
        val_items.extend(cls_items[split:])

    rng.shuffle(train_items)
    rng.shuffle(val_items)
    return train_items, val_items


def load_manifest(manifest_path, train_dir):
    with open(manifest_path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)

    def deserialize(items):
        return [
            (os.path.join(train_dir, row["path"]), int(row["label"]))
            for row in items
        ]

    return (
        deserialize(payload["train"]),
        deserialize(payload["val"]),
        payload.get("seed", payload.get("method", "manifest")),
        float(payload.get("val_fraction", 0.1)),
    )


@torch.no_grad()
def predict_tta(model, img_np, img_size, device):
    base_tfm = A.Compose([
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])
    img_resized = robust_resize(img_np, img_size)
    tta_images = [
        img_resized,
        np.fliplr(img_resized).copy(),
        np.flipud(img_resized).copy(),
        np.rot90(img_resized, 1).copy(),
    ]
    probs = []
    for aug_img in tta_images:
        tensor = base_tfm(image=aug_img)["image"].unsqueeze(0).to(device)
        with torch.autocast(device_type="cuda" if "cuda" in device else "cpu", enabled=False):
            logits = model(tensor)
        probs.append(F.softmax(logits, dim=1))
    return torch.stack(probs).mean(dim=0).argmax(dim=1).item()


def main():
    parser = argparse.ArgumentParser(description="Evaluate classification checkpoint with submission-time preprocessing.")
    parser.add_argument("--model_path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--train_dir", default=DEFAULT_TRAIN_DIR)
    parser.add_argument("--manifest_path", default=None)
    parser.add_argument("--split", choices=["train", "val"], default="val")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model not found: {args.model_path}")
    if not os.path.isdir(args.train_dir):
        raise FileNotFoundError(f"Train dir not found: {args.train_dir}")

    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    ckpt = torch.load(args.model_path, map_location=device, weights_only=False)
    manifest_path = args.manifest_path or os.path.join(os.path.dirname(args.model_path), "classification_split_manifest.json")

    if os.path.exists(manifest_path):
        train_items, val_items, seed, val_fraction = load_manifest(manifest_path, args.train_dir)
        manifest_text = manifest_path
    else:
        seed = int(ckpt.get("split_seed", 42))
        val_fraction = float(ckpt.get("val_fraction", 0.1))
        train_items, val_items = build_split(args.train_dir, val_fraction, seed)
        manifest_text = "rebuilt from checkpoint metadata"

    items = train_items if args.split == "train" else val_items
    if args.limit > 0:
        items = items[:args.limit]

    model = timm.create_model(
        ckpt.get("model_name", "tf_efficientnet_b2.ns_jft_in1k"),
        pretrained=False,
        num_classes=ckpt.get("num_classes", 12),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()

    img_size = int(ckpt.get("img_size", 224))
    correct = 0
    total = 0
    per_class_total = Counter()
    per_class_correct = Counter()
    confusions = Counter()

    print(f"🔍 Model: {args.model_path}")
    print(f"🧾 Split source: {manifest_text}")
    print(f"📦 Split id={seed} | val_fraction={val_fraction:.2f}")
    print(f"📊 Evaluating {args.split} split on {len(items)} images with 4x TTA")

    for path, label in tqdm(items, desc="Evaluating"):
        img = np.array(Image.open(path).convert("RGB"))
        pred = predict_tta(model, img, img_size, device)
        per_class_total[label] += 1
        if pred == label:
            correct += 1
            per_class_correct[label] += 1
        else:
            confusions[(label, pred)] += 1
        total += 1

    accuracy = correct / total if total else 0.0
    print("\n" + "=" * 60)
    print(f"Submission-matched {args.split} accuracy: {accuracy * 100:.2f}%")
    print("=" * 60)
    print("Per-class accuracy:")
    for cls_id in sorted(per_class_total):
        cls_correct = per_class_correct[cls_id]
        cls_total = per_class_total[cls_id]
        print(f"  Class {cls_id}: {cls_correct}/{cls_total} = {cls_correct / cls_total:.2%}")

    if confusions:
        print("Top confusions:")
        for (true_label, pred_label), count in confusions.most_common(10):
            print(f"  {true_label} -> {pred_label}: {count}")


if __name__ == "__main__":
    main()
