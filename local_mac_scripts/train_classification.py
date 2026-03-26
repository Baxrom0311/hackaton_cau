"""
AI in Healthcare Hackathon 2026 — Classification Training Script
================================================================
EfficientNet-B2 with Transfer Learning for 12-class biopsy classification.

Usage:
    python train_classification.py

Model will be saved to: models/classification/best_model.pth
"""

import os
import copy
import time
import torch
import torch.nn as nn
import torch.optim as optim
import timm
import numpy as np
import pandas as pd
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, accuracy_score
from tqdm import tqdm
from PIL import Image

# ─── Configuration ───────────────────────────────────────────────────────────
class CFG:
    # Paths
    TRAIN_DIR = "classification/train"
    TEST_DIR = "classification/test"
    MODEL_SAVE_DIR = "models/classification"

    # Model
    MODEL_NAME = "efficientnet_b2"   # timm model name
    NUM_CLASSES = 12
    PRETRAINED = True

    # Training
    EPOCHS = 25
    BATCH_SIZE = 32
    LR = 1e-4
    WEIGHT_DECAY = 1e-4
    NUM_WORKERS = 0                  # 0 for Mac MPS (avoids fork issues)
    IMG_SIZE = 260                   # EfficientNet-B2 optimal input
    SEED = 42
    DEVICE = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

    # Validation
    VAL_RATIO = 0.15                 # 15% for validation from training data


def seed_everything(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ─── Dataset ─────────────────────────────────────────────────────────────────
class BiopClassificationDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = np.array(Image.open(self.image_paths[idx]).convert("RGB"))
        label = self.labels[idx]

        if self.transform:
            img = self.transform(image=img)["image"]

        return img, label


# ─── Augmentations ───────────────────────────────────────────────────────────
def get_train_transforms(img_size):
    return A.Compose([
        A.Resize(img_size, img_size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.Affine(translate_percent=(-0.1, 0.1), scale=(0.85, 1.15), rotate=(-30, 30), p=0.5),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
        A.GaussNoise(p=0.3),
        A.CoarseDropout(num_holes_range=(1, 8), hole_height_range=(1, img_size // 16), hole_width_range=(1, img_size // 16), p=0.3),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])


def get_val_transforms(img_size):
    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])


# ─── Model ───────────────────────────────────────────────────────────────────
def build_model(model_name, num_classes, pretrained=True):
    model = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)
    return model


# ─── Training Functions ──────────────────────────────────────────────────────
def train_one_epoch(model, loader, criterion, optimizer, scheduler, device):
    model.train()
    running_loss = 0.0
    preds_all, labels_all = [], []

    pbar = tqdm(loader, desc="  Train", leave=False)
    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        preds_all.extend(outputs.argmax(dim=1).cpu().numpy())
        labels_all.extend(labels.cpu().numpy())

        pbar.set_postfix(loss=f"{loss.item():.4f}")

    if scheduler is not None:
        scheduler.step()

    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = accuracy_score(labels_all, preds_all)
    return epoch_loss, epoch_acc


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    preds_all, labels_all = [], []

    for images, labels in tqdm(loader, desc="  Valid", leave=False):
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)
        preds_all.extend(outputs.argmax(dim=1).cpu().numpy())
        labels_all.extend(labels.cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = accuracy_score(labels_all, preds_all)
    return epoch_loss, epoch_acc


# ─── Data Preparation ────────────────────────────────────────────────────────
def prepare_data(train_dir):
    image_paths = []
    labels = []

    for class_id in sorted(os.listdir(train_dir), key=lambda x: int(x)):
        class_dir = os.path.join(train_dir, class_id)
        if not os.path.isdir(class_dir):
            continue
        for fname in os.listdir(class_dir):
            if fname.lower().endswith((".png", ".jpg", ".jpeg")):
                image_paths.append(os.path.join(class_dir, fname))
                labels.append(int(class_id))

    return image_paths, labels


# ─── Main Training Pipeline ──────────────────────────────────────────────────
def main():
    seed_everything(CFG.SEED)
    os.makedirs(CFG.MODEL_SAVE_DIR, exist_ok=True)

    print(f"{'='*60}")
    print(f"  AI Healthcare Hackathon — Classification Training")
    print(f"  Device: {CFG.DEVICE}")
    print(f"  Model:  {CFG.MODEL_NAME}")
    print(f"{'='*60}\n")

    # Prepare data
    image_paths, labels = prepare_data(CFG.TRAIN_DIR)
    print(f"Total training images: {len(image_paths)}")

    # Class distribution
    from collections import Counter
    dist = Counter(labels)
    for cls in sorted(dist.keys()):
        print(f"  Class {cls:>2d}: {dist[cls]:>5d} images")

    # Split train/val
    from sklearn.model_selection import train_test_split
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        image_paths, labels,
        test_size=CFG.VAL_RATIO,
        stratify=labels,
        random_state=CFG.SEED
    )
    print(f"\nTrain: {len(train_paths)} | Val: {len(val_paths)}")

    # Datasets & Loaders
    train_dataset = BiopClassificationDataset(
        train_paths, train_labels,
        transform=get_train_transforms(CFG.IMG_SIZE)
    )
    val_dataset = BiopClassificationDataset(
        val_paths, val_labels,
        transform=get_val_transforms(CFG.IMG_SIZE)
    )

    train_loader = DataLoader(
        train_dataset, batch_size=CFG.BATCH_SIZE,
        shuffle=True, num_workers=CFG.NUM_WORKERS,
        pin_memory=False, drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=CFG.BATCH_SIZE,
        shuffle=False, num_workers=CFG.NUM_WORKERS,
        pin_memory=False
    )

    # Class weights for imbalanced data
    class_counts = np.bincount(train_labels, minlength=CFG.NUM_CLASSES).astype(float)
    class_weights = 1.0 / (class_counts + 1e-6)
    class_weights = class_weights / class_weights.sum() * CFG.NUM_CLASSES
    class_weights = torch.FloatTensor(class_weights).to(CFG.DEVICE)

    # Model, Loss, Optimizer
    model = build_model(CFG.MODEL_NAME, CFG.NUM_CLASSES, CFG.PRETRAINED)
    model.to(CFG.DEVICE)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW(model.parameters(), lr=CFG.LR, weight_decay=CFG.WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CFG.EPOCHS, eta_min=1e-6)

    # Training loop
    best_val_acc = 0.0
    best_model_state = None

    for epoch in range(1, CFG.EPOCHS + 1):
        print(f"\nEpoch {epoch}/{CFG.EPOCHS}")
        print("-" * 40)

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, scheduler, CFG.DEVICE
        )
        val_loss, val_acc = validate(model, val_loader, criterion, CFG.DEVICE)

        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"  Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = copy.deepcopy(model.state_dict())
            save_path = os.path.join(CFG.MODEL_SAVE_DIR, "best_model.pth")
            torch.save({
                "model_state_dict": best_model_state,
                "model_name": CFG.MODEL_NAME,
                "num_classes": CFG.NUM_CLASSES,
                "img_size": CFG.IMG_SIZE,
                "val_acc": best_val_acc,
                "epoch": epoch,
            }, save_path)
            print(f"  ✅ Best model saved! Val Acc: {best_val_acc:.4f}")

    print(f"\n{'='*60}")
    print(f"  Training Finished! Best Val Accuracy: {best_val_acc:.4f}")
    print(f"  Model saved at: {CFG.MODEL_SAVE_DIR}/best_model.pth")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
