# ============================================================
# AI Healthcare Hackathon 2026 — Classification Training (Colab)
# ============================================================
# QADAMLAR:
# 1. Google Drive ga datasetni yuklang (zip qilib)
# 2. Google Colab oching: https://colab.research.google.com
# 3. Runtime → Change runtime type → T4 GPU tanlang
# 4. Bu faylni to'liq copy-paste qiling va Run qiling
# ============================================================

# ── 1-QADAM: Kutubxonalarni o'rnatish ──
# !pip install timm albumentations openpyxl -q

# ── 2-QADAM: Google Drive ulash ──
from google.colab import drive
drive.mount('/content/drive')

import os

# ============================================================
# 4-QADAM: TRAIN_DIR va TEST_DIR ni aniqlang
# Dataset ichida classification/train va classification/test
# papkalar qaerdaligini tekshiring va quyida to'g'rilang
# ============================================================

# ── TRAINING SCRIPT ──────────────────────────────────────────

import copy
import torch
import torch.nn as nn
import torch.optim as optim
import timm
import numpy as np
import pandas as pd
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from collections import Counter
from tqdm import tqdm
from PIL import Image


class CFG:
    TRAIN_DIR = "/content/drive/MyDrive/Main hackathon dataset/classification/train"
    TEST_DIR  = "/content/drive/MyDrive/Main hackathon dataset/classification/test"
    MODEL_SAVE_DIR = "/content/drive/MyDrive/hackaton_models/classification"

    MODEL_NAME = "efficientnet_b2"
    NUM_CLASSES = 12
    PRETRAINED = True

    EPOCHS = 25
    BATCH_SIZE = 64          # Colab GPU da kattaroq batch
    LR = 1e-4
    WEIGHT_DECAY = 1e-4
    NUM_WORKERS = 2
    IMG_SIZE = 260
    SEED = 42
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    VAL_RATIO = 0.15


def seed_everything(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


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


def get_train_transforms(img_size):
    return A.Compose([
        A.Resize(img_size, img_size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.Affine(translate_percent=(-0.1, 0.1), scale=(0.85, 1.15), rotate=(-30, 30), p=0.5),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
        A.GaussNoise(p=0.3),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])


def get_val_transforms(img_size):
    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])


def build_model(model_name, num_classes, pretrained=True):
    return timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)


def train_one_epoch(model, loader, criterion, optimizer, scheduler, device):
    model.train()
    running_loss, preds_all, labels_all = 0.0, [], []

    pbar = tqdm(loader, desc="  Train", leave=False)
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
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

    return running_loss / len(loader.dataset), accuracy_score(labels_all, preds_all)


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    running_loss, preds_all, labels_all = 0.0, [], []

    for images, labels in tqdm(loader, desc="  Valid", leave=False):
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        running_loss += loss.item() * images.size(0)
        preds_all.extend(outputs.argmax(dim=1).cpu().numpy())
        labels_all.extend(labels.cpu().numpy())

    return running_loss / len(loader.dataset), accuracy_score(labels_all, preds_all)


def prepare_data(train_dir):
    image_paths, labels = [], []
    for class_id in sorted(os.listdir(train_dir), key=lambda x: int(x)):
        class_dir = os.path.join(train_dir, class_id)
        if not os.path.isdir(class_dir):
            continue
        for fname in os.listdir(class_dir):
            if fname.lower().endswith((".png", ".jpg", ".jpeg")):
                image_paths.append(os.path.join(class_dir, fname))
                labels.append(int(class_id))
    return image_paths, labels


# ── MAIN TRAINING ────────────────────────────────────────────
def main():
    seed_everything(CFG.SEED)
    os.makedirs(CFG.MODEL_SAVE_DIR, exist_ok=True)

    print(f"{'='*60}")
    print(f"  Device: {CFG.DEVICE}")
    print(f"  Model:  {CFG.MODEL_NAME}")
    if CFG.DEVICE == "cuda":
        print(f"  GPU:    {torch.cuda.get_device_name(0)}")
    print(f"{'='*60}\n")

    image_paths, labels = prepare_data(CFG.TRAIN_DIR)
    print(f"Total: {len(image_paths)} images")
    dist = Counter(labels)
    for cls in sorted(dist.keys()):
        print(f"  Class {cls:>2d}: {dist[cls]:>5d}")

    train_paths, val_paths, train_labels, val_labels = train_test_split(
        image_paths, labels, test_size=CFG.VAL_RATIO, stratify=labels, random_state=CFG.SEED
    )
    print(f"\nTrain: {len(train_paths)} | Val: {len(val_paths)}")

    train_dataset = BiopClassificationDataset(train_paths, train_labels, get_train_transforms(CFG.IMG_SIZE))
    val_dataset = BiopClassificationDataset(val_paths, val_labels, get_val_transforms(CFG.IMG_SIZE))

    train_loader = DataLoader(train_dataset, batch_size=CFG.BATCH_SIZE, shuffle=True, num_workers=CFG.NUM_WORKERS, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=CFG.BATCH_SIZE, shuffle=False, num_workers=CFG.NUM_WORKERS, pin_memory=True)

    # Class weights
    class_counts = np.bincount(train_labels, minlength=CFG.NUM_CLASSES).astype(float)
    class_weights = 1.0 / (class_counts + 1e-6)
    class_weights = class_weights / class_weights.sum() * CFG.NUM_CLASSES
    class_weights = torch.FloatTensor(class_weights).to(CFG.DEVICE)

    model = build_model(CFG.MODEL_NAME, CFG.NUM_CLASSES, CFG.PRETRAINED).to(CFG.DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW(model.parameters(), lr=CFG.LR, weight_decay=CFG.WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CFG.EPOCHS, eta_min=1e-6)

    best_val_acc = 0.0
    for epoch in range(1, CFG.EPOCHS + 1):
        print(f"\nEpoch {epoch}/{CFG.EPOCHS}")
        print("-" * 40)
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, scheduler, CFG.DEVICE)
        val_loss, val_acc = validate(model, val_loader, criterion, CFG.DEVICE)
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"  Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_path = os.path.join(CFG.MODEL_SAVE_DIR, "best_model.pth")
            torch.save({
                "model_state_dict": model.state_dict(),
                "model_name": CFG.MODEL_NAME,
                "num_classes": CFG.NUM_CLASSES,
                "img_size": CFG.IMG_SIZE,
                "val_acc": best_val_acc,
                "epoch": epoch,
            }, save_path)
            print(f"  ✅ Best model saved! Val Acc: {best_val_acc:.4f}")

    print(f"\n{'='*60}")
    print(f"  DONE! Best Val Accuracy: {best_val_acc:.4f}")
    print(f"  Model: {CFG.MODEL_SAVE_DIR}/best_model.pth")
    print(f"{'='*60}")


main()


# ============================================================
# 5-QADAM: TEST PREDICTION (training tugagandan keyin ishlating)
# ============================================================
#
# @torch.no_grad()
# def predict_test():
#     device = CFG.DEVICE
#     checkpoint = torch.load(f"{CFG.MODEL_SAVE_DIR}/best_model.pth", map_location=device)
#     model = timm.create_model(checkpoint["model_name"], pretrained=False, num_classes=checkpoint["num_classes"])
#     model.load_state_dict(checkpoint["model_state_dict"])
#     model.to(device).eval()
#
#     transform = get_val_transforms(checkpoint["img_size"])
#     test_files = sorted([f for f in os.listdir(CFG.TEST_DIR) if f.lower().endswith((".png", ".jpg", ".jpeg"))])
#
#     all_ids, all_preds = [], []
#     for fname in tqdm(test_files, desc="Predicting"):
#         img = np.array(Image.open(os.path.join(CFG.TEST_DIR, fname)).convert("RGB"))
#         img_t = transform(image=img)["image"].unsqueeze(0).to(device)
#         pred = model(img_t).argmax(dim=1).item()
#         all_ids.append(int(os.path.splitext(fname)[0]))
#         all_preds.append(pred)
#
#     df = pd.DataFrame({"Image_ID": all_ids, "Label": all_preds})
#     df = df.sort_values("Image_ID").reset_index(drop=True)
#     out_path = f"{CFG.MODEL_SAVE_DIR}/test_ground_truth.xlsx"
#     df.to_excel(out_path, index=False)
#     print(f"✅ Saved {len(df)} predictions to {out_path}")
#
# predict_test()
