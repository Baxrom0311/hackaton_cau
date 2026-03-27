# ============================================================
# AI Healthcare Hackathon 2026 — Segmentation Training (Colab)
# ============================================================
# 1. Runtime → Change runtime type → T4 GPU
# 2. Run all cells
# ============================================================

# !pip install segmentation-models-pytorch albumentations -q

from google.colab import drive
drive.mount('/content/drive')

import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm
import segmentation_models_pytorch as smp
import time


# ─── Configuration ───────────────────────────────────────────────────────────
class CFG:
    BASE = "/content/drive/MyDrive/Main hackathon dataset"
    TRAIN_IMG_DIR = f"{BASE}/Segmentation/training/images"
    TRAIN_MASK_DIR = f"{BASE}/Segmentation/training/masks"
    VAL_IMG_DIR = f"{BASE}/Segmentation/validation/images"
    VAL_MASK_DIR = f"{BASE}/Segmentation/validation/masks"
    TEST_IMG_DIR = f"{BASE}/Segmentation/testing/images"
    MODEL_SAVE_DIR = "/content/drive/MyDrive/hackaton_models/segmentation"

    ENCODER = "efficientnet-b3"
    ENCODER_WEIGHTS = "imagenet"
    ACTIVATION = None

    # Colab T4 GPU — tezkor sozlamalar
    EPOCHS = 40
    BATCH_SIZE = 32                   # T4 da 32 batch ishlaydi
    LR = 3e-4
    WEIGHT_DECAY = 1e-4
    NUM_WORKERS = 4                   # Colab da 4 worker
    IMG_SIZE = 256
    SEED = 42
    VAL_EVERY = 2
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def seed_everything(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


# ─── Dataset ─────────────────────────────────────────────────────────────────
class SegmentationDataset(Dataset):
    def __init__(self, img_dir, mask_dir, transform=None):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = sorted([
            f for f in os.listdir(img_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        fname = self.images[idx]
        img = np.array(Image.open(os.path.join(self.img_dir, fname)).convert("RGB"))

        mask_fname = os.path.splitext(fname)[0] + ".png"
        mask_path = os.path.join(self.mask_dir, mask_fname)
        if not os.path.exists(mask_path):
            mask_path = os.path.join(self.mask_dir, fname)
        mask = np.array(Image.open(mask_path).convert("L"))
        mask = (mask > 127).astype(np.float32)

        if self.transform:
            augmented = self.transform(image=img, mask=mask)
            img = augmented["image"]
            mask = augmented["mask"]

        mask = mask.unsqueeze(0) if isinstance(mask, torch.Tensor) else torch.tensor(mask).unsqueeze(0)
        return img, mask


# ─── Augmentations ───────────────────────────────────────────────────────────
def get_train_transforms(img_size):
    return A.Compose([
        A.Resize(img_size, img_size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.Affine(translate_percent=(-0.1, 0.1), scale=(0.8, 1.2), rotate=(-30, 30), p=0.5),
        A.OneOf([
            A.ElasticTransform(alpha=120, sigma=120 * 0.05, p=0.5),
            A.GridDistortion(p=0.5),
        ], p=0.3),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05, p=0.5),
        A.GaussNoise(p=0.2),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])


def get_val_transforms(img_size):
    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])


# ─── Loss & Metrics ─────────────────────────────────────────────────────────
@torch.no_grad()
def compute_iou(preds, targets, threshold=0.5):
    preds = (torch.sigmoid(preds) > threshold).float()
    intersection = (preds * targets).sum(dim=(2, 3))
    union = preds.sum(dim=(2, 3)) + targets.sum(dim=(2, 3)) - intersection
    iou = (intersection + 1e-6) / (union + 1e-6)
    return iou.mean().item()


class DiceBCELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, inputs, targets):
        bce_loss = self.bce(inputs, targets)
        inputs_sig = torch.sigmoid(inputs)
        intersection = (inputs_sig * targets).sum(dim=(2, 3))
        total = inputs_sig.sum(dim=(2, 3)) + targets.sum(dim=(2, 3))
        dice = 1 - (2.0 * intersection + 1e-6) / (total + 1e-6)
        return bce_loss + dice.mean()


# ─── Training with AMP (CUDA) ───────────────────────────────────────────────
def train_one_epoch(model, loader, criterion, optimizer, scheduler, scaler, device):
    model.train()
    running_loss = 0.0
    running_iou = 0.0
    n = 0

    pbar = tqdm(loader, desc="  Train", leave=False)
    for images, masks in pbar:
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast('cuda'):
            outputs = model(images)
            loss = criterion(outputs, masks)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        bs = images.size(0)
        running_loss += loss.item() * bs
        running_iou += compute_iou(outputs.float().detach(), masks) * bs
        n += bs
        pbar.set_postfix(loss=f"{loss.item():.3f}")

    return running_loss / n, running_iou / n


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    running_iou = 0.0
    n = 0

    for images, masks in tqdm(loader, desc="  Valid", leave=False):
        images = images.to(device)
        masks = masks.to(device)
        with torch.amp.autocast('cuda'):
            outputs = model(images)
            loss = criterion(outputs, masks)
        bs = images.size(0)
        running_loss += loss.item() * bs
        running_iou += compute_iou(outputs.float(), masks) * bs
        n += bs

    return running_loss / n, running_iou / n


# ─── Main ────────────────────────────────────────────────────────────────────
def main():
    seed_everything(CFG.SEED)
    os.makedirs(CFG.MODEL_SAVE_DIR, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  🚀 Colab Segmentation Training")
    print(f"  Device : {CFG.DEVICE}")
    if CFG.DEVICE == "cuda":
        print(f"  GPU    : {torch.cuda.get_device_name(0)}")
    print(f"  Encoder: {CFG.ENCODER}")
    print(f"  Batch  : {CFG.BATCH_SIZE}")
    print(f"{'='*60}\n")

    train_dataset = SegmentationDataset(
        CFG.TRAIN_IMG_DIR, CFG.TRAIN_MASK_DIR,
        transform=get_train_transforms(CFG.IMG_SIZE)
    )
    val_dataset = SegmentationDataset(
        CFG.VAL_IMG_DIR, CFG.VAL_MASK_DIR,
        transform=get_val_transforms(CFG.IMG_SIZE)
    )
    print(f"  Train: {len(train_dataset)} | Val: {len(val_dataset)}\n")

    train_loader = DataLoader(
        train_dataset, batch_size=CFG.BATCH_SIZE,
        shuffle=True, num_workers=CFG.NUM_WORKERS,
        pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=CFG.BATCH_SIZE * 2,
        shuffle=False, num_workers=CFG.NUM_WORKERS,
        pin_memory=True
    )

    model = smp.UnetPlusPlus(
        encoder_name=CFG.ENCODER,
        encoder_weights=CFG.ENCODER_WEIGHTS,
        in_channels=3, classes=1, activation=CFG.ACTIVATION,
    ).to(CFG.DEVICE)

    criterion = DiceBCELoss()
    optimizer = optim.AdamW(model.parameters(), lr=CFG.LR, weight_decay=CFG.WEIGHT_DECAY)
    scaler = torch.amp.GradScaler('cuda')

    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=CFG.LR,
        total_steps=len(train_loader) * CFG.EPOCHS,
        pct_start=0.1, anneal_strategy='cos',
    )

    best_iou = 0.0
    start_time = time.time()

    for epoch in range(1, CFG.EPOCHS + 1):
        t0 = time.time()
        print(f"\nEpoch {epoch}/{CFG.EPOCHS}")

        train_loss, train_iou = train_one_epoch(
            model, train_loader, criterion, optimizer, scheduler, scaler, CFG.DEVICE
        )
        dt = time.time() - t0
        print(f"  Train Loss: {train_loss:.4f} | IoU: {train_iou:.4f} | ⏱️ {dt:.0f}s")

        if epoch % CFG.VAL_EVERY == 0 or epoch == CFG.EPOCHS:
            val_loss, val_iou = validate(model, val_loader, criterion, CFG.DEVICE)
            print(f"  Val   Loss: {val_loss:.4f} | IoU: {val_iou:.4f}")

            if val_iou > best_iou:
                best_iou = val_iou
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "encoder": CFG.ENCODER,
                    "img_size": CFG.IMG_SIZE,
                    "val_iou": best_iou,
                    "epoch": epoch,
                }, os.path.join(CFG.MODEL_SAVE_DIR, "best_model.pth"))
                print(f"  ✅ Best model! IoU: {best_iou:.4f}")

        elapsed = time.time() - start_time
        eta = elapsed / epoch * (CFG.EPOCHS - epoch)
        print(f"  ⏱️ {elapsed/60:.1f}min | ETA: {eta/60:.1f}min")

    print(f"\n{'='*60}")
    print(f"  ✅ Done! Best IoU: {best_iou:.4f}")
    print(f"  ⏱️ {(time.time()-start_time)/60:.1f} min")
    print(f"  📁 {CFG.MODEL_SAVE_DIR}/best_model.pth")
    print(f"{'='*60}")


main()
