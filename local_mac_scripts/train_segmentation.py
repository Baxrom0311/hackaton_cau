"""
AI in Healthcare Hackathon 2026 — Segmentation Training (Mac MPS)
=================================================================
UNet++ with EfficientNet-B3 — Optimized for Mac MPS
- RAM preloading (disk I/O yo'q)
- NUM_WORKERS=0 (Mac fork crash yo'q)
- NO AMP on MPS (MPS autocast juda sekin!)
- OneCycleLR for fast convergence
- Validate every 3 epochs

Usage:
    python train_segmentation.py
"""

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
    TRAIN_IMG_DIR = "Segmentation/training/images"
    TRAIN_MASK_DIR = "Segmentation/training/masks"
    VAL_IMG_DIR = "Segmentation/validation/images"
    VAL_MASK_DIR = "Segmentation/validation/masks"
    MODEL_SAVE_DIR = "models/segmentation"

    ENCODER = "efficientnet-b3"
    ENCODER_WEIGHTS = "imagenet"
    ACTIVATION = None

    # Mac MPS optimal settings
    EPOCHS = 30
    BATCH_SIZE = 8                    # MPS uchun 8 optimal
    LR = 3e-4
    WEIGHT_DECAY = 1e-4
    NUM_WORKERS = 0                   # Mac da 0 bo'lishi SHART
    IMG_SIZE = 256
    SEED = 42
    VAL_EVERY = 3                     # Har 3 epochda validation
    DEVICE = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")


def seed_everything(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)


# ─── Dataset — RAM ga oldindan yuklash ────────────────────────────────────────
class SegmentationDataset(Dataset):
    def __init__(self, img_dir, mask_dir, transform=None):
        self.transform = transform
        self.images_data = []
        self.masks_data = []

        filenames = sorted([
            f for f in os.listdir(img_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ])

        print(f"    Loading {len(filenames)} images from {img_dir}...", end=" ", flush=True)
        t0 = time.time()

        for fname in filenames:
            img = np.array(Image.open(os.path.join(img_dir, fname)).convert("RGB"))

            mask_fname = os.path.splitext(fname)[0] + ".png"
            mask_path = os.path.join(mask_dir, mask_fname)
            if not os.path.exists(mask_path):
                mask_path = os.path.join(mask_dir, fname)
            mask = np.array(Image.open(mask_path).convert("L"))
            mask = (mask > 127).astype(np.float32)

            self.images_data.append(img)
            self.masks_data.append(mask)

        print(f"✅ ({time.time()-t0:.1f}s)")

    def __len__(self):
        return len(self.images_data)

    def __getitem__(self, idx):
        img = self.images_data[idx]
        mask = self.masks_data[idx]

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


# ─── Metrics & Loss ─────────────────────────────────────────────────────────
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


# ─── Training ────────────────────────────────────────────────────────────────
def train_one_epoch(model, loader, criterion, optimizer, scheduler, device):
    model.train()
    running_loss = 0.0
    running_iou = 0.0
    n = 0

    pbar = tqdm(loader, desc="  Train", leave=False)
    for images, masks in pbar:
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad(set_to_none=True)
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        bs = images.size(0)
        running_loss += loss.item() * bs
        running_iou += compute_iou(outputs.detach(), masks) * bs
        n += bs
        pbar.set_postfix(loss=f"{loss.item():.3f}", lr=f"{scheduler.get_last_lr()[0]:.2e}")

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
        outputs = model(images)
        loss = criterion(outputs, masks)
        bs = images.size(0)
        running_loss += loss.item() * bs
        running_iou += compute_iou(outputs, masks) * bs
        n += bs

    return running_loss / n, running_iou / n


# ─── Main ────────────────────────────────────────────────────────────────────
def main():
    seed_everything(CFG.SEED)
    os.makedirs(CFG.MODEL_SAVE_DIR, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  🚀 Segmentation Training (Mac MPS Optimized)")
    print(f"  Device : {CFG.DEVICE}")
    print(f"  Encoder: {CFG.ENCODER}")
    print(f"  Batch  : {CFG.BATCH_SIZE}")
    print(f"  Epochs : {CFG.EPOCHS} (val every {CFG.VAL_EVERY})")
    print(f"{'='*60}\n")

    print("📦 Loading datasets into RAM...")
    train_dataset = SegmentationDataset(
        CFG.TRAIN_IMG_DIR, CFG.TRAIN_MASK_DIR,
        transform=get_train_transforms(CFG.IMG_SIZE)
    )
    val_dataset = SegmentationDataset(
        CFG.VAL_IMG_DIR, CFG.VAL_MASK_DIR,
        transform=get_val_transforms(CFG.IMG_SIZE)
    )
    print(f"\n  Train: {len(train_dataset)} | Val: {len(val_dataset)}\n")

    train_loader = DataLoader(
        train_dataset, batch_size=CFG.BATCH_SIZE,
        shuffle=True, num_workers=0, pin_memory=False, drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=CFG.BATCH_SIZE * 2,
        shuffle=False, num_workers=0, pin_memory=False
    )

    model = smp.UnetPlusPlus(
        encoder_name=CFG.ENCODER,
        encoder_weights=CFG.ENCODER_WEIGHTS,
        in_channels=3, classes=1, activation=CFG.ACTIVATION,
    )
    model.to(CFG.DEVICE)

    criterion = DiceBCELoss()
    optimizer = optim.AdamW(model.parameters(), lr=CFG.LR, weight_decay=CFG.WEIGHT_DECAY)

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
        print("-" * 40)

        train_loss, train_iou = train_one_epoch(
            model, train_loader, criterion, optimizer, scheduler, CFG.DEVICE
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
        print(f"  ⏱️ {elapsed/60:.1f}min elapsed | ETA: {eta/60:.1f}min")

    print(f"\n{'='*60}")
    print(f"  ✅ Done! Best IoU: {best_iou:.4f}")
    print(f"  ⏱️ {(time.time()-start_time)/60:.1f} min total")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
