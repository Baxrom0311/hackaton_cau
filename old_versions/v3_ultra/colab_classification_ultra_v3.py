# ============================================================
# AI Healthcare Hackathon 2026 — ULTRA Classification (Colab)
# ============================================================
# 🚨 MAQSAD: Classification bo'yicha maksimal 30 balni olish (Accuracy > 90%)
# 
# ✅ Model: tf_efficientnet_b5_ns (Noisy Student weights juda zo'r)
# ✅ Resolution: 512x512
# ✅ Augmentations: Mixup (80%) + CutMix, ElasticTransform
# ✅ SWA (Stochastic Weight Averaging)
# ✅ Focal Loss + Label Smoothing
# ✅ SOTA Schedulers (CosineAnnealingWarmRestarts)
# ============================================================

# Cell 1: install
# !pip install timm albumentations openpyxl -q

from google.colab import drive
drive.mount('/content/drive')

import os, copy, time, random
import torch
import torch.nn as nn
import torch.nn.functional as F
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


# ─── Configuration ───────────────────────────────────────────────────────────
class CFG:
    BASE = "/content/drive/MyDrive/Main hackathon dataset"
    TRAIN_DIR = f"{BASE}/classification/train"
    TEST_DIR  = f"{BASE}/classification/test"
    MODEL_SAVE_DIR = "/content/drive/MyDrive/hackaton_models/classification_ultra"

    # ULTRA: tf_efficientnet_b5_ns
    MODEL_NAME = "tf_efficientnet_b5_ns"
    NUM_CLASSES = 12
    PRETRAINED = True

    # Training
    EPOCHS = 40
    BATCH_SIZE = 16
    GRAD_ACCUM_STEPS = 2
    LR = 2e-4
    WEIGHT_DECAY = 1e-4
    NUM_WORKERS = 2
    IMG_SIZE = 512
    SEED = 42
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # ULTRA sozlamalar
    LABEL_SMOOTHING = 0.1
    MIXUP_ALPHA = 0.4
    CUTMIX_ALPHA = 1.0
    MIXUP_PROB = 0.7                  # Oliy darajadagi regularization
    SWA_START = 30
    SWA_LR = 1e-5
    FOCAL_GAMMA = 2.0
    GRAD_CLIP = 1.0
    VAL_RATIO = 0.15


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


# ─── Focal Loss ──────────────────────────────────────────────────────────────
class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=2.0, label_smoothing=0.0):
        super().__init__()
        self.weight = weight
        self.gamma = gamma
        self.label_smoothing = label_smoothing

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(
            inputs, targets,
            weight=self.weight,
            label_smoothing=self.label_smoothing,
            reduction='none'
        )
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()


# ─── Mixup & CutMix ─────────────────────────────────────────────────────────
def mixup_data(x, y, alpha=0.4):
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)
    mixed_x = lam * x + (1 - lam) * x[index]
    return mixed_x, y, y[index], lam

def cutmix_data(x, y, alpha=1.0):
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)

    _, _, H, W = x.shape
    cut_rat = np.sqrt(1.0 - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)
    cx = np.random.randint(W)
    cy = np.random.randint(H)
    x1 = np.clip(cx - cut_w // 2, 0, W)
    y1 = np.clip(cy - cut_h // 2, 0, H)
    x2 = np.clip(cx + cut_w // 2, 0, W)
    y2 = np.clip(cy + cut_h // 2, 0, H)

    x[:, :, y1:y2, x1:x2] = x[index, :, y1:y2, x1:x2]
    lam = 1 - ((x2 - x1) * (y2 - y1) / (W * H))
    return x, y, y[index], lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


# ─── Dataset ─────────────────────────────────────────────────────────────────
class BiopDataset(Dataset):
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


# ─── Augmentations (ULTRA) ──────────────────────────────────────────────────
def get_train_transforms(img_size):
    return A.Compose([
        A.Resize(img_size, img_size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ElasticTransform(alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03, p=0.3),
        A.Affine(translate_percent=(-0.0625, 0.0625), scale=(0.9, 1.1), rotate=(-15, 15), p=0.5),
        A.OneOf([
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=1),
            A.CLAHE(clip_limit=3.0, p=1),
        ], p=0.5),
        A.CoarseDropout(
            num_holes_range=(1, 3), hole_height_range=(0.05, 0.15),
            hole_width_range=(0.05, 0.15), fill=0, p=0.4
        ),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

def get_val_transforms(img_size):
    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

def get_tta_transforms(img_size):
    base = [A.Resize(img_size, img_size), A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), ToTensorV2()]
    return [
        A.Compose(base),
        A.Compose([A.HorizontalFlip(p=1.0)] + base),
        A.Compose([A.VerticalFlip(p=1.0)] + base),
        A.Compose([A.RandomRotate90(p=1.0)] + base),
    ]


# ─── Training Flow ──────────────────────────────────────────────────────────
def train_one_epoch(model, loader, criterion, optimizer, scheduler, scaler, device, epoch, cfg):
    model.train()
    running_loss, preds_all, labels_all = 0.0, [], []

    optimizer.zero_grad(set_to_none=True)
    pbar = tqdm(loader, desc=f"  Train E{epoch}", leave=False)
    
    for i, (images, labels) in enumerate(pbar):
        images, labels = images.to(device), labels.to(device)

        use_mix = np.random.random() < cfg.MIXUP_PROB and epoch < cfg.SWA_START
        if use_mix:
            if np.random.random() < 0.5:
                images, y_a, y_b, lam = mixup_data(images, labels, cfg.MIXUP_ALPHA)
            else:
                images, y_a, y_b, lam = cutmix_data(images, labels, cfg.CUTMIX_ALPHA)

        with torch.amp.autocast('cuda'):
            outputs = model(images)
            if use_mix:
                loss = mixup_criterion(criterion, outputs, y_a, y_b, lam)
            else:
                loss = criterion(outputs, labels)
            loss = loss / cfg.GRAD_ACCUM_STEPS

        scaler.scale(loss).backward()
        
        if (i + 1) % cfg.GRAD_ACCUM_STEPS == 0 or (i + 1) == len(loader):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg.GRAD_CLIP)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            
            if scheduler is not None and not isinstance(scheduler, torch.optim.swa_utils.SWALR):
                scheduler.step(epoch + i / len(loader))

        running_loss += (loss.item() * cfg.GRAD_ACCUM_STEPS) * images.size(0)
        preds_all.extend(outputs.argmax(dim=1).cpu().numpy())
        labels_all.extend(labels.cpu().numpy())
        pbar.set_postfix(loss=f"{loss.item()*cfg.GRAD_ACCUM_STEPS:.3f}")

    return running_loss / len(loader.dataset), accuracy_score(labels_all, preds_all)

@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    running_loss, preds_all, labels_all = 0.0, [], []

    for images, labels in tqdm(loader, desc="  Valid", leave=False):
        images, labels = images.to(device), labels.to(device)
        with torch.amp.autocast('cuda'):
            outputs = model(images)
            loss = criterion(outputs, labels)
        running_loss += loss.item() * images.size(0)
        preds_all.extend(outputs.argmax(dim=1).cpu().numpy())
        labels_all.extend(labels.cpu().numpy())

    return running_loss / len(loader.dataset), accuracy_score(labels_all, preds_all)


# ─── MAIN ────────────────────────────────────────────────────────────────────
def main():
    global epoch
    seed_everything(CFG.SEED)
    os.makedirs(CFG.MODEL_SAVE_DIR, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  🏆 ULTRA Classification Training (Noisy Student)")
    print(f"  Device : {CFG.DEVICE} | {torch.cuda.get_device_name(0) if CFG.DEVICE == 'cuda' else ''}")
    print(f"  Model  : {CFG.MODEL_NAME} | ImgSize: {CFG.IMG_SIZE}")
    print(f"  🔥 Focal Loss + WarmRestarts + Mixup/CutMix + SWA + TTA")
    print(f"{'='*60}\n")

    image_paths, labels = [], []
    for class_id in sorted(os.listdir(CFG.TRAIN_DIR), key=lambda x: int(x)):
        class_dir = os.path.join(CFG.TRAIN_DIR, class_id)
        if not os.path.isdir(class_dir): continue
        for fname in os.listdir(class_dir):
            if fname.lower().endswith((".png", ".jpg", ".jpeg")):
                image_paths.append(os.path.join(class_dir, fname))
                labels.append(int(class_id))

    train_paths, val_paths, train_labels, val_labels = train_test_split(
        image_paths, labels, test_size=CFG.VAL_RATIO, stratify=labels, random_state=CFG.SEED
    )

    train_dataset = BiopDataset(train_paths, train_labels, get_train_transforms(CFG.IMG_SIZE))
    val_dataset = BiopDataset(val_paths, val_labels, get_val_transforms(CFG.IMG_SIZE))

    train_loader = DataLoader(train_dataset, batch_size=CFG.BATCH_SIZE, shuffle=True,
                              num_workers=CFG.NUM_WORKERS, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=CFG.BATCH_SIZE, shuffle=False,
                            num_workers=CFG.NUM_WORKERS, pin_memory=True)

    class_counts = np.bincount(train_labels, minlength=CFG.NUM_CLASSES).astype(float)
    class_weights = 1.0 / (class_counts + 1e-6)
    class_weights = class_weights / class_weights.sum() * CFG.NUM_CLASSES
    class_weights = torch.FloatTensor(class_weights).to(CFG.DEVICE)

    model = timm.create_model(CFG.MODEL_NAME, pretrained=CFG.PRETRAINED, num_classes=CFG.NUM_CLASSES)
    model.to(CFG.DEVICE)

    criterion = FocalLoss(weight=class_weights, gamma=CFG.FOCAL_GAMMA, label_smoothing=CFG.LABEL_SMOOTHING)
    optimizer = optim.AdamW(model.parameters(), lr=CFG.LR, weight_decay=CFG.WEIGHT_DECAY)
    scaler = torch.amp.GradScaler('cuda')

    # ULTRA: CosineAnnealingWarmRestarts
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-6
    )

    swa_model = torch.optim.swa_utils.AveragedModel(model)
    swa_scheduler = torch.optim.swa_utils.SWALR(optimizer, swa_lr=CFG.SWA_LR)

    best_val_acc = 0.0
    start_time = time.time()

    for epoch in range(1, CFG.EPOCHS + 1):
        t0 = time.time()
        print(f"\nEpoch {epoch}/{CFG.EPOCHS}" + (" [SWA]" if epoch >= CFG.SWA_START else ""))

        current_scheduler = swa_scheduler if epoch >= CFG.SWA_START else scheduler

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, current_scheduler, scaler, CFG.DEVICE, epoch, CFG
        )

        if epoch >= CFG.SWA_START:
            swa_model.update_parameters(model)
            swa_scheduler.step()

        val_loss, val_acc = validate(model, val_loader, criterion, CFG.DEVICE)
        dt = time.time() - t0

        print(f"  Train Loss: {train_loss:.4f} | Acc: {train_acc:.4f}")
        print(f"  Val   Loss: {val_loss:.4f} | Acc: {val_acc:.4f} | ⏱️ {dt:.0f}s")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "model_state_dict": model.state_dict(),
                "model_name": CFG.MODEL_NAME,
                "num_classes": CFG.NUM_CLASSES,
                "img_size": CFG.IMG_SIZE,
                "val_acc": best_val_acc,
                "epoch": epoch,
            }, os.path.join(CFG.MODEL_SAVE_DIR, "best_model.pth"))
            print(f"  ✅ Best model! Acc: {best_val_acc:.4f}")

    print("\n🔄 SWA: Updating batch normalization...")
    torch.optim.swa_utils.update_bn(train_loader, swa_model, device=CFG.DEVICE)
    swa_val_loss, swa_val_acc = validate(swa_model, val_loader, criterion, CFG.DEVICE)
    print(f"  SWA Val Acc: {swa_val_acc:.4f} (vs Best: {best_val_acc:.4f})")

    if swa_val_acc > best_val_acc:
        best_val_acc = swa_val_acc
        torch.save({
            "model_state_dict": swa_model.module.state_dict(),
            "model_name": CFG.MODEL_NAME,
            "num_classes": CFG.NUM_CLASSES,
            "img_size": CFG.IMG_SIZE,
            "val_acc": best_val_acc,
            "epoch": "SWA",
        }, os.path.join(CFG.MODEL_SAVE_DIR, "best_model.pth"))
        print(f"  ✅ SWA model saved! Acc: {best_val_acc:.4f}")

    print(f"\n{'='*60}")
    print(f"  ✅ Training Done! Best Acc: {best_val_acc:.4f}")
    print(f"  ⏱️ {(time.time()-start_time)/60:.1f} min")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
