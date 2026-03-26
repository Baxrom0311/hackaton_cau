# ============================================================
# AI Healthcare Hackathon 2026 — PRO Classification (Colab T4)
# ============================================================
# ✅ EfficientNet-B4 (kattaroq model)
# ✅ Label Smoothing
# ✅ Mixup + CutMix augmentation
# ✅ SWA (Stochastic Weight Averaging)
# ✅ Focal Loss (kam uchraydigan klasslar uchun)
# ✅ TTA (Test-Time Augmentation) inference
# ✅ Gradient clipping + AMP (mixed precision)
# ✅ Auto-generate submission Excel
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
from sklearn.metrics import accuracy_score, classification_report
from collections import Counter
from tqdm import tqdm
from PIL import Image


# ─── Configuration ───────────────────────────────────────────────────────────
class CFG:
    BASE = "/content/drive/MyDrive/Main hackathon dataset"
    TRAIN_DIR = f"{BASE}/classification/train"
    TEST_DIR  = f"{BASE}/classification/test"
    MODEL_SAVE_DIR = "/content/drive/MyDrive/hackaton_models/classification_pro"

    # PRO: kattaroq model
    MODEL_NAME = "efficientnet_b4"
    NUM_CLASSES = 12
    PRETRAINED = True

    # Training
    EPOCHS = 30
    BATCH_SIZE = 32                   # T4 da b4 uchun 32
    LR = 2e-4
    WEIGHT_DECAY = 1e-4
    NUM_WORKERS = 2
    IMG_SIZE = 380                    # b4 uchun optimal 380
    SEED = 42
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # PRO sozlamalar
    LABEL_SMOOTHING = 0.1             # Overconfidence oldini oladi
    MIXUP_ALPHA = 0.4                 # Mixup kuchi
    CUTMIX_ALPHA = 1.0                # CutMix kuchi
    MIXUP_PROB = 0.5                  # Har bir batch da mixup/cutmix ehtimoli
    SWA_START = 22                    # SWA boshlanadigan epoch
    SWA_LR = 1e-5                     # SWA learning rate
    FOCAL_GAMMA = 2.0                 # Focal loss gamma
    GRAD_CLIP = 1.0
    VAL_RATIO = 0.15

    # TTA
    TTA_TRANSFORMS = 4               # original + 3 ta augmented


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


# ─── Focal Loss ──────────────────────────────────────────────────────────────
class FocalLoss(nn.Module):
    """
    Kam uchraydigan klasslarni kuchliroq o'rgatadi.
    gamma=0 bo'lsa oddiy CrossEntropy bilan bir xil.
    gamma=2 bo'lsa qiyin misollarni 4x kuchliroq jazoylaydi.
    """
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
    """
    Ikki rasmni aralashtirib yangi training namuna yaratadi.
    Bu model generalization ni kuchaytiradi.
    """
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)
    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def cutmix_data(x, y, alpha=1.0):
    """
    Rasmning bir qismini boshqa rasm bilan almashtiradi.
    Mixup dan farqi — geometrik aralashma.
    """
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
    y_a, y_b = y, y[index]
    return x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Mixup/CutMix uchun loss hisoblash."""
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


# ─── Augmentations (PRO) ────────────────────────────────────────────────────
def get_train_transforms(img_size):
    return A.Compose([
        A.Resize(img_size, img_size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.Affine(translate_percent=(-0.0625, 0.0625), scale=(0.9, 1.1), rotate=(-15, 15), p=0.5),
        A.OneOf([
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=1),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1),
        ], p=0.5),
        A.OneOf([
            A.GaussNoise(p=1),
            A.GaussianBlur(blur_limit=(3, 5), p=1),
        ], p=0.3),
        A.CoarseDropout(
            num_holes_range=(1, 3), hole_height_range=(0.05, 0.15),
            hole_width_range=(0.05, 0.15), fill=0, p=0.3
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
    """TTA: 4 xil ko'rinishda bashorat qilish."""
    base = [
        A.Resize(img_size, img_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ]
    return [
        A.Compose(base),                                          # original
        A.Compose([A.HorizontalFlip(p=1.0)] + base),              # horizontal flip
        A.Compose([A.VerticalFlip(p=1.0)] + base),                # vertical flip
        A.Compose([A.RandomRotate90(p=1.0)] + base),              # rotate 90
    ]


# ─── Training with AMP + Mixup/CutMix ───────────────────────────────────────
def train_one_epoch(model, loader, criterion, optimizer, scheduler, scaler, device, epoch, cfg):
    model.train()
    running_loss, preds_all, labels_all = 0.0, [], []

    pbar = tqdm(loader, desc=f"  Train E{epoch}", leave=False)
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)

        # Mixup / CutMix
        use_mix = np.random.random() < cfg.MIXUP_PROB and epoch < cfg.SWA_START
        if use_mix:
            if np.random.random() < 0.5:
                images, y_a, y_b, lam = mixup_data(images, labels, cfg.MIXUP_ALPHA)
            else:
                images, y_a, y_b, lam = cutmix_data(images, labels, cfg.CUTMIX_ALPHA)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast('cuda'):
            outputs = model(images)
            if use_mix:
                loss = mixup_criterion(criterion, outputs, y_a, y_b, lam)
            else:
                loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg.GRAD_CLIP)
        scaler.step(optimizer)
        scaler.update()

        if scheduler is not None and not isinstance(scheduler, torch.optim.swa_utils.SWALR):
            scheduler.step()

        running_loss += loss.item() * images.size(0)
        preds_all.extend(outputs.argmax(dim=1).cpu().numpy())
        labels_all.extend(labels.cpu().numpy())
        pbar.set_postfix(loss=f"{loss.item():.3f}")

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


# ─── TTA Prediction ─────────────────────────────────────────────────────────
@torch.no_grad()
def predict_with_tta(model, img_np, tta_transforms, device):
    """
    Rasmni 4 xil ko'rinishda bashorat qiladi va natijalarni averaging qiladi.
    Bu oddiy bashoratdan +2-5% aniqroq!
    """
    model.eval()
    all_probs = []

    for tfm in tta_transforms:
        img_t = tfm(image=img_np)["image"].unsqueeze(0).to(device)
        with torch.amp.autocast('cuda'):
            logits = model(img_t)
        probs = F.softmax(logits, dim=1)
        all_probs.append(probs)

    # Barcha augmentatsiya natijalarini o'rtacha
    avg_probs = torch.stack(all_probs).mean(dim=0)
    return avg_probs.argmax(dim=1).item(), avg_probs.max().item()


# ─── Data Preparation ───────────────────────────────────────────────────────
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


# ─── MAIN ────────────────────────────────────────────────────────────────────
def main():
    seed_everything(CFG.SEED)
    os.makedirs(CFG.MODEL_SAVE_DIR, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  🏆 PRO Classification Training")
    print(f"  Device : {CFG.DEVICE}")
    if CFG.DEVICE == "cuda":
        print(f"  GPU    : {torch.cuda.get_device_name(0)}")
    print(f"  Model  : {CFG.MODEL_NAME}")
    print(f"  ImgSize: {CFG.IMG_SIZE}")
    print(f"  🔥 Focal Loss + Label Smoothing + Mixup + CutMix + SWA + TTA")
    print(f"{'='*60}\n")

    # Data
    image_paths, labels = prepare_data(CFG.TRAIN_DIR)
    print(f"Total: {len(image_paths)} images")
    dist = Counter(labels)
    for cls in sorted(dist.keys()):
        print(f"  Class {cls:>2d}: {dist[cls]:>5d}")

    train_paths, val_paths, train_labels, val_labels = train_test_split(
        image_paths, labels, test_size=CFG.VAL_RATIO, stratify=labels, random_state=CFG.SEED
    )
    print(f"\nTrain: {len(train_paths)} | Val: {len(val_paths)}")

    train_dataset = BiopDataset(train_paths, train_labels, get_train_transforms(CFG.IMG_SIZE))
    val_dataset = BiopDataset(val_paths, val_labels, get_val_transforms(CFG.IMG_SIZE))

    train_loader = DataLoader(train_dataset, batch_size=CFG.BATCH_SIZE, shuffle=True,
                              num_workers=CFG.NUM_WORKERS, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=CFG.BATCH_SIZE, shuffle=False,
                            num_workers=CFG.NUM_WORKERS, pin_memory=True)

    # Class weights
    class_counts = np.bincount(train_labels, minlength=CFG.NUM_CLASSES).astype(float)
    class_weights = 1.0 / (class_counts + 1e-6)
    class_weights = class_weights / class_weights.sum() * CFG.NUM_CLASSES
    class_weights = torch.FloatTensor(class_weights).to(CFG.DEVICE)

    # Model
    model = timm.create_model(CFG.MODEL_NAME, pretrained=CFG.PRETRAINED, num_classes=CFG.NUM_CLASSES)
    model.to(CFG.DEVICE)

    # Focal Loss with Label Smoothing
    criterion = FocalLoss(
        weight=class_weights,
        gamma=CFG.FOCAL_GAMMA,
        label_smoothing=CFG.LABEL_SMOOTHING
    )

    optimizer = optim.AdamW(model.parameters(), lr=CFG.LR, weight_decay=CFG.WEIGHT_DECAY)
    scaler = torch.amp.GradScaler('cuda')

    # OneCycleLR scheduler
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=CFG.LR,
        total_steps=len(train_loader) * CFG.SWA_START,
        pct_start=0.1, anneal_strategy='cos',
    )

    # SWA model
    swa_model = torch.optim.swa_utils.AveragedModel(model)
    swa_scheduler = torch.optim.swa_utils.SWALR(optimizer, swa_lr=CFG.SWA_LR)

    best_val_acc = 0.0
    start_time = time.time()

    for epoch in range(1, CFG.EPOCHS + 1):
        t0 = time.time()
        print(f"\nEpoch {epoch}/{CFG.EPOCHS}" + (" [SWA]" if epoch >= CFG.SWA_START else ""))

        # SWA phase
        if epoch >= CFG.SWA_START:
            current_scheduler = swa_scheduler
        else:
            current_scheduler = scheduler

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, current_scheduler, scaler, CFG.DEVICE, epoch, CFG
        )

        # SWA update
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

    # SWA: batch norm yangilash
    print("\n🔄 SWA: Updating batch normalization...")
    torch.optim.swa_utils.update_bn(train_loader, swa_model, device=CFG.DEVICE)

    # SWA modelni saqlash
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

    # ─── TEST PREDICTION with TTA ────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  🎯 Generating Test Predictions with TTA...")
    print(f"{'='*60}")

    # Load best model
    ckpt = torch.load(os.path.join(CFG.MODEL_SAVE_DIR, "best_model.pth"), map_location=CFG.DEVICE)
    model_inf = timm.create_model(ckpt["model_name"], pretrained=False, num_classes=ckpt["num_classes"])
    model_inf.load_state_dict(ckpt["model_state_dict"])
    model_inf.to(CFG.DEVICE).eval()

    tta_tfms = get_tta_transforms(ckpt["img_size"])

    test_files = sorted([f for f in os.listdir(CFG.TEST_DIR) if f.lower().endswith((".png", ".jpg", ".jpeg"))])
    print(f"  Test images: {len(test_files)}")

    all_ids, all_preds, all_confs = [], [], []
    for fname in tqdm(test_files, desc="  TTA Predict"):
        img = np.array(Image.open(os.path.join(CFG.TEST_DIR, fname)).convert("RGB"))
        pred, conf = predict_with_tta(model_inf, img, tta_tfms, CFG.DEVICE)
        all_ids.append(os.path.splitext(fname)[0])
        all_preds.append(pred)
        all_confs.append(conf)

    # Save Excel
    df = pd.DataFrame({"Image_ID": all_ids, "Label": all_preds})
    df = df.sort_values("Image_ID").reset_index(drop=True)
    out_path = os.path.join(CFG.MODEL_SAVE_DIR, "test_ground_truth.xlsx")
    df.to_excel(out_path, index=False)

    print(f"\n  ✅ Saved {len(df)} predictions to {out_path}")
    print(f"  📊 Avg confidence: {np.mean(all_confs):.4f}")
    print(f"  📊 Label distribution: {Counter(all_preds)}")


main()
