# ============================================================
# AI Healthcare Hackathon 2026 — PRO Segmentation (Colab T4)
# ============================================================
# ✅ EfficientNet-B4 encoder (kattaroq)
# ✅ IMG_SIZE 384 (yaxshiroq detallar)
# ✅ Lovász Loss + DiceBCE (IoU ni to'g'ridan-to'g'ri optimize)
# ✅ SWA (Stochastic Weight Averaging)
# ✅ TTA (Test-Time Augmentation) inference
# ✅ Post-processing (morphological operations)
# ✅ AMP (mixed precision) training
# ✅ Auto-generate submission masks
# ============================================================

# Cell 1: install
# !pip install segmentation-models-pytorch albumentations -q

from google.colab import drive
drive.mount('/content/drive')

import os, time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm
import segmentation_models_pytorch as smp
import cv2


# ─── Configuration ───────────────────────────────────────────────────────────
class CFG:
    BASE = "/content/drive/MyDrive/Main hackathon dataset"
    TRAIN_IMG_DIR = f"{BASE}/Segmentation/training/images"
    TRAIN_MASK_DIR = f"{BASE}/Segmentation/training/masks"
    VAL_IMG_DIR = f"{BASE}/Segmentation/validation/images"
    VAL_MASK_DIR = f"{BASE}/Segmentation/validation/masks"
    TEST_IMG_DIR = f"{BASE}/Segmentation/testing/images"
    MODEL_SAVE_DIR = "/content/drive/MyDrive/hackaton_models/segmentation_pro"

    # PRO: kattaroq encoder + rasm
    ENCODER = "efficientnet-b4"
    ENCODER_WEIGHTS = "imagenet"
    ACTIVATION = None
    DECODER = "UnetPlusPlus"          # UNet++ eng yaxshi segmentation decoder

    # Training
    EPOCHS = 40
    BATCH_SIZE = 16                   # T4 da 384x384 uchun 16
    LR = 3e-4
    WEIGHT_DECAY = 1e-4
    NUM_WORKERS = 2
    IMG_SIZE = 384                    # Kattaroq = yaxshiroq detallar
    SEED = 42
    VAL_EVERY = 2
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # PRO
    SWA_START = 30
    SWA_LR = 1e-5
    GRAD_CLIP = 1.0
    LOVASZ_WEIGHT = 0.5              # Lovász loss vazni
    DICE_WEIGHT = 0.5                # DiceBCE loss vazni


def seed_everything(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


# ─── Lovász Loss ─────────────────────────────────────────────────────────────
def lovasz_grad(gt_sorted):
    """Lovász gradient hisoblash."""
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1.0 - intersection / union
    if p > 1:
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard


def lovasz_hinge_flat(logits, labels):
    """
    Binary Lovász hinge loss.
    IoU metrikasini to'g'ridan-to'g'ri optimize qiladi!
    DiceLoss dan farqi — gradient aniqroq.
    """
    if len(labels) == 0:
        return logits.sum() * 0.0
    signs = 2.0 * labels.float() - 1.0
    errors = 1.0 - logits * signs
    errors_sorted, perm = torch.sort(errors, dim=0, descending=True)
    perm = perm.data
    gt_sorted = labels[perm]
    grad = lovasz_grad(gt_sorted)
    loss = torch.dot(F.relu(errors_sorted), grad)
    return loss


def lovasz_hinge(logits, labels):
    """Batch bo'yicha Lovász loss."""
    losses = []
    for logit, label in zip(logits.view(logits.size(0), -1),
                            labels.view(labels.size(0), -1)):
        losses.append(lovasz_hinge_flat(logit, label))
    return torch.stack(losses).mean()


# ─── Combined Loss ───────────────────────────────────────────────────────────
class CombinedLoss(nn.Module):
    """
    DiceBCE + Lovász = eng kuchli segmentation loss!
    DiceBCE: barqaror gradient beradi
    Lovász: IoU ni to'g'ridan-to'g'ri optimize qiladi
    """
    def __init__(self, lovasz_weight=0.5, dice_weight=0.5):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.lovasz_w = lovasz_weight
        self.dice_w = dice_weight

    def forward(self, inputs, targets):
        # BCE + Dice
        bce_loss = self.bce(inputs, targets)
        inputs_sig = torch.sigmoid(inputs)
        intersection = (inputs_sig * targets).sum(dim=(2, 3))
        total = inputs_sig.sum(dim=(2, 3)) + targets.sum(dim=(2, 3))
        dice = 1 - (2.0 * intersection + 1e-6) / (total + 1e-6)
        dice_loss = dice.mean()
        dicebce = bce_loss + dice_loss

        # Lovász
        lov_loss = lovasz_hinge(inputs, targets)

        return self.dice_w * dicebce + self.lovasz_w * lov_loss


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


# ─── Augmentations (PRO) ────────────────────────────────────────────────────
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
            A.OpticalDistortion(distort_limit=0.05, p=0.5),
        ], p=0.3),
        A.OneOf([
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05, p=1),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1),
        ], p=0.5),
        A.OneOf([
            A.GaussNoise(p=1),
            A.GaussianBlur(blur_limit=(3, 5), p=1),
        ], p=0.3),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])


def get_val_transforms(img_size):
    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])


# ─── Metrics ─────────────────────────────────────────────────────────────────
@torch.no_grad()
def compute_iou(preds, targets, threshold=0.5):
    preds = (torch.sigmoid(preds) > threshold).float()
    intersection = (preds * targets).sum(dim=(2, 3))
    union = preds.sum(dim=(2, 3)) + targets.sum(dim=(2, 3)) - intersection
    iou = (intersection + 1e-6) / (union + 1e-6)
    return iou.mean().item()


# ─── Training with AMP ──────────────────────────────────────────────────────
def train_one_epoch(model, loader, criterion, optimizer, scheduler, scaler, device):
    model.train()
    running_loss, running_iou, n = 0.0, 0.0, 0

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
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=CFG.GRAD_CLIP)
        scaler.step(optimizer)
        scaler.update()

        if scheduler is not None and not isinstance(scheduler, torch.optim.swa_utils.SWALR):
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
    running_loss, running_iou, n = 0.0, 0.0, 0

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


# ─── Post-Processing ────────────────────────────────────────────────────────
def postprocess_mask(mask_binary):
    """
    Segmentation maskani tozalash:
    1. Kichik shovqinlarni olib tashlash
    2. Teshiklarni to'ldirish
    """
    mask = mask_binary.astype(np.uint8)

    # Morphological closing — teshiklarni to'ldirish
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # Kichik komponentlarni olib tashlash (shovqin)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels > 1:
        # Faqat eng katta componentni qoldirish (background dan tashqari)
        largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        mask = (labels == largest).astype(np.uint8)

    return mask


# ─── TTA for Segmentation ───────────────────────────────────────────────────
@torch.no_grad()
def predict_mask_tta(model, img_np, img_size, device):
    """
    Segmentation TTA: 4 yo'nalishda bashorat → averaging → threshold.
    """
    model.eval()
    h_orig, w_orig = img_np.shape[:2]

    normalize = A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

    def predict_single(image):
        t = normalize(image=image)["image"].unsqueeze(0).to(device)
        with torch.amp.autocast('cuda'):
            logits = model(t)
        return torch.sigmoid(logits.float()).squeeze(0).squeeze(0).cpu().numpy()

    # 4 xil bashorat
    preds = []
    preds.append(predict_single(img_np))                                      # original
    preds.append(np.fliplr(predict_single(np.fliplr(img_np).copy())))        # h-flip
    preds.append(np.flipud(predict_single(np.flipud(img_np).copy())))        # v-flip
    preds.append(np.rot90(predict_single(np.rot90(img_np, 1).copy()), -1))   # rotate90

    # O'rtacha
    avg_pred = np.mean(preds, axis=0)

    # Threshold
    binary_mask = (avg_pred > 0.5).astype(np.uint8)

    # Asl o'lchamga qaytarish
    binary_mask = cv2.resize(binary_mask, (w_orig, h_orig), interpolation=cv2.INTER_NEAREST)

    # Post-processing
    binary_mask = postprocess_mask(binary_mask)

    return binary_mask * 255  # 0/255 formatda saqlash uchun


# ─── Main ────────────────────────────────────────────────────────────────────
def main():
    seed_everything(CFG.SEED)
    os.makedirs(CFG.MODEL_SAVE_DIR, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  🏆 PRO Segmentation Training")
    print(f"  Device : {CFG.DEVICE}")
    if CFG.DEVICE == "cuda":
        print(f"  GPU    : {torch.cuda.get_device_name(0)}")
    print(f"  Encoder: {CFG.ENCODER}")
    print(f"  ImgSize: {CFG.IMG_SIZE}")
    print(f"  🔥 Lovász + DiceBCE + SWA + TTA + Post-processing")
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
        val_dataset, batch_size=CFG.BATCH_SIZE,
        shuffle=False, num_workers=CFG.NUM_WORKERS,
        pin_memory=True
    )

    # Model
    model = smp.UnetPlusPlus(
        encoder_name=CFG.ENCODER,
        encoder_weights=CFG.ENCODER_WEIGHTS,
        in_channels=3, classes=1, activation=CFG.ACTIVATION,
    ).to(CFG.DEVICE)

    criterion = CombinedLoss(lovasz_weight=CFG.LOVASZ_WEIGHT, dice_weight=CFG.DICE_WEIGHT)
    optimizer = optim.AdamW(model.parameters(), lr=CFG.LR, weight_decay=CFG.WEIGHT_DECAY)
    scaler = torch.amp.GradScaler('cuda')

    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=CFG.LR,
        total_steps=len(train_loader) * CFG.SWA_START,
        pct_start=0.1, anneal_strategy='cos',
    )

    swa_model = torch.optim.swa_utils.AveragedModel(model)
    swa_scheduler = torch.optim.swa_utils.SWALR(optimizer, swa_lr=CFG.SWA_LR)

    best_iou = 0.0
    start_time = time.time()

    for epoch in range(1, CFG.EPOCHS + 1):
        t0 = time.time()
        is_swa = epoch >= CFG.SWA_START
        print(f"\nEpoch {epoch}/{CFG.EPOCHS}" + (" [SWA]" if is_swa else ""))

        current_scheduler = swa_scheduler if is_swa else scheduler

        train_loss, train_iou = train_one_epoch(
            model, train_loader, criterion, optimizer, current_scheduler, scaler, CFG.DEVICE
        )

        if is_swa:
            swa_model.update_parameters(model)
            swa_scheduler.step()

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

    # SWA finalize
    print("\n🔄 SWA: Updating batch normalization...")
    torch.optim.swa_utils.update_bn(train_loader, swa_model, device=CFG.DEVICE)

    swa_val_loss, swa_val_iou = validate(swa_model, val_loader, criterion, CFG.DEVICE)
    print(f"  SWA Val IoU: {swa_val_iou:.4f} (vs Best: {best_iou:.4f})")

    if swa_val_iou > best_iou:
        best_iou = swa_val_iou
        torch.save({
            "model_state_dict": swa_model.module.state_dict(),
            "encoder": CFG.ENCODER,
            "img_size": CFG.IMG_SIZE,
            "val_iou": best_iou,
            "epoch": "SWA",
        }, os.path.join(CFG.MODEL_SAVE_DIR, "best_model.pth"))
        print(f"  ✅ SWA model saved! IoU: {best_iou:.4f}")

    print(f"\n{'='*60}")
    print(f"  ✅ Training Done! Best IoU: {best_iou:.4f}")
    print(f"  ⏱️ {(time.time()-start_time)/60:.1f} min")
    print(f"{'='*60}")

    # ─── Generate Test Masks with TTA ────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  🎯 Generating Test Masks with TTA + Post-processing...")
    print(f"{'='*60}")

    ckpt = torch.load(os.path.join(CFG.MODEL_SAVE_DIR, "best_model.pth"), map_location=CFG.DEVICE)
    model_inf = smp.UnetPlusPlus(
        encoder_name=ckpt["encoder"],
        encoder_weights=None,
        in_channels=3, classes=1, activation=None,
    )
    model_inf.load_state_dict(ckpt["model_state_dict"])
    model_inf.to(CFG.DEVICE).eval()

    # Test images
    test_files = sorted([f for f in os.listdir(CFG.TEST_IMG_DIR)
                         if f.lower().endswith((".png", ".jpg", ".jpeg"))])
    print(f"  Test images: {len(test_files)}")

    # Jamoa nomi papkasi
    TEAM_NAME = "TeamName"  # ⚠️ O'ZINGIZNING JAMOA NOMINI YOZING!
    output_dir = os.path.join(CFG.MODEL_SAVE_DIR, TEAM_NAME)
    os.makedirs(output_dir, exist_ok=True)

    for fname in tqdm(test_files, desc="  TTA Predict"):
        img = np.array(Image.open(os.path.join(CFG.TEST_IMG_DIR, fname)).convert("RGB"))

        mask = predict_mask_tta(model_inf, img, ckpt["img_size"], CFG.DEVICE)

        # Save as PNG with same name
        out_name = os.path.splitext(fname)[0] + ".png"
        Image.fromarray(mask.astype(np.uint8)).save(os.path.join(output_dir, out_name))

    print(f"\n  ✅ Saved {len(test_files)} masks to {output_dir}")
    print(f"  📁 Ready for submission!")


main()
