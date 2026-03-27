# ============================================================
# AI Healthcare Hackathon 2026 — V4 ULTRA Segmentation
# ============================================================
# 🚨 FASTAI & KAGGLE TRICKS APPLIED:
# ✅ cv2 o'qish (Tezkor I/O)
# ✅ torch.compile (Yadroviy tezlashtirish PyTorch 2.0+)
# ✅ Channels_last xotira formati (NVidia TensorCores uchun)
# ✅ OneCycleLR (Super-Convergence: tez o'qish va converge)
# ✅ gc.collect() & empty_cache() (OOM oldini olish)
# ============================================================

from google.colab import drive
drive.mount('/content/drive')

import os, gc, time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import segmentation_models_pytorch as smp

# ─── Configuration ───────────────────────────────────────────────────────────
class CFG:
    BASE = "/content/drive/MyDrive/Main hackathon dataset"
    TRAIN_IMG_DIR = f"{BASE}/Segmentation/training/images"
    TRAIN_MASK_DIR = f"{BASE}/Segmentation/training/masks"
    VAL_IMG_DIR = f"{BASE}/Segmentation/validation/images"
    VAL_MASK_DIR = f"{BASE}/Segmentation/validation/masks"
    MODEL_SAVE_DIR = "/content/drive/MyDrive/hackaton_models/segmentation_v4"

    ENCODER = "timm-efficientnet-b5"
    ENCODER_WEIGHTS = "noisy-student"
    DECODER = "UnetPlusPlus"

    EPOCHS = 35                       # Super-Convergence tufaoyli 50 dan 35 ga tushirildi
    BATCH_SIZE = 8
    GRAD_ACCUM_STEPS = 2
    MAX_LR = 1.5e-3                   # OneCycleLR Peak LR
    WEIGHT_DECAY = 1e-4
    NUM_WORKERS = 2
    IMG_SIZE = 512
    SEED = 42
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    GRAD_CLIP = 1.0
    LOVASZ_WEIGHT = 0.5
    DICE_WEIGHT = 0.5

def seed_everything(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True

# ─── Loss Functions ──────────────────────────────────────────────────────────
def lovasz_grad(gt_sorted):
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1.0 - intersection / union
    if p > 1: jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard

def lovasz_hinge_flat(logits, labels):
    if len(labels) == 0: return logits.sum() * 0.0
    signs = 2.0 * labels.float() - 1.0
    errors = 1.0 - logits * signs
    errors_sorted, perm = torch.sort(errors, dim=0, descending=True)
    grad = lovasz_grad(labels[perm.data])
    return torch.dot(F.relu(errors_sorted), grad)

def lovasz_hinge(logits, labels):
    return torch.stack([lovasz_hinge_flat(l, lbl) for l, lbl in zip(logits.view(logits.size(0), -1), labels.view(labels.size(0), -1))]).mean()

class CombinedLoss(nn.Module):
    def __init__(self, lw=0.5, dw=0.5):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.lw = lw
        self.dw = dw

    def forward(self, inputs, targets):
        bce = self.bce(inputs, targets)
        inputs_s = torch.sigmoid(inputs)
        inter = (inputs_s * targets).sum(dim=(2, 3))
        tot = inputs_s.sum(dim=(2, 3)) + targets.sum(dim=(2, 3))
        dice = (1 - (2.0 * inter + 1e-6) / (tot + 1e-6)).mean()
        lov = lovasz_hinge(inputs, targets)
        return self.dw * (bce + dice) + self.lw * lov

# ─── Dataset (CV2) ──────────────────────────────────────────────────────────
class SegmentationDataset(Dataset):
    def __init__(self, img_dir, mask_dir, transform=None):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = sorted([f for f in os.listdir(img_dir) if f.lower().endswith((".png", ".jpg"))])

    def __len__(self): return len(self.images)

    def __getitem__(self, idx):
        fname = self.images[idx]
        img = cv2.imread(os.path.join(self.img_dir, fname), cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        m_path = os.path.join(self.mask_dir, os.path.splitext(fname)[0] + ".png")
        if not os.path.exists(m_path): m_path = os.path.join(self.mask_dir, fname)
        mask = cv2.imread(m_path, cv2.IMREAD_GRAYSCALE)
        mask = (mask > 127).astype(np.float32)

        if self.transform:
            aug = self.transform(image=img, mask=mask)
            img, mask = aug["image"], aug["mask"]
        return img, (mask.unsqueeze(0) if isinstance(mask, torch.Tensor) else torch.tensor(mask).unsqueeze(0))

# ─── Augmentations ──────────────────────────────────────────────────────────
def get_train_transforms(sz):
    return A.Compose([
        A.Resize(sz, sz),
        A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.5), A.RandomRotate90(p=0.5),
        A.ElasticTransform(alpha=120, sigma=6.0, alpha_affine=3.6, p=0.4),
        A.CLAHE(clip_limit=3.0, p=0.3), A.GridDistortion(p=0.3),
        A.Affine(scale=(0.8, 1.2), rotate=(-30, 30), p=0.4),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

def get_val_transforms(sz):
    return A.Compose([A.Resize(sz, sz), A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), ToTensorV2()])

@torch.no_grad()
def compute_iou(preds, targets, th=0.5):
    p = (torch.sigmoid(preds) > th).float()
    inter = (p * targets).sum(dim=(2, 3))
    uni = p.sum(dim=(2, 3)) + targets.sum(dim=(2, 3)) - inter
    return ((inter + 1e-6) / (uni + 1e-6)).mean().item()

# ─── Training Loop ──────────────────────────────────────────────────────────
def train_one_epoch(model, loader, criterion, optimizer, scheduler, scaler, device, cfg):
    model.train()
    running_loss, running_iou, n = 0.0, 0.0, 0
    optimizer.zero_grad(set_to_none=True)
    pbar = tqdm(loader, desc="  Train", leave=False)
    
    for i, (images, masks) in enumerate(pbar):
        # Channels_last tezlachitirovchi
        images = images.to(device, memory_format=torch.channels_last)
        masks = masks.to(device)

        with torch.amp.autocast('cuda'):
            outputs = model(images)
            loss = criterion(outputs, masks) / cfg.GRAD_ACCUM_STEPS

        scaler.scale(loss).backward()
        
        if (i + 1) % cfg.GRAD_ACCUM_STEPS == 0 or (i + 1) == len(loader):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg.GRAD_CLIP)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()

        bs = images.size(0)
        running_loss += (loss.item() * cfg.GRAD_ACCUM_STEPS) * bs
        running_iou += compute_iou(outputs.float(), masks) * bs
        n += bs
        pbar.set_postfix(loss=f"{loss.item()*cfg.GRAD_ACCUM_STEPS:.3f}", lr=f"{scheduler.get_last_lr()[0]:.6f}")

    return running_loss / n, running_iou / n

@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    running_loss, running_iou, n = 0.0, 0.0, 0
    for images, masks in tqdm(loader, desc="  Valid", leave=False):
        images = images.to(device, memory_format=torch.channels_last)
        masks = masks.to(device)
        with torch.amp.autocast('cuda'):
            outputs = model(images)
            loss = criterion(outputs, masks)
        bs = images.size(0)
        running_loss += loss.item() * bs
        running_iou += compute_iou(outputs.float(), masks) * bs
        n += bs
    return running_loss / n, running_iou / n

# ─── MAIN ────────────────────────────────────────────────────────────────────
def main():
    seed_everything(CFG.SEED)
    os.makedirs(CFG.MODEL_SAVE_DIR, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  🚀 V4 ULTRA Segmentation (Super-Convergence)")
    print(f"  ⚡️ torch.compile + channels_last + cv2 + OneCycleLR")
    print(f"{'='*60}\n")

    train_loader = DataLoader(SegmentationDataset(CFG.TRAIN_IMG_DIR, CFG.TRAIN_MASK_DIR, get_train_transforms(CFG.IMG_SIZE)), 
                              batch_size=CFG.BATCH_SIZE, shuffle=True, num_workers=CFG.NUM_WORKERS, pin_memory=True, drop_last=True)
    val_loader = DataLoader(SegmentationDataset(CFG.VAL_IMG_DIR, CFG.VAL_MASK_DIR, get_val_transforms(CFG.IMG_SIZE)), 
                            batch_size=CFG.BATCH_SIZE, shuffle=False, num_workers=CFG.NUM_WORKERS, pin_memory=True)

    model = smp.UnetPlusPlus(encoder_name=CFG.ENCODER, encoder_weights=CFG.ENCODER_WEIGHTS, in_channels=3, classes=1).to(CFG.DEVICE)
    
    # ⚡️ Hardware Optimizations
    model = model.to(memory_format=torch.channels_last)
    if int(torch.__version__.split('.')[0]) >= 2 and sys.platform != "win32":
        print("  🔥 Compiling model with torch.compile()...")
        model = torch.compile(model)

    criterion = CombinedLoss(CFG.LOVASZ_WEIGHT, CFG.DICE_WEIGHT)
    optimizer = optim.AdamW(model.parameters(), lr=CFG.MAX_LR, weight_decay=CFG.WEIGHT_DECAY, foreach=True)
    scaler = torch.amp.GradScaler('cuda')

    import math
    steps_per_epoch = math.ceil(len(train_loader) / CFG.GRAD_ACCUM_STEPS)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=CFG.MAX_LR, epochs=CFG.EPOCHS, steps_per_epoch=steps_per_epoch)

    best_iou = 0.0
    start_time = time.time()

    for epoch in range(1, CFG.EPOCHS + 1):
        t0 = time.time()
        print(f"\nEpoch {epoch}/{CFG.EPOCHS}")

        train_loss, train_iou = train_one_epoch(model, train_loader, criterion, optimizer, scheduler, scaler, CFG.DEVICE, CFG)
        val_loss, val_iou = validate(model, val_loader, criterion, CFG.DEVICE)

        print(f"  Train Loss: {train_loss:.4f} | IoU: {train_iou:.4f}")
        print(f"  Val   Loss: {val_loss:.4f} | IoU: {val_iou:.4f} | ⏱️ {time.time()-t0:.0f}s")

        if val_iou > best_iou:
            best_iou = val_iou
            state_dict = model._orig_mod.state_dict() if hasattr(model, "_orig_mod") else model.state_dict()
            torch.save({
                "model_state_dict": state_dict, "encoder": CFG.ENCODER,
                "img_size": CFG.IMG_SIZE, "val_iou": best_iou, "epoch": epoch
            }, os.path.join(CFG.MODEL_SAVE_DIR, "best_model.pth"))
            print(f"  ✅ Best saved! IoU: {best_iou:.4f}")

        # ⚡️ Memory Optimization
        gc.collect()
        torch.cuda.empty_cache()

    print(f"\n{'='*60}\n  ✅ V4 Training Done! Best IoU: {best_iou:.4f} | Total ⏱️ {(time.time()-start_time)/60:.1f} min\n{'='*60}")

if __name__ == "__main__":
    import sys
    main()
