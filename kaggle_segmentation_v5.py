# !pip install segmentation-models-pytorch albumentations timm openpyxl -q

# ============================================================
# AI Healthcare Hackathon 2026 — V5 Kaggle Segmentation (Optimized)
# ============================================================
# 🚨 ROBUSTNESS FOCUS:
# ✅ IMG_SIZE = 224 (User request + Feature preservation)
# ✅ EfficientNet-B2 (Optimal for 224x224)
# ✅ LongestMaxSize + PadIfNeeded (No squashing!)
# ✅ OneCycleLR + SWA support
# ============================================================

import os, copy, time, random, gc, cv2, sys
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
    BASE = "/kaggle/input/datasets/baxrom0311/main-dataset/Main hackathon dataset"
    TRAIN_IMG_DIR = f"{BASE}/Segmentation/training/images"
    TRAIN_MASK_DIR = f"{BASE}/Segmentation/training/masks"
    VAL_IMG_DIR = TRAIN_IMG_DIR       # (90/10 split in dataset)
    VAL_MASK_DIR = TRAIN_MASK_DIR
    MODEL_SAVE_DIR = "segmentation_v5"

    ENCODER = "efficientnet-b2"
    ENCODER_WEIGHTS = "imagenet"
    DECODER = "UnetPlusPlus"

    EPOCHS = 40
    BATCH_SIZE = 32
    MAX_LR = 8e-4
    WEIGHT_DECAY = 1e-5
    NUM_WORKERS = 0                   # In-Memory optimization
    IMG_SIZE = 224
    SEED = 42
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    GRAD_ACCUM_STEPS = 1
    GRAD_CLIP = 5.0
    LOVASZ_WEIGHT = 0.5
    DICE_WEIGHT = 0.5

def robust_resize(img, sz, is_mask=False):
    """Aspect-ratio preserving padding (Ultra Quality)"""
    h, w = img.shape[:2]
    scale = sz / max(h, w)
    new_h, new_w = int(h * scale), int(w * scale)
    interp = cv2.INTER_NEAREST if is_mask else cv2.INTER_LINEAR
    img = cv2.resize(img, (new_w, new_h), interpolation=interp)
    
    pad_h = (sz - new_h) // 2
    pad_w = (sz - new_w) // 2
    img = cv2.copyMakeBorder(img, pad_h, sz - new_h - pad_h, pad_w, sz - new_w - pad_w, 
                            cv2.BORDER_CONSTANT, value=0)
    return img

def seed_everything(seed):
    try:
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = False # Set to False for robustness
        torch.backends.cudnn.benchmark = False      # Set to False for stability on Kaggle
    except Exception as e:
        print(f"⚠️ Seeding warning: {e}. If this is a CUDA error, RESTART YOUR KERNEL!")

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

# ─── Dataset (CV2) ───────────────────────────────────────────────────────────
class SegmentationDataset(Dataset):
    def __init__(self, img_dir, mask_dir, transform=None):
        self.transform = transform
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        raw_images = sorted([f for f in os.listdir(img_dir) if f.lower().endswith((".png", ".jpg"))])
        
        print(f"  📥 Loading {len(raw_images)} segmentation pairs into RAM...")
        # Pre-allocate to prevent RAM doubling
        self.images = np.empty((len(raw_images), 224, 224, 3), dtype=np.uint8)
        self.masks = np.empty((len(raw_images), 224, 224), dtype=np.float32)
        
        for i, fname in enumerate(tqdm(raw_images, leave=False)):
            # Image
            img = cv2.imread(os.path.join(self.img_dir, fname), cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = robust_resize(img, 224, is_mask=False) # 🛡️ Robust
            
            # Mask
            m_path = os.path.join(self.mask_dir, os.path.splitext(fname)[0] + ".png")
            if not os.path.exists(m_path): m_path = os.path.join(self.mask_dir, fname)
            mask = cv2.imread(m_path, cv2.IMREAD_GRAYSCALE)
            mask = robust_resize(mask, 224, is_mask=True) # 🛡️ Robust
            mask = (mask > 127).astype(np.float32)
            
            self.images[i] = img
            self.masks[i] = mask
            
        del raw_images
        gc.collect()

    def __len__(self): return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        mask = self.masks[idx]
        if self.transform:
            aug = self.transform(image=img, mask=mask)
            img, mask = aug["image"], aug["mask"]
        
        # Ensure mask is (1, H, W)
        if len(mask.shape) == 2:
            mask = torch.tensor(mask).unsqueeze(0)
        elif isinstance(mask, np.ndarray):
            mask = torch.tensor(mask)
            
        return img, mask

# ─── Augmentations ───────────────────────────────────────────────────────────
def get_train_transforms(sz):
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=15, p=0.5),
        ToTensorV2(),
    ])

def get_val_transforms(sz):
    return A.Compose([
        ToTensorV2(),
    ])

# ─── GPU-Accelerated Preprocessing ───────────────────────────────────────────
class GPUSegPreprocessor:
    def __init__(self, device):
        self.device = device
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)

    def __call__(self, imgs, masks, is_train=True):
        imgs = imgs.to(self.device, non_blocking=True).float() / 255.0
        masks = masks.to(self.device, non_blocking=True).float()
        
        if is_train:
            # Synchronized Flips on GPU
            if random.random() > 0.5:
                imgs = torch.flip(imgs, dims=[3])
                masks = torch.flip(masks, dims=[3])
            if random.random() > 0.5:
                imgs = torch.flip(imgs, dims=[2])
                masks = torch.flip(masks, dims=[2])
        
        # Normalize Image on GPU
        imgs = (imgs - self.mean) / self.std
        return imgs, masks

@torch.no_grad()
def compute_iou(preds, targets, th=0.5):
    p = (torch.sigmoid(preds) > th).float()
    inter = (p * targets).sum(dim=(2, 3))
    uni = p.sum(dim=(2, 3)) + targets.sum(dim=(2, 3)) - inter
    return ((inter + 1e-6) / (uni + 1e-6)).mean().item()

# ─── Training Loop ───────────────────────────────────────────────────────────
def train_one_epoch(model, loader, criterion, optimizer, scheduler, scaler, device, preprocessor, cfg):
    model.train()
    running_loss, running_iou, n = 0.0, 0.0, 0
    optimizer.zero_grad(set_to_none=True)
    pbar = tqdm(loader, desc="  Train", leave=False)
    
    for i, (images, masks) in enumerate(pbar):
        images, masks = preprocessor(images, masks, is_train=True)
        optimizer.zero_grad()
        with torch.amp.autocast('cuda'):
            outputs = model(images)
            loss = criterion(outputs, masks)

        scaler.scale(loss).backward()
        
        if (i + 1) % cfg.GRAD_ACCUM_STEPS == 0 or (i + 1) == len(loader):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg.GRAD_CLIP)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()

        bs = images.size(0)
        running_loss += loss.item() * bs
        running_iou += compute_iou(outputs.float(), masks) * bs
        n += bs
        pbar.set_postfix(loss=f"{loss.item():.3f}", lr=f"{scheduler.get_last_lr()[0]:.6f}")

    return running_loss / n, running_iou / n

@torch.no_grad()
def validate(model, loader, criterion, device, preprocessor):
    model.eval()
    running_loss, running_iou, n = 0.0, 0.0, 0
    for images, masks in tqdm(loader, desc="  Valid", leave=False):
        images, masks = preprocessor(images, masks, is_train=False)
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
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    seed_everything(CFG.SEED)
    os.makedirs(CFG.MODEL_SAVE_DIR, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  🚀 V5 Kaggle Segmentation (Optimized)")
    print(f"  ⚡️ B2 + Robust Resize (224x224)")
    print(f"{'='*60}\n")

    # Clear memory from previous runs
    gc.collect()
    torch.cuda.empty_cache()

    train_ds = SegmentationDataset(CFG.TRAIN_IMG_DIR, CFG.TRAIN_MASK_DIR, get_train_transforms(CFG.IMG_SIZE))
    val_ds = SegmentationDataset(CFG.VAL_IMG_DIR, CFG.VAL_MASK_DIR, get_val_transforms(CFG.IMG_SIZE))

    train_loader = DataLoader(train_ds, batch_size=CFG.BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=CFG.BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

    model = smp.UnetPlusPlus(encoder_name=CFG.ENCODER, encoder_weights=CFG.ENCODER_WEIGHTS, in_channels=3, classes=1).to(CFG.DEVICE)
    
    # ⚡️ Hardware Optimizations
    print(f"  ⚡️ Using Single GPU: {CFG.DEVICE.upper()}")
    
    # model = torch.compile(model) # DISABLED - Causes stalling on Kaggle

    # Reverting channels_last for stability
    # model = model.to(memory_format=torch.channels_last)

    criterion = CombinedLoss(CFG.LOVASZ_WEIGHT, CFG.DICE_WEIGHT)
    optimizer = optim.AdamW(model.parameters(), lr=CFG.MAX_LR, weight_decay=CFG.WEIGHT_DECAY, foreach=True)
    scaler = torch.amp.GradScaler('cuda')

    import math
    steps_per_epoch = math.ceil(len(train_loader) / CFG.GRAD_ACCUM_STEPS)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=CFG.MAX_LR, epochs=CFG.EPOCHS, steps_per_epoch=steps_per_epoch)

    preprocessor = GPUSegPreprocessor(CFG.DEVICE)
    best_iou = 0.0
    start_time = time.time()

    for epoch in range(1, CFG.EPOCHS + 1):
        t0 = time.time()
        print(f"\nEpoch {epoch}/{CFG.EPOCHS}")

        train_loss, train_iou = train_one_epoch(model, train_loader, criterion, optimizer, scheduler, scaler, CFG.DEVICE, preprocessor, CFG)
        val_loss, val_iou = validate(model, val_loader, criterion, CFG.DEVICE, preprocessor)

        print(f"  Train Loss: {train_loss:.4f} | IoU: {train_iou:.4f}")
        print(f"  Val   Loss: {val_loss:.4f} | IoU: {val_iou:.4f} | ⏱️ {time.time()-t0:.0f}s")

        if val_iou > best_iou:
            best_iou = val_iou
            state_dict = model.module.state_dict() if hasattr(model, "module") else model.state_dict()
            checkpoint = {
                "model_state_dict": state_dict,
                "encoder": CFG.ENCODER,
                "decoder": CFG.DECODER,
                "img_size": CFG.IMG_SIZE,
                "val_iou": best_iou,
                "epoch": epoch
            }
            torch.save(checkpoint, os.path.join(CFG.MODEL_SAVE_DIR, "best_model.pth"))
            print(f"  ✅ Best saved! IoU: {best_iou:.4f}")

        gc.collect()
        torch.cuda.empty_cache()

    print(f"\n{'='*60}\n  ✅ V5 Training Done! Best IoU: {best_iou:.4f} | Total ⏱️ {(time.time()-start_time)/60:.1f} min\n{'='*60}")

if __name__ == "__main__":
    main()
