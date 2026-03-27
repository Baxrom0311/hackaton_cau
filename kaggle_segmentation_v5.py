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

def resolve_existing_dir(candidates, description):
    for candidate in candidates:
        if os.path.isdir(candidate):
            return candidate
    checked = "\n".join(f"  - {candidate}" for candidate in candidates)
    raise FileNotFoundError(f"{description} not found. Checked:\n{checked}")

# ─── Configuration ───────────────────────────────────────────────────────────
class CFG:
    BASE = "/kaggle/input/datasets/baxrom0311/main-dataset/Main hackathon dataset"
    TRAIN_IMG_CANDIDATES = [
        f"{BASE}/Segmentation/training/images",
        f"{BASE}/segmentation/training/images",
    ]
    TRAIN_MASK_CANDIDATES = [
        f"{BASE}/Segmentation/training/masks",
        f"{BASE}/segmentation/training/masks",
    ]
    VAL_IMG_CANDIDATES = [
        f"{BASE}/Segmentation/validation/images",
        f"{BASE}/segmentation/validation/images",
    ]
    VAL_MASK_CANDIDATES = [
        f"{BASE}/Segmentation/validation/masks",
        f"{BASE}/segmentation/validation/masks",
    ]
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
    EARLY_STOP_PATIENCE = 7

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
        A.OneOf([
            A.CLAHE(clip_limit=4.0, p=1.0),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, p=1.0),
        ], p=0.3),
        A.ElasticTransform(alpha=50, sigma=5, p=0.15),
        A.CoarseDropout(num_holes_range=(1, 4), hole_height_range=(8, 16), hole_width_range=(8, 16), p=0.15), # Removed fill_value
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
        # Augmentation is handled by Albumentations only (no double-flip!)
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
    amp_enabled = str(device).startswith("cuda")
    amp_device = "cuda" if amp_enabled else "cpu"
    pbar = tqdm(loader, desc="  Train", leave=False)
    
    for i, (images, masks) in enumerate(pbar):
        images, masks = preprocessor(images, masks, is_train=True)
        optimizer.zero_grad()
        with torch.amp.autocast(amp_device, enabled=amp_enabled):
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
    amp_enabled = str(device).startswith("cuda")
    amp_device = "cuda" if amp_enabled else "cpu"
    for images, masks in tqdm(loader, desc="  Valid", leave=False):
        images, masks = preprocessor(images, masks, is_train=False)
        with torch.amp.autocast(amp_device, enabled=amp_enabled):
            # TTA: Original + Horizontal + Vertical Flips
            out1 = model(images)
            out2 = torch.flip(model(torch.flip(images, dims=[3])), dims=[3])
            out3 = torch.flip(model(torch.flip(images, dims=[2])), dims=[2])
            outputs = (out1 + out2 + out3) / 3.0
            loss = criterion(outputs, masks)
        bs = images.size(0)
        running_loss += loss.item() * bs
        running_iou += compute_iou(outputs.float(), masks) * bs
        n += bs
    return running_loss / n, running_iou / n

@torch.no_grad()
def find_best_threshold(model, loader, device, preprocessor):
    """Sweep thresholds 0.3-0.7 to find optimal IoU threshold."""
    model.eval()
    amp_enabled = str(device).startswith("cuda")
    amp_device = "cuda" if amp_enabled else "cpu"
    all_preds, all_masks = [], []
    for images, masks in tqdm(loader, desc="  Threshold search", leave=False):
        images, masks = preprocessor(images, masks, is_train=False)
        with torch.amp.autocast(amp_device, enabled=amp_enabled):
            out1 = model(images)
            out2 = torch.flip(model(torch.flip(images, dims=[3])), dims=[3])
            out3 = torch.flip(model(torch.flip(images, dims=[2])), dims=[2])
            outputs = (out1 + out2 + out3) / 3.0
        all_preds.append(torch.sigmoid(outputs.float()).cpu())
        all_masks.append(masks.cpu())
    
    preds = torch.cat(all_preds)
    targets = torch.cat(all_masks)
    
    best_th, best_iou = 0.5, 0.0
    for th in [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]:
        p = (preds > th).float()
        inter = (p * targets).sum(dim=(2, 3))
        uni = p.sum(dim=(2, 3)) + targets.sum(dim=(2, 3)) - inter
        iou = ((inter + 1e-6) / (uni + 1e-6)).mean().item()
        print(f"    Threshold {th:.2f} → IoU: {iou:.4f}")
        if iou > best_iou:
            best_iou = iou
            best_th = th
    print(f"  ✅ Best threshold: {best_th:.2f} (IoU: {best_iou:.4f})")
    return best_th

# ─── MAIN ────────────────────────────────────────────────────────────────────
def main():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    seed_everything(CFG.SEED)
    os.makedirs(CFG.MODEL_SAVE_DIR, exist_ok=True)
    train_img_dir = resolve_existing_dir(CFG.TRAIN_IMG_CANDIDATES, "Segmentation training images")
    train_mask_dir = resolve_existing_dir(CFG.TRAIN_MASK_CANDIDATES, "Segmentation training masks")
    val_img_dir = resolve_existing_dir(CFG.VAL_IMG_CANDIDATES, "Segmentation validation images")
    val_mask_dir = resolve_existing_dir(CFG.VAL_MASK_CANDIDATES, "Segmentation validation masks")

    print(f"\n{'='*60}")
    print(f"  🚀 V5 Kaggle Segmentation (Optimized)")
    print(f"  ⚡️ B2 + Robust Resize (224x224)")
    print(f"{'='*60}\n")
    print(f"  📁 Train images: {train_img_dir}")
    print(f"  📁 Val   images: {val_img_dir}")

    # Clear memory from previous runs
    gc.collect()
    torch.cuda.empty_cache()

    train_ds = SegmentationDataset(train_img_dir, train_mask_dir, get_train_transforms(CFG.IMG_SIZE))
    val_ds = SegmentationDataset(val_img_dir, val_mask_dir, get_val_transforms(CFG.IMG_SIZE))

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
    scaler = torch.amp.GradScaler('cuda', enabled=str(CFG.DEVICE).startswith("cuda"))

    import math
    steps_per_epoch = math.ceil(len(train_loader) / CFG.GRAD_ACCUM_STEPS)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=CFG.MAX_LR, epochs=CFG.EPOCHS, steps_per_epoch=steps_per_epoch)

    preprocessor = GPUSegPreprocessor(CFG.DEVICE)
    best_iou = 0.0
    no_improve_count = 0
    start_time = time.time()

    for epoch in range(1, CFG.EPOCHS + 1):
        t0 = time.time()
        print(f"\nEpoch {epoch}/{CFG.EPOCHS}")

        train_loss, train_iou = train_one_epoch(model, train_loader, criterion, optimizer, scheduler, scaler, CFG.DEVICE, preprocessor, CFG)
        val_loss, val_iou = validate(model, val_loader, criterion, CFG.DEVICE, preprocessor)

        print(f"  Train Loss: {train_loss:.4f} | IoU: {train_iou:.4f}")
        print(f"  Val   Loss: {val_loss:.4f} | IoU: {val_iou:.4f} | ⏱️ {time.time()-t0:.0f}s")

        gap = train_iou - val_iou
        if gap > 0.15:
            print(f"  ⚠️ Overfitting alert! Train-Val IoU gap: {gap:.4f}")

        if val_iou > best_iou:
            best_iou = val_iou
            no_improve_count = 0
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
        else:
            no_improve_count += 1
            print(f"  ⏸️ No improvement ({no_improve_count}/{CFG.EARLY_STOP_PATIENCE})")
            if no_improve_count >= CFG.EARLY_STOP_PATIENCE:
                print(f"  🛑 Early stopping at epoch {epoch}!")
                break

        gc.collect()
        torch.cuda.empty_cache()

    # ── Threshold Optimization ──
    print(f"\n🔍 Searching for optimal threshold on Validation set...")
    best_th = find_best_threshold(model, val_loader, CFG.DEVICE, preprocessor)
    
    # Re-save best model with optimal threshold
    best_ckpt = torch.load(os.path.join(CFG.MODEL_SAVE_DIR, "best_model.pth"), map_location=CFG.DEVICE, weights_only=False)
    best_ckpt["best_threshold"] = best_th
    torch.save(best_ckpt, os.path.join(CFG.MODEL_SAVE_DIR, "best_model.pth"))
    print(f"  ✅ Threshold {best_th:.2f} saved to checkpoint")

    print(f"\n{'='*60}\n  ✅ Training Done! Best IoU: {best_iou:.4f} | Optimal Threshold: {best_th:.2f} | Total ⏱️ {(time.time()-start_time)/60:.1f} min\n{'='*60}")

if __name__ == "__main__":
    main()
