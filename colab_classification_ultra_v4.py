# ============================================================
# AI Healthcare Hackathon 2026 — V4 ULTRA Classification
# ============================================================
# 🚨 FASTAI & KAGGLE TRICKS APPLIED:
# ✅ cv2 o'qish (Tezkor I/O)
# ✅ torch.compile (Yadroviy tezlashtirish PyTorch 2.0+)
# ✅ Channels_last xotira formati (NVidia TensorCores uchun)
# ✅ OneCycleLR (Super-Convergence: 40 epochlik ishni 25 da qiladi)
# ✅ gc.collect() & empty_cache() (OOM oldini olish)
# ============================================================

from google.colab import drive
drive.mount('/content/drive')

import os, gc, time, random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import timm
import numpy as np
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tqdm import tqdm

# ─── Configuration ───────────────────────────────────────────────────────────
class CFG:
    BASE = "/content/drive/MyDrive/Main hackathon dataset"
    TRAIN_DIR = f"{BASE}/classification/train"
    TEST_DIR  = f"{BASE}/classification/test"
    MODEL_SAVE_DIR = "/content/drive/MyDrive/hackaton_models/classification_v4"

    MODEL_NAME = "tf_efficientnet_b5_ns"
    NUM_CLASSES = 12
    PRETRAINED = True

    EPOCHS = 25                       # Super-Convergence sababli qisqartirildi
    BATCH_SIZE = 16
    GRAD_ACCUM_STEPS = 2
    MAX_LR = 3e-4                     # B5 uchun 1e-3 juda katta bo'lishi mumkin (Divergence oldini olish uchun 3e-4)
    WEIGHT_DECAY = 1e-4
    NUM_WORKERS = 2
    IMG_SIZE = 512
    SEED = 42
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    LABEL_SMOOTHING = 0.1
    MIXUP_ALPHA = 0.4
    CUTMIX_ALPHA = 1.0
    MIXUP_PROB = 0.2                  # Biopsiya (hujayra) rasmlari uchun Mixup/CutMix kamaytirildi! (0.7 dan 0.2 ga)
    FOCAL_GAMMA = 2.0
    GRAD_CLIP = 1.0
    VAL_RATIO = 0.15

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True  # Conv algoritmlarini tezlashtiradi

# ─── Focal Loss ──────────────────────────────────────────────────────────────
class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=2.0, label_smoothing=0.0):
        super().__init__()
        self.weight = weight
        self.gamma = gamma
        self.label_smoothing = label_smoothing

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, weight=self.weight,
                                  label_smoothing=self.label_smoothing, reduction='none')
        pt = torch.exp(-ce_loss)
        return (((1 - pt) ** self.gamma) * ce_loss).mean()

# ─── Mixup & Cutmix ─────────────────────────────────────────────────────────
def mixup_data(x, y, alpha=0.4):
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    index = torch.randperm(x.size(0), device=x.device)
    return lam * x + (1 - lam) * x[index], y, y[index], lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

# ─── Dataset (CV2 Bilan Tezlashtirilgan) ────────────────────────────────────
class BiopDataset(Dataset):
    def __init__(self, paths, labels, transform=None):
        self.paths = paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        # CV2 C++ darajasida o'qiydi (PIL dan 2x tezroq)
        img = cv2.imread(self.paths[idx], cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        label = self.labels[idx]
        if self.transform:
            img = self.transform(image=img)["image"]
        return img, label

# ─── Augmentations ──────────────────────────────────────────────────────────
def get_train_transforms(sz):
    return A.Compose([
        A.Resize(sz, sz),
        A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.5), A.RandomRotate90(p=0.5),
        A.ElasticTransform(alpha=120, sigma=6.0, alpha_affine=3.6, p=0.3),
        A.Affine(scale=(0.9, 1.1), rotate=(-15, 15), p=0.5),
        A.CoarseDropout(num_holes_range=(1, 3), hole_height_range=(0.05, 0.15), hole_width_range=(0.05, 0.15), fill=0, p=0.4),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

def get_val_transforms(sz):
    return A.Compose([
        A.Resize(sz, sz),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

# ─── Training Loop ──────────────────────────────────────────────────────────
def train_one_epoch(model, loader, criterion, optimizer, scheduler, scaler, device, epoch, cfg):
    model.train()
    running_loss, preds_all, labels_all = 0.0, [], []
    optimizer.zero_grad(set_to_none=True)
    pbar = tqdm(loader, desc=f"  Train E{epoch}", leave=False)
    
    for i, (images, labels) in enumerate(pbar):
        # CHANNELS_LAST Engineering hiylasi
        images = images.to(device, memory_format=torch.channels_last)
        labels = labels.to(device)

        use_mix = np.random.random() < cfg.MIXUP_PROB
        if use_mix:
            images, y_a, y_b, lam = mixup_data(images, labels, cfg.MIXUP_ALPHA)

        with torch.amp.autocast('cuda'):
            outputs = model(images)
            loss = mixup_criterion(criterion, outputs, y_a, y_b, lam) if use_mix else criterion(outputs, labels)
            loss = loss / cfg.GRAD_ACCUM_STEPS

        scaler.scale(loss).backward()
        
        if (i + 1) % cfg.GRAD_ACCUM_STEPS == 0 or (i + 1) == len(loader):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg.GRAD_CLIP)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step() # OneCycleLR har qadamda o'zgaradi

        running_loss += (loss.item() * cfg.GRAD_ACCUM_STEPS) * images.size(0)
        preds_all.extend(outputs.argmax(dim=1).cpu().numpy())
        labels_all.extend(labels.cpu().numpy())
        pbar.set_postfix(loss=f"{loss.item()*cfg.GRAD_ACCUM_STEPS:.3f}", lr=f"{scheduler.get_last_lr()[0]:.6f}")

    return running_loss / len(loader.dataset), accuracy_score(labels_all, preds_all)

@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    running_loss, preds_all, labels_all = 0.0, [], []
    for images, labels in tqdm(loader, desc="  Valid", leave=False):
        images = images.to(device, memory_format=torch.channels_last)
        labels = labels.to(device)
        with torch.amp.autocast('cuda'):
            outputs = model(images)
            loss = criterion(outputs, labels)
        running_loss += loss.item() * images.size(0)
        preds_all.extend(outputs.argmax(dim=1).cpu().numpy())
        labels_all.extend(labels.cpu().numpy())
    return running_loss / len(loader.dataset), accuracy_score(labels_all, preds_all)


# ─── MAIN ────────────────────────────────────────────────────────────────────
def main():
    seed_everything(CFG.SEED)
    os.makedirs(CFG.MODEL_SAVE_DIR, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  🏆 V4 ULTRA Classification (Super-Convergence)")
    print(f"  ⚡️ torch.compile + channels_last + cv2 + OneCycleLR")
    print(f"{'='*60}\n")

    paths, labels = [], []
    for class_id in sorted(os.listdir(CFG.TRAIN_DIR), key=lambda x: int(x)):
        d = os.path.join(CFG.TRAIN_DIR, class_id)
        if not os.path.isdir(d): continue
        for f in os.listdir(d):
            if f.lower().endswith((".png", ".jpg")):
                paths.append(os.path.join(d, f))
                labels.append(int(class_id))

    t_paths, v_paths, t_lbls, v_lbls = train_test_split(paths, labels, test_size=CFG.VAL_RATIO, stratify=labels, random_state=CFG.SEED)
    
    train_loader = DataLoader(BiopDataset(t_paths, t_lbls, get_train_transforms(CFG.IMG_SIZE)), 
                              batch_size=CFG.BATCH_SIZE, shuffle=True, num_workers=CFG.NUM_WORKERS, pin_memory=True, drop_last=True)
    val_loader = DataLoader(BiopDataset(v_paths, v_lbls, get_val_transforms(CFG.IMG_SIZE)), 
                            batch_size=CFG.BATCH_SIZE, shuffle=False, num_workers=CFG.NUM_WORKERS, pin_memory=True)

    counts = np.bincount(t_lbls, minlength=CFG.NUM_CLASSES).astype(float)
    cw = torch.FloatTensor((1.0 / (counts + 1e-6)) / (1.0 / (counts + 1e-6)).sum() * CFG.NUM_CLASSES).to(CFG.DEVICE)

    model = timm.create_model(CFG.MODEL_NAME, pretrained=CFG.PRETRAINED, num_classes=CFG.NUM_CLASSES)
    
    # ⚡️ Hardware Optimizations
    model = model.to(CFG.DEVICE, memory_format=torch.channels_last)
    if int(torch.__version__.split('.')[0]) >= 2 and sys.platform != "win32":
        print("  🔥 Compiling model with torch.compile()...")
        model = torch.compile(model)

    criterion = FocalLoss(weight=cw, gamma=CFG.FOCAL_GAMMA, label_smoothing=CFG.LABEL_SMOOTHING)
    optimizer = optim.AdamW(model.parameters(), lr=CFG.MAX_LR, weight_decay=CFG.WEIGHT_DECAY, foreach=True)
    scaler = torch.amp.GradScaler('cuda')

    # ⚡️ Algorithmic Optimization: Super-Convergence
    steps_per_epoch = len(train_loader) // CFG.GRAD_ACCUM_STEPS
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=CFG.MAX_LR, epochs=CFG.EPOCHS, steps_per_epoch=steps_per_epoch)

    best_val_acc = 0.0
    start_time = time.time()

    for epoch in range(1, CFG.EPOCHS + 1):
        t0 = time.time()
        print(f"\nEpoch {epoch}/{CFG.EPOCHS}")

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, scheduler, scaler, CFG.DEVICE, epoch, CFG)
        val_loss, val_acc = validate(model, val_loader, criterion, CFG.DEVICE)

        print(f"  Train Loss: {train_loss:.4f} | Acc: {train_acc:.4f}")
        print(f"  Val   Loss: {val_loss:.4f} | Acc: {val_acc:.4f} | ⏱️ {time.time()-t0:.0f}s")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            # Uncompiled holidagi modelni saqlaymiz (.pth inference scripts ishlashi uchun)
            state_dict = model._orig_mod.state_dict() if hasattr(model, "_orig_mod") else model.state_dict()
            torch.save({
                "model_state_dict": state_dict,
                "model_name": CFG.MODEL_NAME,
                "num_classes": CFG.NUM_CLASSES,
                "img_size": CFG.IMG_SIZE,
                "val_acc": best_val_acc,
            }, os.path.join(CFG.MODEL_SAVE_DIR, "best_model.pth"))
            print(f"  ✅ Best model! Acc: {best_val_acc:.4f}")

        # ⚡️ Xotira tozalash (Memory Optimization)
        gc.collect()
        torch.cuda.empty_cache()

    print(f"\n{'='*60}\n  ✅ V4 Training Done! Best Acc: {best_val_acc:.4f} | Total ⏱️ {(time.time()-start_time)/60:.1f} min\n{'='*60}")

if __name__ == "__main__":
    import sys
    main()
