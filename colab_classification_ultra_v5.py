# ============================================================
# AI Healthcare Hackathon 2026 — V5 ULTRA Classification (Optimized)
# ============================================================
# 🚨 ROBUSTNESS FOCUS:
# ✅ IMG_SIZE = 224 (Optimal for feature preservation)
# ✅ EfficientNet-B2 (Ideal balance for 224x224)
# ✅ LongestMaxSize + PadIfNeeded (Anti-distortion)
# ✅ TTA (Test Time Augmentation) during Validation
# ✅ Focal Loss (For imbalanced classes)
# ============================================================

import os, gc, time, random, math
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
import timm

# ─── Google Drive Support ────────────────────────────────────────────────────
try:
    if os.path.exists('/var/colab/hostname'):
        from google.colab import drive
        drive.mount('/content/drive')
        print("  ✅ Google Drive mounted.")
    else:
        print("  ℹ️ Not on Google Colab. Skipping Drive mount.")
except Exception as e:
    print(f"  ⚠️ Drive mount skipped or failed: {e}")

def discover_path(target_name="Main hackathon dataset"):
    """Robustly finds the dataset folder in Google Drive."""
    # Check common locations first
    candidates = [
        f"/content/drive/MyDrive/{target_name}",
        f"/content/drive/Shared drives/{target_name}",
        f"/content/drive/MyDrive/hackaton/{target_name}",
        f"/content/drive/MyDrive/hackathon/{target_name}",
    ]
    for c in candidates:
        if os.path.exists(c): return c
    
    # Recursive check (if not found in root)
    if os.path.exists("/content/drive/MyDrive"):
        for root, dirs, _ in os.walk("/content/drive/MyDrive"):
            if target_name in dirs:
                return os.path.join(root, target_name)
    return None

# ─── Configuration ───────────────────────────────────────────────────────────
_base_path = discover_path() or "/content/drive/MyDrive/Main hackathon dataset"

class CFG:
    BASE = _base_path
    TRAIN_DIR = f"{BASE}/classification/train"
    MODEL_SAVE_DIR = "/content/drive/MyDrive/hackaton_models/classification_v5"

    MODEL_NAME = "tf_efficientnet_b2.ns_jft_in1k"
    IMG_SIZE = 224
    BATCH_SIZE = 64
    EPOCHS = 40
    MAX_LR = 1e-3
    WEIGHT_DECAY = 1e-5
    NUM_CLASSES = 12                   # Unified count for 0-11 folders
    NUM_WORKERS = 0                   # In-Memory optimization
    SEED = 42
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    GRAD_ACCUM_STEPS = 1
    LABEL_SMOOTHING = 0.1

def robust_resize(img, sz):
    """Aspect-ratio preserving padding (Ultra Quality)"""
    h, w = img.shape[:2]
    scale = sz / max(h, w)
    new_h, new_w = int(h * scale), int(w * scale)
    img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    
    pad_h = (sz - new_h) // 2
    pad_w = (sz - new_w) // 2
    img = cv2.copyMakeBorder(img, pad_h, sz - new_h - pad_h, pad_w, sz - new_w - pad_w, 
                            cv2.BORDER_CONSTANT, value=0)
    return img

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

# ─── Dataset ────────────────────────────────────────────────────────────────
class ClassificationDataset(Dataset):
    def __init__(self, root_dir, transform=None, is_train=True):
        self.transform = transform
        raw_data = []
        for label in sorted(os.listdir(root_dir)):
            if label.isdigit():
                cls_dir = os.path.join(root_dir, label)
                for img_name in os.listdir(cls_dir):
                    if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                        raw_data.append((os.path.join(cls_dir, img_name), int(label)))
        
        # Simple Split (90% Train, 10% Val)
        random.seed(42)
        random.shuffle(raw_data)
        split = int(0.9 * len(raw_data))
        raw_data = raw_data[:split] if is_train else raw_data[split:]

        print(f"  📥 Loading {len(raw_data)} images into RAM (Train={is_train})...")
        self.images = np.empty((len(raw_data), 224, 224, 3), dtype=np.uint8)
        self.labels = np.empty(len(raw_data), dtype=np.int64)
        
        for i, (path, label) in enumerate(tqdm(raw_data, leave=False)):
            img = cv2.imread(path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = robust_resize(img, 224) # 🛡️ Robust Padding
            self.images[i] = img
            self.labels[i] = label
        
        del raw_data
        gc.collect()

    def __len__(self): return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        label = self.labels[idx]
        if self.transform:
            img = self.transform(image=img)["image"]
        return img, label

# ─── Transforms ──────────────────────────────────────────────────────────────
def get_train_transforms(sz):
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=15, p=0.5),
        A.OneOf([
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.5),
            A.GaussianBlur(blur_limit=(3, 7), p=0.5),
        ], p=0.3),
        ToTensorV2(),
    ])

def get_val_transforms(sz):
    return A.Compose([
        # Already resized in RAM
        ToTensorV2(),
    ])

# ─── GPU-Accelerated Preprocessing ───────────────────────────────────────────
class GPUPreprocessor:
    def __init__(self, device):
        self.device = device
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)

    def __call__(self, imgs, is_train=True):
        imgs = imgs.to(self.device, non_blocking=True).float() / 255.0
        if is_train:
            # Random Flips on GPU
            if random.random() > 0.5: imgs = torch.flip(imgs, dims=[3]) # Horizontal
            if random.random() > 0.5: imgs = torch.flip(imgs, dims=[2]) # Vertical
        
        # Normalize on GPU
        imgs = (imgs - self.mean) / self.std
        return imgs

# ─── Focal Loss ─────────────────────────────────────────────────────────────
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt)**self.gamma * ce_loss
        return focal_loss.mean()

# ─── Training Loop ──────────────────────────────────────────────────────────
def train_one_epoch(model, loader, criterion, optimizer, scheduler, device, preprocessor, cfg):
    model.train()
    running_loss, correct, n = 0.0, 0, 0
    pbar = tqdm(loader, desc="  Train", leave=False)
    for i, (imgs, labels) in enumerate(pbar):
        imgs = preprocessor(imgs, is_train=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad()
        
        with torch.amp.autocast('cuda'):
            outputs = model(imgs)
            loss = criterion(outputs, labels)
        
        loss.backward()
        if (i+1) % cfg.GRAD_ACCUM_STEPS == 0:
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        running_loss += loss.item() * imgs.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        n += imgs.size(0)
        pbar.set_postfix(loss=f"{loss.item():.3f}", acc=f"{correct/n:.3f}")
    return running_loss / n, correct / n

@torch.no_grad()
def validate(model, loader, criterion, device, preprocessor):
    model.eval()
    running_loss, correct, n = 0.0, 0, 0
    for imgs, labels in tqdm(loader, desc="  Valid", leave=False):
        imgs = preprocessor(imgs, is_train=False)
        labels = labels.to(device, non_blocking=True)
        
        # Simple TTA: Average original + Horizontal Flip
        outputs = model(imgs)
        outputs_flip = model(torch.flip(imgs, dims=[3]))
        outputs = (outputs + outputs_flip) / 2.0
        
        loss = criterion(outputs, labels)
        running_loss += loss.item() * imgs.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        n += imgs.size(0)
    return running_loss / n, correct / n

# ─── MAIN ────────────────────────────────────────────────────────────────────
def main():
    seed_everything(CFG.SEED)
    os.makedirs(CFG.MODEL_SAVE_DIR, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  🚀 V5 ULTRA Classification (Optimized)")
    print(f"  ⚡️ EfficientNet-B2 + Robust Padding + TTA")
    print(f"{'='*60}\n")

    train_ds = ClassificationDataset(CFG.TRAIN_DIR, get_train_transforms(CFG.IMG_SIZE), True)
    val_ds = ClassificationDataset(CFG.TRAIN_DIR, get_val_transforms(CFG.IMG_SIZE), False)
    
    train_loader = DataLoader(train_ds, batch_size=CFG.BATCH_SIZE, shuffle=True, num_workers=CFG.NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=CFG.BATCH_SIZE, shuffle=False, num_workers=CFG.NUM_WORKERS, pin_memory=True)

    model = timm.create_model(CFG.MODEL_NAME, pretrained=True, num_classes=CFG.NUM_CLASSES).to(CFG.DEVICE)
    model = torch.compile(model) if torch.__version__.startswith("2") else model

    criterion = FocalLoss()
    optimizer = optim.AdamW(model.parameters(), lr=CFG.MAX_LR, weight_decay=CFG.WEIGHT_DECAY)
    
    steps_per_epoch = math.ceil(len(train_loader) / CFG.GRAD_ACCUM_STEPS)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=CFG.MAX_LR, epochs=CFG.EPOCHS, steps_per_epoch=steps_per_epoch)

    preprocessor = GPUPreprocessor(CFG.DEVICE)
    best_acc = 0.0
    start_time = time.time()

    for epoch in range(1, CFG.EPOCHS + 1):
        t0 = time.time()
        print(f"\nEpoch {epoch}/{CFG.EPOCHS}")

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, scheduler, CFG.DEVICE, preprocessor, CFG)
        val_loss, val_acc = validate(model, val_loader, criterion, CFG.DEVICE, preprocessor)

        print(f"  Train Loss: {train_loss:.4f} | Acc: {train_acc:.4f}")
        print(f"  Val   Loss: {val_loss:.4f} | Acc: {val_acc:.4f} | ⏱️ {time.time()-t0:.0f}s")

        if val_acc > best_acc:
            best_acc = val_acc
            state_dict = model.state_dict()
            checkpoint = {
                "model_state_dict": state_dict,
                "model_name": CFG.MODEL_NAME,
                "num_classes": CFG.NUM_CLASSES,
                "img_size": CFG.IMG_SIZE,
                "val_acc": best_acc
            }
            torch.save(checkpoint, os.path.join(CFG.MODEL_SAVE_DIR, "best_model.pth"))
            print(f"  ✅ Best saved! Accuracy: {best_acc:.4f}")

        gc.collect()
        torch.cuda.empty_cache()

    print(f"\n{'='*60}\n  ✅ V5 Training Done! Best Acc: {best_acc:.4f} | Total ⏱️ {(time.time()-start_time)/60:.1f} min\n{'='*60}")

if __name__ == "__main__":
    main()
