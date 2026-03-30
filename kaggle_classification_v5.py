# !pip install segmentation-models-pytorch albumentations timm openpyxl -q

# ============================================================
# AI Healthcare Hackathon 2026 — V5 Kaggle Classification
# ============================================================
# 🛡️ ANTI-OVERFITTING DESIGN:
# ✅ IMG_SIZE = 224 (Matches small biopsy images ~128-256px)
# ✅ EfficientNet-B2 + Dropout (0.3)
# ✅ Robust Padding (aspect-ratio preserving)
# ✅ Focal Loss + Label Smoothing + Class Weights
# ✅ Medical-grade Augmentation (ColorJitter, CLAHE, ElasticTransform)
# ✅ Stratified Train/Val Split
# ✅ Early Stopping (patience=7)
# ============================================================

import json
import os, gc, time, random, math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from collections import Counter
from tqdm import tqdm
import timm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ─── Configuration ───────────────────────────────────────────────────────────
class CFG:
    BASE = "/kaggle/input/datasets/baxrom0311/main-dataset/Main hackathon dataset"
    TRAIN_DIR = f"{BASE}/classification/train"
    MODEL_SAVE_DIR = "classification_v6"
    SPLIT_MANIFEST_PATH = None
    DEFAULT_SPLIT_MANIFEST_CANDIDATES = [
        os.path.join(SCRIPT_DIR, "classification_hard_split_manifest.json"),
        os.path.join(os.getcwd(), "classification_hard_split_manifest.json"),
    ]

    MODEL_NAME = "tf_efficientnet_b4.ns_jft_in1k"
    IMG_SIZE = 380
    BATCH_SIZE = 32
    EPOCHS = 30
    MAX_LR = 5e-4
    WEIGHT_DECAY = 1e-4
    NUM_CLASSES = 12
    NUM_WORKERS = 0
    SEED = 42
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    GRAD_ACCUM_STEPS = 2
    LABEL_SMOOTHING = 0.15
    DROP_RATE = 0.4
    EARLY_STOP_PATIENCE = 8
    GRAD_CLIP = 5.0
    MIXUP_ALPHA = 0.4
    CUTMIX_ALPHA = 1.0
    CUTMIX_PROB = 0.5
    VAL_FRACTION = 0.1
    SPLIT_SEED = 42
    USE_WEIGHTED_SAMPLER = True
    SAMPLER_POWER = 0.5

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

def collect_classification_items(root_dir):
    items = []
    for label in sorted((entry for entry in os.listdir(root_dir) if entry.isdigit()), key=int):
        cls_dir = os.path.join(root_dir, label)
        for img_name in sorted(os.listdir(cls_dir)):
            if img_name.lower().endswith((".png", ".jpg", ".jpeg")):
                items.append((os.path.join(cls_dir, img_name), int(label)))
    return items

def build_classification_split(root_dir, val_fraction, seed):
    rng = random.Random(seed)
    by_class = {}
    for item in collect_classification_items(root_dir):
        by_class.setdefault(item[1], []).append(item)

    train_data, val_data = [], []
    for cls_id in sorted(by_class):
        cls_items = list(by_class[cls_id])
        rng.shuffle(cls_items)
        split = int((1.0 - val_fraction) * len(cls_items))
        train_data.extend(cls_items[:split])
        val_data.extend(cls_items[split:])

    rng.shuffle(train_data)
    rng.shuffle(val_data)
    return train_data, val_data

def save_split_manifest(path, root_dir, train_items, val_items, seed, val_fraction):
    def serialize(items):
        return [
            {
                "path": os.path.relpath(item_path, root_dir),
                "label": label,
            }
            for item_path, label in items
        ]

    payload = {
        "root_dir": root_dir,
        "seed": seed,
        "val_fraction": val_fraction,
        "train_count": len(train_items),
        "val_count": len(val_items),
        "train": serialize(train_items),
        "val": serialize(val_items),
    }
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

def load_split_manifest(path, root_dir):
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)

    def deserialize(rows):
        return [
            (os.path.join(root_dir, row["path"]), int(row["label"]))
            for row in rows
        ]

    return deserialize(payload["train"]), deserialize(payload["val"])


def resolve_split_manifest_path(cfg):
    candidates = []
    if cfg.SPLIT_MANIFEST_PATH:
        candidates.append(cfg.SPLIT_MANIFEST_PATH)
    candidates.extend(getattr(cfg, "DEFAULT_SPLIT_MANIFEST_CANDIDATES", []))
    for candidate in candidates:
        if candidate and os.path.exists(candidate):
            return candidate
    return None


def build_weighted_sampler(labels, num_classes, seed, power):
    label_counts = Counter(int(label) for label in labels)
    total_samples = sum(label_counts.values())
    class_weights = {
        cls_id: total_samples / (num_classes * label_counts.get(cls_id, 1))
        for cls_id in range(num_classes)
    }
    sample_weights = [float(class_weights[int(label)] ** power) for label in labels]
    generator = torch.Generator()
    generator.manual_seed(seed)
    sampler = WeightedRandomSampler(
        weights=torch.as_tensor(sample_weights, dtype=torch.double),
        num_samples=len(sample_weights),
        replacement=True,
        generator=generator,
    )
    return sampler, class_weights


def file_sha256(path):
    import hashlib

    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def cfg_to_dict(cfg):
    payload = {}
    for key, value in cfg.__dict__.items():
        if key.startswith("_") or callable(value):
            continue
        if isinstance(value, (str, int, float, bool)) or value is None:
            payload[key] = value
        elif isinstance(value, (list, tuple)):
            payload[key] = list(value)
    return payload


def build_run_metadata(cfg, manifest_path, train_items, val_items, label_counts):
    metadata = {
        "config": cfg_to_dict(cfg),
        "train_count": len(train_items),
        "val_count": len(val_items),
        "train_label_counts": {str(key): int(value) for key, value in sorted(label_counts.items())},
        "package_versions": {
            "torch": torch.__version__,
            "timm": getattr(timm, "__version__", "unknown"),
            "albumentations": getattr(A, "__version__", "unknown"),
            "numpy": np.__version__,
        },
    }
    if manifest_path and os.path.exists(manifest_path):
        with open(manifest_path, "r", encoding="utf-8") as handle:
            manifest_payload = json.load(handle)
        metadata["split_manifest_path"] = manifest_path
        metadata["split_manifest_sha256"] = file_sha256(manifest_path)
        metadata["split_method"] = manifest_payload.get("method", "manifest")
    else:
        metadata["split_method"] = "stratified_random"
    return metadata

# ─── Dataset ────────────────────────────────────────────────────────────────
class ClassificationDataset(Dataset):
    def __init__(self, items, transform=None, split_name="train"):
        self.transform = transform
        print(f"  📥 Loading {len(items)} images into RAM ({split_name})...")
        # Pre-allocate to prevent RAM doubling
        self.images = np.empty((len(items), CFG.IMG_SIZE, CFG.IMG_SIZE, 3), dtype=np.uint8)
        self.labels = np.empty(len(items), dtype=np.int64)
        
        for i, (path, label) in enumerate(tqdm(items, leave=False)):
            img = cv2.imread(path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = robust_resize(img, CFG.IMG_SIZE)
            self.images[i] = img
            self.labels[i] = label
        
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
            A.CLAHE(clip_limit=4.0, p=1.0),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05, p=1.0),
        ], p=0.4),
        A.OneOf([
            A.GaussNoise(p=1.0),
            A.GaussianBlur(blur_limit=(3, 7), p=1.0),
        ], p=0.3),
        A.ElasticTransform(alpha=50, sigma=5, p=0.15),
        A.CoarseDropout(num_holes_range=(1, 4), hole_height_range=(8, 16), hole_width_range=(8, 16), p=0.2), # Removed fill_value
        ToTensorV2(),
    ])

def get_val_transforms(sz):
    return A.Compose([
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
        # Augmentation is handled by Albumentations only (no double-flip!)
        # Normalize on GPU
        imgs = (imgs - self.mean) / self.std
        return imgs

# ─── Focal Loss ─────────────────────────────────────────────────────────────
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, label_smoothing=0.1, class_weights=None):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        self.class_weights = class_weights

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(
            inputs, targets, reduction='none',
            label_smoothing=self.label_smoothing,
            weight=self.class_weights,
        )
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt)**self.gamma * ce_loss
        return focal_loss.mean()

# ─── Mixup ───────────────────────────────────────────────────────────────────
def mixup_data(x, y, alpha=0.2):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0
    index = torch.randperm(x.size(0), device=x.device)
    mixed_x = lam * x + (1 - lam) * x[index]
    return mixed_x, y, y[index], lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

# ─── CutMix (SOTA Kaggle Trick) ─────────────────────────────────────────────
def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1.0 - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)
    cx = np.random.randint(W)
    cy = np.random.randint(H)
    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    return bbx1, bby1, bbx2, bby2

def cutmix_data(x, y, alpha=1.0):
    lam = np.random.beta(alpha, alpha)
    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)
    bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
    x[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size()[-1] * x.size()[-2]))
    y_a, y_b = y, y[index]
    return x, y_a, y_b, lam

# ─── Training Loop ──────────────────────────────────────────────────────────
def train_one_epoch(model, loader, criterion, optimizer, scheduler, scaler, device, preprocessor, cfg):
    model.train()
    running_loss, correct, n = 0.0, 0, 0
    amp_enabled = str(device).startswith("cuda")
    amp_device = "cuda" if amp_enabled else "cpu"
    optimizer.zero_grad(set_to_none=True)
    pbar = tqdm(loader, desc="  Train", leave=False)
    for i, (imgs, labels) in enumerate(pbar):
        imgs = preprocessor(imgs, is_train=True)
        labels = labels.to(device, non_blocking=True)
        
        # CutMix or Mixup (randomly chosen)
        use_cutmix = hasattr(cfg, 'CUTMIX_ALPHA') and np.random.rand() < cfg.CUTMIX_PROB
        if use_cutmix:
            imgs, labels_a, labels_b, lam = cutmix_data(imgs, labels, cfg.CUTMIX_ALPHA)
        elif cfg.MIXUP_ALPHA > 0:
            imgs, labels_a, labels_b, lam = mixup_data(imgs, labels, cfg.MIXUP_ALPHA)
        else:
            labels_a, labels_b, lam = labels, labels, 1.0
        
        with torch.amp.autocast(amp_device, enabled=amp_enabled):
            outputs = model(imgs)
            loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam) / cfg.GRAD_ACCUM_STEPS
        
        scaler.scale(loss).backward()
        
        if (i + 1) % cfg.GRAD_ACCUM_STEPS == 0 or (i + 1) == len(loader):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg.GRAD_CLIP)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

        running_loss += (loss.item() * cfg.GRAD_ACCUM_STEPS) * imgs.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        n += imgs.size(0)
        pbar.set_postfix(loss=f"{loss.item()*cfg.GRAD_ACCUM_STEPS:.3f}", acc=f"{correct/n:.3f}")
    return running_loss / n, correct / n

@torch.no_grad()
def validate(model, loader, criterion, device, preprocessor):
    model.eval()
    running_loss, correct, n = 0.0, 0, 0
    amp_enabled = str(device).startswith("cuda")
    amp_device = "cuda" if amp_enabled else "cpu"
    for imgs, labels in tqdm(loader, desc="  Valid", leave=False):
        imgs = preprocessor(imgs, is_train=False)
        labels = labels.to(device, non_blocking=True)
        
        # Match submission-time TTA so validation is leaderboard-adjacent.
        with torch.amp.autocast(amp_device, enabled=amp_enabled):
            out1 = model(imgs)
            out2 = model(torch.flip(imgs, dims=[3]))
            out3 = model(torch.flip(imgs, dims=[2]))
            out4 = model(torch.rot90(imgs, 1, dims=[2, 3]))
            outputs = (out1 + out2 + out3 + out4) / 4.0
            loss = criterion(outputs, labels)
        running_loss += loss.item() * imgs.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        n += imgs.size(0)
    return running_loss / n, correct / n

# ─── MAIN ────────────────────────────────────────────────────────────────────
def main():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    seed_everything(CFG.SEED)
    os.makedirs(CFG.MODEL_SAVE_DIR, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  🚀 V5 Kaggle Classification (Optimized)")
    print(f"  ⚡️ EfficientNet-B2 + Robust Padding + TTA")
    print(f"{'='*60}\n")

    # Clear memory from previous runs
    gc.collect()
    torch.cuda.empty_cache()

    manifest_path = resolve_split_manifest_path(CFG)
    if manifest_path:
        train_items, val_items = load_split_manifest(manifest_path, CFG.TRAIN_DIR)
        print(f"  🧾 Using split manifest: {manifest_path}")
    else:
        train_items, val_items = build_classification_split(CFG.TRAIN_DIR, CFG.VAL_FRACTION, CFG.SPLIT_SEED)
        manifest_path = os.path.join(CFG.MODEL_SAVE_DIR, "classification_split_manifest.json")
        save_split_manifest(manifest_path, CFG.TRAIN_DIR, train_items, val_items, CFG.SPLIT_SEED, CFG.VAL_FRACTION)
        print(f"  🧾 Split manifest: {manifest_path}")

    train_ds = ClassificationDataset(train_items, get_train_transforms(CFG.IMG_SIZE), "train")
    val_ds = ClassificationDataset(val_items, get_val_transforms(CFG.IMG_SIZE), "val")
    
    sampler = None
    if CFG.USE_WEIGHTED_SAMPLER:
        sampler, sampler_class_weights = build_weighted_sampler(train_ds.labels.tolist(), CFG.NUM_CLASSES, CFG.SEED, CFG.SAMPLER_POWER)
        print(
            "  ⚖️ Weighted sampler enabled: "
            + ", ".join(f"{cls}:{sampler_class_weights[cls]:.2f}" for cls in range(CFG.NUM_CLASSES))
        )
    train_loader = DataLoader(
        train_ds,
        batch_size=CFG.BATCH_SIZE,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=0,
        pin_memory=True,
    )
    val_loader = DataLoader(val_ds, batch_size=CFG.BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

    model = timm.create_model(CFG.MODEL_NAME, pretrained=True, num_classes=CFG.NUM_CLASSES, drop_rate=CFG.DROP_RATE).to(CFG.DEVICE)
    
    print(f"  ⚡️ Using Single GPU: {CFG.DEVICE.upper()}")
    
    # model = torch.compile(model) # DISABLED - Causes stalling on Kaggle

    # Compute class weights from training data (inverse frequency)
    label_counts = Counter(train_ds.labels.tolist())
    total_samples = sum(label_counts.values())
    class_weights = torch.tensor(
        [total_samples / (CFG.NUM_CLASSES * label_counts.get(i, 1)) for i in range(CFG.NUM_CLASSES)],
        dtype=torch.float32
    ).to(CFG.DEVICE)
    print(f"  📊 Class weights: {', '.join(f'{w:.2f}' for w in class_weights.tolist())}")
    run_metadata = build_run_metadata(CFG, manifest_path, train_items, val_items, label_counts)

    criterion = FocalLoss(label_smoothing=CFG.LABEL_SMOOTHING, class_weights=class_weights)
    optimizer = optim.AdamW(model.parameters(), lr=CFG.MAX_LR, weight_decay=CFG.WEIGHT_DECAY)
    scaler = torch.amp.GradScaler('cuda', enabled=str(CFG.DEVICE).startswith("cuda"))
    
    steps_per_epoch = math.ceil(len(train_loader) / CFG.GRAD_ACCUM_STEPS)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=CFG.MAX_LR, epochs=CFG.EPOCHS, steps_per_epoch=steps_per_epoch)

    preprocessor = GPUPreprocessor(CFG.DEVICE)

    best_acc = 0.0
    no_improve_count = 0
    start_time = time.time()

    for epoch in range(1, CFG.EPOCHS + 1):
        t0 = time.time()
        print(f"\nEpoch {epoch}/{CFG.EPOCHS}")

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, scheduler, scaler, CFG.DEVICE, preprocessor, CFG)
        val_loss, val_acc = validate(model, val_loader, criterion, CFG.DEVICE, preprocessor)

        print(f"  Train Loss: {train_loss:.4f} | Acc: {train_acc:.4f}")
        print(f"  Val   Loss: {val_loss:.4f} | Acc: {val_acc:.4f} | ⏱️ {time.time()-t0:.0f}s")

        gap = train_acc - val_acc
        if gap > 0.15:
            print(f"  ⚠️ Overfitting alert! Train-Val gap: {gap:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            no_improve_count = 0
            state_dict = model.module.state_dict() if hasattr(model, "module") else model.state_dict()
            checkpoint = {
                "model_state_dict": state_dict,
                "model_name": CFG.MODEL_NAME,
                "num_classes": CFG.NUM_CLASSES,
                "img_size": CFG.IMG_SIZE,
                "val_acc": best_acc,
                "epoch": epoch,
                "split_seed": CFG.SPLIT_SEED,
                "val_fraction": CFG.VAL_FRACTION,
                "split_manifest_path": manifest_path,
                "run_metadata": run_metadata,
            }
            torch.save(checkpoint, os.path.join(CFG.MODEL_SAVE_DIR, "best_model.pth"))
            print(f"  ✅ Best saved! Accuracy: {best_acc:.4f}")
        else:
            no_improve_count += 1
            print(f"  ⏸️ No improvement ({no_improve_count}/{CFG.EARLY_STOP_PATIENCE})")
            if no_improve_count >= CFG.EARLY_STOP_PATIENCE:
                print(f"  🛑 Early stopping at epoch {epoch}!")
                break

        gc.collect()
        torch.cuda.empty_cache()

    print(f"\n{'='*60}\n  ✅ Training Done! Best Acc: {best_acc:.4f} | Total ⏱️ {(time.time()-start_time)/60:.1f} min\n{'='*60}")

if __name__ == "__main__":
    main()
