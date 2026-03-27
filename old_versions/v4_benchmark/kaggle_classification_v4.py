# !pip install segmentation-models-pytorch albumentations timm openpyxl -q

# ============================================================
# AI Healthcare Hackathon 2026 — V4 Kaggle Classification
# ============================================================
# 🚨 BENCHMARK VERSION (B5 @ 512):
# ✅ EfficientNet-B5 + 512x512
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

# ─── Configuration ───────────────────────────────────────────────────────────
class CFG:
    BASE = "/kaggle/input/datasets/baxrom0311/main-dataset/Main hackathon dataset"
    TRAIN_DIR = f"{BASE}/classification/train"
    MODEL_SAVE_DIR = "classification_v4"

    MODEL_NAME = "efficientnet_b5"
    IMG_SIZE = 512
    BATCH_SIZE = 4                    # Kaggle T4 da 8 sig'masligi mumkin
    EPOCHS = 35
    MAX_LR = 1e-3
    WEIGHT_DECAY = 1e-4
    NUM_CLASSES = 12
    NUM_WORKERS = 2
    SEED = 42
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    GRAD_ACCUM_STEPS = 4

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
        self.data = []
        for label in os.listdir(root_dir):
            if label.isdigit():
                cls_dir = os.path.join(root_dir, label)
                for img_name in os.listdir(cls_dir):
                    if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                        self.data.append((os.path.join(cls_dir, img_name), int(label)))
        
        # Simple Split (90% Train, 10% Val)
        random.shuffle(self.data)
        split = int(0.9 * len(self.data))
        self.data = self.data[:split] if is_train else self.data[split:]

    def __len__(self): return len(self.data)

    def __getitem__(self, idx):
        path, label = self.data[idx]
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.transform:
            img = self.transform(image=img)["image"]
        return img, label

# ─── Transforms ──────────────────────────────────────────────────────────────
def get_train_transforms(sz):
    return A.Compose([
        A.Resize(sz, sz),
        A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.5), A.RandomRotate90(p=0.5),
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
def train_one_epoch(model, loader, criterion, optimizer, scheduler, device, cfg):
    model.train()
    running_loss, correct, n = 0.0, 0, 0
    pbar = tqdm(loader, desc="  Train", leave=False)
    for i, (imgs, labels) in enumerate(pbar):
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        
        with torch.amp.autocast('cuda'):
            outputs = model(imgs)
            loss = criterion(outputs, labels) / cfg.GRAD_ACCUM_STEPS
        
        loss.backward()
        if (i+1) % cfg.GRAD_ACCUM_STEPS == 0:
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        running_loss += (loss.item() * cfg.GRAD_ACCUM_STEPS) * imgs.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        n += imgs.size(0)
        pbar.set_postfix(loss=f"{loss.item()*cfg.GRAD_ACCUM_STEPS:.3f}", acc=f"{correct/n:.3f}")
    return running_loss / n, correct / n

@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    running_loss, correct, n = 0.0, 0, 0
    for imgs, labels in tqdm(loader, desc="  Valid", leave=False):
        imgs, labels = imgs.to(device), labels.to(device)
        with torch.amp.autocast('cuda'):
            outputs = model(imgs)
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
    print(f"  🚀 V4 Kaggle Classification")
    print(f"  ⚡️ EfficientNet-B5 + 512px Standard")
    print(f"{'='*60}\n")

    train_ds = ClassificationDataset(CFG.TRAIN_DIR, get_train_transforms(CFG.IMG_SIZE), True)
    val_ds = ClassificationDataset(CFG.TRAIN_DIR, get_val_transforms(CFG.IMG_SIZE), False)
    
    train_loader = DataLoader(train_ds, batch_size=CFG.BATCH_SIZE, shuffle=True, num_workers=CFG.NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=CFG.BATCH_SIZE, shuffle=False, num_workers=CFG.NUM_WORKERS, pin_memory=True)

    model = timm.create_model(CFG.MODEL_NAME, pretrained=True, num_classes=CFG.NUM_CLASSES).to(CFG.DEVICE)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=CFG.MAX_LR, weight_decay=CFG.WEIGHT_DECAY)
    
    steps_per_epoch = math.ceil(len(train_loader) / CFG.GRAD_ACCUM_STEPS)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=CFG.MAX_LR, epochs=CFG.EPOCHS, steps_per_epoch=steps_per_epoch)

    best_acc = 0.0
    start_time = time.time()

    for epoch in range(1, CFG.EPOCHS + 1):
        t0 = time.time()
        print(f"\nEpoch {epoch}/{CFG.EPOCHS}")

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, scheduler, CFG.DEVICE, CFG)
        val_loss, val_acc = validate(model, val_loader, criterion, CFG.DEVICE)

        print(f"  Train Loss: {train_loss:.4f} | Acc: {train_acc:.4f}")
        print(f"  Val   Loss: {val_loss:.4f} | Acc: {val_acc:.4f} | ⏱️ {time.time()-t0:.0f}s")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), os.path.join(CFG.MODEL_SAVE_DIR, "best_model.pth"))
            print(f"  ✅ Best saved! Accuracy: {best_acc:.4f}")

        gc.collect()
        torch.cuda.empty_cache()

    print(f"\n{'='*60}\n  ✅ V4 Training Done! Best Acc: {best_acc:.4f} | Total ⏱️ {(time.time()-start_time)/60:.1f} min\n{'='*60}")

if __name__ == "__main__":
    main()
