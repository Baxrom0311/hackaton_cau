#!/usr/bin/env python3
"""
Guide-Compliant Submission Assembler
Builds the exact folder structure from guide.txt Section 7.
"""
import os, shutil, sys
from zipfile import ZipFile

BASE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(BASE, "OxDEAD_Submission")
DEST = os.path.join(BASE, "OxDEAD_Final")
TEAM = "OxDEAD"

# Clean start
if os.path.exists(DEST):
    shutil.rmtree(DEST)

# Create guide-compliant structure
masks_dir = os.path.join(DEST, TEAM)
cls_dir = os.path.join(DEST, "models", "classification")
seg_dir = os.path.join(DEST, "models", "segmentation")
os.makedirs(masks_dir, exist_ok=True)
os.makedirs(cls_dir, exist_ok=True)
os.makedirs(seg_dir, exist_ok=True)

# 1. Excel
src_excel = os.path.join(SRC, f"{TEAM} test_ground_truth.xlsx")
dst_excel = os.path.join(DEST, f"{TEAM} test_ground_truth.xlsx")
shutil.copy2(src_excel, dst_excel)
print(f"✅ 1. Excel: {os.path.basename(dst_excel)}")

# 2. Masks (unzip into OxDEAD/ folder)
src_zip = os.path.join(SRC, f"{TEAM} masks.zip")
with ZipFile(src_zip, 'r') as z:
    z.extractall(masks_dir)
mask_count = len([f for f in os.listdir(masks_dir) if f.endswith('.png')])
print(f"✅ 2. Masks: {TEAM}/ ({mask_count} ta PNG)")

# 3. Classification script (guide says: classify.py)
shutil.copy2(os.path.join(SRC, f"{TEAM}Class.py"), os.path.join(cls_dir, "classify.py"))
print(f"✅ 3. Script: models/classification/classify.py")

# 4. Classification model
shutil.copy2(os.path.join(SRC, f"{TEAM}ClassModel.pth"), os.path.join(cls_dir, f"{TEAM}ClassModel.pth"))
print(f"✅ 4. Model: models/classification/{TEAM}ClassModel.pth")

# 5. Segmentation script (guide says: segment.py)
shutil.copy2(os.path.join(SRC, f"{TEAM}Seg.py"), os.path.join(seg_dir, "segment.py"))
print(f"✅ 5. Script: models/segmentation/segment.py")

# 6. Segmentation model
shutil.copy2(os.path.join(SRC, f"{TEAM}SegModel.pth"), os.path.join(seg_dir, f"{TEAM}SegModel.pth"))
print(f"✅ 6. Model: models/segmentation/{TEAM}SegModel.pth")

# 7. Requirements.txt files
with open(os.path.join(cls_dir, "requirements.txt"), "w") as f:
    f.write("torch>=2.0\ntimm>=0.9\nalbumentations>=1.3\npandas>=2.0\nopenpyxl>=3.1\nopencv-python>=4.8\nnumpy>=1.24\nPillow>=10.0\ntqdm>=4.65\n")
with open(os.path.join(seg_dir, "requirements.txt"), "w") as f:
    f.write("torch>=2.0\nsegmentation-models-pytorch>=0.3\nalbumentations>=1.3\nopencv-python>=4.8\nnumpy>=1.24\nPillow>=10.0\ntqdm>=4.65\n")
print(f"✅ 7. requirements.txt (classification + segmentation)")

# Print final structure
print(f"\n{'='*50}")
print(f"📁 YAKUNIY STRUKTURA ({DEST}):")
print(f"{'='*50}")
for root, dirs, files in os.walk(DEST):
    level = root.replace(DEST, '').count(os.sep)
    indent = '  ' * level
    print(f"{indent}{os.path.basename(root)}/")
    subindent = '  ' * (level + 1)
    for file in sorted(files):
        size = os.path.getsize(os.path.join(root, file))
        if size > 1024*1024:
            print(f"{subindent}{file} ({size/1024/1024:.1f} MB)")
        else:
            print(f"{subindent}{file} ({size/1024:.1f} KB)")
print(f"{'='*50}")
print(f"🏆 GUIDE BO'YICHA TO'LIQ MOS SUBMISSION TAYYOR!")
