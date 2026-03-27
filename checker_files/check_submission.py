#!/usr/bin/env python3
"""
Google Form Submission — Yakuniy Tekshiruvchi
OxDEAD jamoasi uchun 6 ta faylni chuqur tekshiradi.
"""
import torch, os, sys
import pandas as pd
import numpy as np
from zipfile import ZipFile
from PIL import Image

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SUB = os.path.join(BASE, 'OxDEAD_Submission')
TEAM = 'OxDEAD'

errors = []

print('='*60)
print('  🔍 SUBMISSION TEKSHIRUVI (6 ta fayl)')
print('='*60)

# ═══════════════════════════════════════
# 1. EXCEL FILE
# ═══════════════════════════════════════
excel_name = f'{TEAM} test_ground_truth.xlsx'
excel_path = os.path.join(SUB, excel_name)
print(f'\n📊 1. Excel: {excel_name}')
if os.path.exists(excel_path):
    df = pd.read_excel(excel_path)
    print(f'   ✅ Fayl mavjud ({os.path.getsize(excel_path)/1024:.1f} KB)')
    print(f'   Ustunlar: {list(df.columns)}')
    print(f'   Qatorlar: {len(df)}')
    if set(df.columns) == {'Image_ID', 'Label'}:
        print(f'   ✅ Ustunlar to\'g\'ri (Image_ID, Label)')
    else:
        errors.append(f'Excel ustunlari noto\'g\'ri: {list(df.columns)}')
    if len(df) == 1276:
        print(f'   ✅ 1276 ta qator (to\'g\'ri)')
    else:
        errors.append(f'Excel da {len(df)} qator bor, 1276 bo\'lishi kerak!')
    dups = df['Image_ID'].duplicated().sum()
    if dups == 0:
        print(f'   ✅ Dublikatlar yo\'q')
    else:
        errors.append(f'{dups} ta takroriy Image_ID!')
    valid_labels = set(range(12))
    actual_labels = set(int(x) for x in df['Label'].unique())
    invalid = actual_labels - valid_labels
    if not invalid:
        print(f'   ✅ Labellar 0-11 orasida ({len(actual_labels)} xil sinf)')
    else:
        errors.append(f'Noto\'g\'ri labellar: {invalid}')
    missing = df['Image_ID'].isna().sum()
    if missing > 0:
        errors.append(f'{missing} ta bo\'sh Image_ID!')
    else:
        print(f'   ✅ Bo\'sh Image_ID yo\'q')
else:
    errors.append(f'Excel topilmadi: {excel_name}')

# ═══════════════════════════════════════
# 2. MASKS ZIP
# ═══════════════════════════════════════
zip_name = f'{TEAM} masks.zip'
zip_path = os.path.join(SUB, zip_name)
print(f'\n🗂️  2. Masks ZIP: {zip_name}')
if os.path.exists(zip_path):
    print(f'   ✅ Fayl mavjud ({os.path.getsize(zip_path)/1024:.1f} KB)')
    with ZipFile(zip_path, 'r') as z:
        files = z.namelist()
        png_files = [f for f in files if f.lower().endswith('.png')]
        print(f'   ZIP ichida: {len(files)} fayl (PNG: {len(png_files)})')
        if len(png_files) == 200:
            print(f'   ✅ 200 ta PNG mask (to\'g\'ri)')
        else:
            errors.append(f'ZIP da {len(png_files)} ta PNG, 200 bo\'lishi kerak!')
        binary_ok = True
        for pf in png_files[:5]:
            with z.open(pf) as mf:
                mask = np.array(Image.open(mf))
                unique = set(np.unique(mask).tolist())
                if not unique.issubset({0, 255}):
                    errors.append(f'Mask {pf} binary emas: {unique}')
                    binary_ok = False
        if binary_ok:
            print(f'   ✅ Maskalar binary (0/255)')
else:
    errors.append(f'Masks ZIP topilmadi: {zip_name}')

# ═══════════════════════════════════════
# 3-4. PYTHON SCRIPTS
# ═══════════════════════════════════════
for idx, (sn, req) in enumerate([(f'{TEAM}Class.py','timm'),(f'{TEAM}Seg.py','smp')], start=3):
    sp = os.path.join(SUB, sn)
    print(f'\n🐍 {idx}. Script: {sn}')
    if os.path.exists(sp):
        print(f'   ✅ Fayl mavjud ({os.path.getsize(sp)/1024:.1f} KB)')
        with open(sp) as f:
            c = f.read()
        if req in c: print(f'   ✅ "{req}" kutubxonasi bor')
        if 'argparse' in c: print(f'   ✅ argparse (CLI) mavjud')
    else:
        errors.append(f'Script topilmadi: {sn}')

# ═══════════════════════════════════════
# 5-6. MODEL FILES
# ═══════════════════════════════════════
for idx, mn in enumerate([f'{TEAM}ClassModel.pth', f'{TEAM}SegModel.pth'], start=5):
    mp2 = os.path.join(SUB, mn)
    print(f'\n🧠 {idx}. Model: {mn}')
    if os.path.exists(mp2):
        size_mb = os.path.getsize(mp2) / (1024*1024)
        print(f'   ✅ Fayl mavjud ({size_mb:.1f} MB)')
        ckpt = torch.load(mp2, map_location='cpu', weights_only=False)
        if 'model_state_dict' in ckpt:
            print(f'   ✅ model_state_dict ({len(ckpt["model_state_dict"])} qatlam)')
        for key in ['encoder','img_size','val_iou','epoch','val_acc','best_threshold','model_name']:
            if key in ckpt:
                val = ckpt[key]
                if isinstance(val, float) and val < 5:
                    print(f'   📎 {key}: {val*100:.2f}%')
                else:
                    print(f'   📎 {key}: {val}')
    else:
        errors.append(f'Model topilmadi: {mn}')

# ═══════════════════════════════════════
# XULOSA
# ═══════════════════════════════════════
print(f'\n{"="*60}')
if not errors:
    print('🏆 BARCHA TEKSHIRUVLAR O\'TDI! Submission 100% tayyor! 🎉')
else:
    print(f'❌ {len(errors)} TA XATOLIK TOPILDI:')
    for e in errors:
        print(f'   ❌ {e}')
print('='*60)
