import os
from PIL import Image
from collections import Counter

# 🛠️ Portable Path Detection (Kaggle or Mac)
potential_roots = [
    "classification",                                    # Mac Local
    "Segmentation",                                      # Mac Local
    "/kaggle/input/datasets/baxrom0311/main-dataset/Main hackathon dataset" # Kaggle
]

def scan_dir(root):
    sizes = Counter()
    print(f"🔍 Skaner qilinmoqda: {root} ...")
    for r, d, files in os.walk(root):
        for f in files:
            if f.lower().endswith(('.jpg', '.png', '.jpeg')):
                try:
                    with Image.open(os.path.join(r, f)) as img:
                        sizes[img.size] += 1
                except:
                    pass
    return sizes

for root in potential_roots:
    if os.path.exists(root):
        print(f"\n📂 Topildi: {root}")
        results = scan_dir(root)
        if results:
            print(f"📊 O'lchamlar (W, H) va soni:")
            for size, count in results.items():
                print(f"   - {size}: {count} ta rasm")
        else:
            print("❗ Rasm topilmadi.")

print("\n✅ Tekshirish tugadi!")
