# dataset_check.py
import os
from collections import Counter

DATA_ROOT = "Dataset"  # change if needed

def count_images(folder):
    exts = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')
    c = 0
    for root, _, files in os.walk(folder):
        for f in files:
            if f.lower().endswith(exts):
                c += 1
    return c

def summarize_split(split):
    split_path = os.path.join(DATA_ROOT, split)
    if not os.path.isdir(split_path):
        print(f"Missing folder: {split_path}")
        return
    classes = [d for d in os.listdir(split_path) if os.path.isdir(os.path.join(split_path, d))]
    classes.sort()
    print(f"\n=== {split} ===")
    total = 0
    for cls in classes:
        cls_path = os.path.join(split_path, cls)
        cnt = count_images(cls_path)
        print(f"  {cls:12s} : {cnt}")
        total += cnt
    print(f"  Total images in {split}: {total}")

if __name__ == "__main__":
    for s in ('Train','Val','Test'):
        summarize_split(s)
