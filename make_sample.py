# make_sample.py
import os
import argparse
import random
import shutil
from pathlib import Path
import zipfile

def copy_sample(src, dst, per_class):
    src = Path(src)
    dst = Path(dst)
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True, exist_ok=True)

    splits = ['Train','Val','Test']
    exts = ('.jpg','.jpeg','.png','.bmp','.tif','.tiff')

    for split in splits:
        split_src = src / split
        if not split_src.exists():
            print(f"Skipping missing split: {split_src}")
            continue
        for cls in sorted([d for d in split_src.iterdir() if d.is_dir()]):
            files = [f for f in cls.iterdir() if f.suffix.lower() in exts]
            if not files:
                continue
            dst_cls_dir = dst / split / cls.name
            dst_cls_dir.mkdir(parents=True, exist_ok=True)
            k = min(per_class, len(files))
            sampled = random.sample(files, k)
            for f in sampled:
                shutil.copy2(f, dst_cls_dir / f.name)
            print(f"Copied {len(sampled):4d} files for {split}/{cls.name}")

def zip_folder(folder, out_zip):
    folder = Path(folder)
    with zipfile.ZipFile(out_zip, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
        for root, _, files in os.walk(folder):
            for f in files:
                full = Path(root) / f
                rel = full.relative_to(folder.parent)
                zf.write(full, arcname=rel)
    print(f"Created zip: {out_zip}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--src', required=True, help='source dataset root (contains Train/Val/Test)')
    parser.add_argument('--dst', required=True, help='destination sample folder to create')
    parser.add_argument('--per_class', type=int, default=500, help='max images per class per split')
    args = parser.parse_args()

    copy_sample(args.src, args.dst, args.per_class)
    zip_path = f"{args.dst}.zip"
    zip_folder(args.dst, zip_path)
