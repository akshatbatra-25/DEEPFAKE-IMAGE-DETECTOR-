# infer.py
"""
Inference script.
Example:
  python infer.py --checkpoint outputs/best_checkpoint.pth --image Dataset/Test/Fake/0001.jpg --topk 2
  python infer.py --checkpoint outputs/best_checkpoint.pth --image Dataset/Test --topk 1
"""
import os
import argparse
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms, models

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', type=str, default='outputs/best_checkpoint.pth')
    p.add_argument('--image', type=str, required=True, help='image path or directory')
    p.add_argument('--img-size', type=int, default=224)
    p.add_argument('--topk', type=int, default=3)
    p.add_argument('--class-map', type=str, default=None, help='optional path to idx_to_class.json')
    return p.parse_args()

def make_transform(img_size):
    return transforms.Compose([
        transforms.Resize(int(img_size * 1.14)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])

def load_checkpoint(path, device):
    ckpt = torch.load(path, map_location=device)
    raw_map = ckpt.get('idx_to_class', None)
    # Normalize to int keys -> class name values
    if isinstance(raw_map, dict):
        try:
            idx_to_class = {int(k): v for k, v in raw_map.items()}
        except Exception:
            # fallback if keys already ints
            idx_to_class = {int(k): v for k, v in enumerate(raw_map.values())}
    else:
        idx_to_class = {}

    # build model architecture (matches training: resnet50 with custom head)
    model = models.resnet50(pretrained=False)
    in_f = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(in_f, 512),
        nn.ReLU(inplace=True),
        nn.Dropout(0.4),
        nn.Linear(512, len(idx_to_class) if len(idx_to_class)>0 else 2)
    )
    model.load_state_dict(ckpt['model_state'])
    model.to(device)
    model.eval()
    return model, idx_to_class

def predict_image(model, img_path, transform, device, idx_to_class, topk=3):
    img = Image.open(img_path).convert('RGB')
    x = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0]
        topv, topi = probs.topk(topk)
        results = []
        for p, i in zip(topv, topi):
            idx = int(i.item())
            label = idx_to_class.get(idx, str(idx))
            results.append((label, float(p.item())))
    return results

def is_image_file(f):
    return f.lower().endswith(('.jpg','.jpeg','.png','.bmp','.tif','.tiff'))

def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model, idx_to_class = load_checkpoint(args.checkpoint, device)

    # optionally override class map from a json file
    if args.class_map and os.path.exists(args.class_map):
        import json
        with open(args.class_map, 'r') as fh:
            raw = json.load(fh)
        try:
            idx_to_class = {int(k): v for k, v in raw.items()}
        except:
            # if keys are ints already in json, keep them
            idx_to_class = raw

    transform = make_transform(args.img_size)

    targets = []
    if os.path.isdir(args.image):
        for fn in sorted(os.listdir(args.image)):
            if is_image_file(fn):
                targets.append(os.path.join(args.image, fn))
    else:
        targets = [args.image]

    for t in targets:
        if not os.path.exists(t):
            print(f"Missing file: {t}")
            continue
        try:
            preds = predict_image(model, t, transform, device, idx_to_class, topk=args.topk)
            print(f"\n{t}:")
            for label, p in preds:
                print(f"  {label:15s} {p:.4f}")
        except Exception as e:
            print(f"Failed {t}: {e}")

if __name__ == '__main__':
    main()
