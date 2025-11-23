# train.py  (AMP + DataLoader tuning + cudnn benchmark)
import os
import argparse
import time
import copy
import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms, datasets, models

# AMP imports
from torch.cuda.amp import GradScaler, autocast

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data-dir', type=str, default='Dataset', help='root dataset folder with Train/Val/Test')
    p.add_argument('--model-out', type=str, default='outputs/best_checkpoint.pth')
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--epochs', type=int, default=10)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--img-size', type=int, default=224)
    p.add_argument('--workers', '--num-workers', dest='workers', type=int, default=8)
    p.add_argument('--persistent-workers', action='store_true', help='use persistent DataLoader workers')
    p.add_argument('--pin-memory', action='store_true', help='pin memory in DataLoader')
    p.add_argument('--freeze-backbone', action='store_true', help='freeze pretrained backbone initially')
    p.add_argument('--use-compile', action='store_true', help='try torch.compile() if available (optional)')
    return p.parse_args()

def make_transforms(img_size):
    train_t = transforms.Compose([
        transforms.RandomResizedCrop(img_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])
    val_t = transforms.Compose([
        transforms.Resize(int(img_size * 1.14)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])
    return train_t, val_t

def prepare_dataloaders(data_dir, batch_size, img_size, workers, persistent, pin_memory):
    train_t, val_t = make_transforms(img_size)
    train_dir = os.path.join(data_dir, 'Train')
    val_dir   = os.path.join(data_dir, 'Val')

    train_ds = datasets.ImageFolder(train_dir, transform=train_t)
    val_ds   = datasets.ImageFolder(val_dir, transform=val_t)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=workers, pin_memory=pin_memory,
                              persistent_workers=persistent)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=workers, pin_memory=pin_memory,
                              persistent_workers=persistent)

    class_to_idx = train_ds.class_to_idx
    idx_to_class = {int(v): k for k, v in class_to_idx.items()}
    return train_loader, val_loader, idx_to_class

def build_model(num_classes, freeze_backbone=False):
    model = models.resnet50(pretrained=True)
    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False
    in_f = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(in_f, 512),
        nn.ReLU(inplace=True),
        nn.Dropout(0.4),
        nn.Linear(512, num_classes)
    )
    return model

def train_one_epoch(model, loader, criterion, optimizer, device, scaler):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for imgs, labels in loader:
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad()
        with autocast():
            outputs = model(imgs)
            loss = criterion(outputs, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * imgs.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += imgs.size(0)
    return running_loss / total, correct / total

def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            with autocast():
                outputs = model(imgs)
                loss = criterion(outputs, labels)
            running_loss += loss.item() * imgs.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += imgs.size(0)
    return running_loss / total, correct / total

def save_checkpoint(path, state):
    torch.save(state, path)

def main():
    args = parse_args()
    Path(os.path.dirname(args.model_out)).mkdir(parents=True, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device:', device)

    # speed options
    torch.backends.cudnn.benchmark = True

    train_loader, val_loader, idx_to_class = prepare_dataloaders(
        args.data_dir, args.batch_size, args.img_size, args.workers,
        args.persistent_workers, args.pin_memory
    )
    num_classes = len(idx_to_class)
    print(f'Classes ({num_classes}):', idx_to_class)

    model = build_model(num_classes, freeze_backbone=args.freeze_backbone)
    model = model.to(device)

    # Optional: try torch.compile for PyTorch 2.x (may help)
    if args.use_compile and hasattr(torch, 'compile'):
        try:
            model = torch.compile(model)
            print("Model compiled with torch.compile()")
        except Exception as e:
            print("torch.compile failed:", e)

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.3, patience=3)
    criterion = nn.CrossEntropyLoss()

    scaler = GradScaler()
    best_acc = 0.0
    best_wts = copy.deepcopy(model.state_dict())
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        t_loss, t_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, scaler)
        v_loss, v_acc = validate(model, val_loader, criterion, device)
        scheduler.step(v_loss)

        history['train_loss'].append(t_loss); history['train_acc'].append(t_acc)
        history['val_loss'].append(v_loss); history['val_acc'].append(v_acc)

        print(f"Epoch {epoch}/{args.epochs} "
              f"| train_loss: {t_loss:.4f} train_acc: {t_acc:.4f} "
              f"| val_loss: {v_loss:.4f} val_acc: {v_acc:.4f}")

        if v_acc > best_acc:
            best_acc = v_acc
            best_wts = copy.deepcopy(model.state_dict())
            ckpt = {
                'epoch': epoch,
                'model_state': best_wts,
                'optimizer_state': optimizer.state_dict(),
                'idx_to_class': {str(k): v for k, v in idx_to_class.items()},
                'history': history
            }
            save_checkpoint(args.model_out, ckpt)
            print(f"Saved best checkpoint ({args.model_out}) val_acc={best_acc:.4f}")

    with open(os.path.join(os.path.dirname(args.model_out), 'idx_to_class.json'), 'w') as f:
        json.dump({str(k): v for k, v in idx_to_class.items()}, f, indent=2)
    with open(os.path.join(os.path.dirname(args.model_out), 'history.json'), 'w') as f:
        json.dump(history, f, indent=2)

    elapsed = time.time() - start_time
    print(f"Done. Best val_acc: {best_acc:.4f}. Time: {int(elapsed//60)}m {int(elapsed%60)}s")

if __name__ == '__main__':
    main()
