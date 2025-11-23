#!/usr/bin/env python3
"""
fine_tune.py

Fine-tuning script for deepfake image classification using ResNet50.
Includes:
- Strong augmentations
- MixUp
- Label smoothing / optional Focal loss
- Head-only training + staged unfreeze
- AMP safe (runs on CPU too)
- Checkpoints + TensorBoard logging
"""

import argparse
import os
import random
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, models, transforms
from torch.utils.tensorboard import SummaryWriter

# AMP support (CPU-safe no-op fallback)
try:
    from torch.cuda.amp import GradScaler, autocast
except Exception:
    autocast = lambda *args, **kwargs: (lambda ctx: (yield))
    class GradScaler:
        def __init__(self): pass
        def scale(self, loss): return loss
        def step(self, optimizer): optimizer.step()
        def update(self): pass
        def unscale_(self, optimizer): pass


# =============== UTILITIES ===============

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, weight=None, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.weight = weight
        self.reduction = reduction
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, target):
        logp = self.log_softmax(input)
        p = logp.exp()
        logp_t = logp.gather(1, target.unsqueeze(1)).squeeze(1)
        p_t = p.gather(1, target.unsqueeze(1)).squeeze(1)
        loss = -((1 - p_t) ** self.gamma) * logp_t
        if self.weight is not None:
            loss = loss * self.weight[target]
        return loss.mean() if self.reduction == 'mean' else loss.sum()


def mixup_data(x, y, alpha=0.2, device='cpu'):
    if alpha <= 0:
        return x, y, None, 1.0
    lam = np.random.beta(alpha, alpha)
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(device)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


# =============== TRANSFORMS & LOADERS ===============

def build_transforms(img_size=224, train=True):
    if train:
        return transforms.Compose([
            transforms.RandomResizedCrop(img_size, scale=(0.6, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.15, hue=0.02),
            transforms.ToTensor(),
            transforms.RandomErasing(p=0.25, scale=(0.02, 0.2), ratio=(0.3, 3.3)),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
        ])
    else:
        return transforms.Compose([
            transforms.Resize(int(img_size * 1.15)),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
        ])


def get_dataloaders(data_dir, batch_size, img_size, num_workers, use_weighted_sampler=False):
    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'val')

    train_ds = datasets.ImageFolder(train_dir, transform=build_transforms(img_size, train=True))
    val_ds = datasets.ImageFolder(val_dir, transform=build_transforms(img_size, train=False))

    if use_weighted_sampler:
        class_counts = Counter([y for _, y in train_ds.samples])
        class_weights = {cls: 1.0 / cnt for cls, cnt in class_counts.items()}
        samples_weight = [class_weights[y] for _, y in train_ds.samples]
        sampler = WeightedRandomSampler(samples_weight, num_samples=len(samples_weight), replacement=True)
        train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler, num_workers=num_workers)
    else:
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader, len(train_ds.classes)


# =============== TRAINING ===============

def train_one_epoch(model, loader, criterion, optimizer, device, scaler, epoch, args, writer=None):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    iters = len(loader)

    for i, (inputs, targets) in enumerate(loader):
        inputs, targets = inputs.to(device), targets.to(device)

        if args.mixup > 0:
            inputs, y_a, y_b, lam = mixup_data(inputs, targets, args.mixup, device)
        else:
            y_a, y_b, lam = targets, targets, 1.0

        with autocast():
            outputs = model(inputs)
            loss = mixup_criterion(criterion, outputs, y_a, y_b, lam) if args.mixup > 0 else criterion(outputs, targets)

        optimizer.zero_grad()

        if isinstance(scaler, GradScaler):
            scaler.scale(loss).backward()
            if args.max_grad_norm:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, preds = outputs.max(1)
        correct += preds.eq(targets).sum().item()
        total += targets.size(0)

        if (i + 1) % args.print_freq == 0:
            avg_loss = running_loss / total
            acc = 100 * correct / total
            print(f"Epoch[{epoch}] {i+1}/{iters} Loss:{avg_loss:.4f} Acc:{acc:.2f}%")

    epoch_loss = running_loss / total
    epoch_acc = 100 * correct / total
    if writer:
        writer.add_scalar('train/loss', epoch_loss, epoch)
        writer.add_scalar('train/acc', epoch_acc, epoch)
    return epoch_loss, epoch_acc


@torch.no_grad()
def validate(model, loader, criterion, device, epoch, writer=None):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        with autocast():
            outputs = model(inputs)
            loss = criterion(outputs, targets)

        running_loss += loss.item() * inputs.size(0)
        _, preds = outputs.max(1)
        correct += preds.eq(targets).sum().item()
        total += targets.size(0)

    epoch_loss = running_loss / total
    epoch_acc = 100 * correct / total

    if writer:
        writer.add_scalar('val/loss', epoch_loss, epoch)
        writer.add_scalar('val/acc', epoch_acc, epoch)

    return epoch_loss, epoch_acc


# =============== MAIN ===============

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, required=True)
    parser.add_argument('--img-size', type=int, default=224)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight-decay', type=float, default=1e-3)
    parser.add_argument('--num-workers', type=int, default=2)
    parser.add_argument('--mixup', type=float, default=0.0)
    parser.add_argument('--label-smoothing', type=float, default=0.0)
    parser.add_argument('--focal-gamma', type=float, default=0.0)
    parser.add_argument('--start-frozen', action='store_true')
    parser.add_argument('--unfreeze', action='store_true')
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--experiment-name', type=str, default='finetune_experiment')
    parser.add_argument('--print-freq', type=int, default=50)
    parser.add_argument('--max-grad-norm', type=float, default=None)
    parser.add_argument('--use-weighted-sampler', action='store_true')
    args = parser.parse_args()

    seed_everything(42)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    save_dir = Path('experiments') / args.experiment-name
    save_dir.mkdir(parents=True, exist_ok=True)

    writer = SummaryWriter(log_dir=str(save_dir / 'tb'))

    train_loader, val_loader, num_classes = get_dataloaders(
        args.data_dir, args.batch_size, args.img_size, args.num_workers,
        args.use_weighted_sampler
    )

    print(f"Found {num_classes} classes.")

    # Build model
    model = models.resnet50(pretrained=True)
    in_feats = model.fc.in_features
    model.fc = nn.Linear(in_feats, num_classes)
    model = model.to(device)

    # Freeze backbone (head-only)
    if args.start_frozen and not args.unfreeze:
        print("Training classifier head only...")
        for name, param in model.named_parameters():
            if 'fc' not in name:
                param.requires_grad = False

    # Loss function
    if args.focal_gamma > 0:
        criterion = FocalLoss(gamma=args.focal_gamma)
    else:
        try:
            criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
        except:
            criterion = nn.CrossEntropyLoss()

    # Optimizer + Scheduler
    params_to_optimize = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(params_to_optimize, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    scaler = GradScaler()
    best_val_acc = 0.0

    # Training loop
    for epoch in range(args.epochs):

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, scaler, epoch, args, writer
        )

        scheduler.step()

        val_loss, val_acc = validate(
            model, val_loader, criterion, device, epoch, writer
        )

        print(f"Epoch {epoch} TrainAcc {train_acc:.2f}%  ValAcc {val_acc:.2f}%")

        # Save best checkpoint
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "best_val_acc": best_val_acc
            }, save_dir / "best_checkpoint.pth")
            print(f"Saved best checkpoint: {best_val_acc:.2f}%")

        # Staged unfreeze at 40% epochs
        if args.start_frozen and epoch == int(args.epochs * 0.4):
            print("Unfreezing full model...")
            for p in model.parameters():
                p.requires_grad = True
            optimizer = optim.AdamW(model.parameters(), lr=args.lr / 5, weight_decay=args.weight_decay)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs - epoch)

    print(f"Training completed. Best Val Acc: {best_val_acc:.2f}%")
    writer.close()


if __name__ == "__main__":
    main()
