from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from tqdm import tqdm

from model import MudraCNN


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train CNN for Naruto hand-sign (mudra) classification.")
    parser.add_argument("--dataset", type=Path, default=Path("dataset"), help="Dataset root containing train/ and test/.")
    parser.add_argument("--output", type=Path, default=Path("models/mudra_cnn.pt"), help="Output model path.")
    parser.add_argument("--image-size", type=int, default=128, help="Input image size.")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size.")
    parser.add_argument("--epochs", type=int, default=30, help="Max training epochs.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay.")
    parser.add_argument("--val-ratio", type=float, default=0.15, help="Validation split ratio from train set.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--workers", type=int, default=2, help="DataLoader workers.")
    return parser.parse_args()


def make_transforms(image_size: int) -> Tuple[transforms.Compose, transforms.Compose]:
    train_tfms = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(18),
            transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.20, hue=0.03),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    eval_tfms = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    return train_tfms, eval_tfms


def split_train_val(train_dir: Path, train_tfms, eval_tfms, val_ratio: float, seed: int):
    base_dataset = datasets.ImageFolder(train_dir, transform=train_tfms)
    eval_base_dataset = datasets.ImageFolder(train_dir, transform=eval_tfms)

    total = len(base_dataset)
    val_size = max(1, int(total * val_ratio))
    train_size = total - val_size

    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(total, generator=generator).tolist()
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]

    train_subset = Subset(base_dataset, train_indices)
    val_subset = Subset(eval_base_dataset, val_indices)

    return train_subset, val_subset, base_dataset.classes


def evaluate(model: nn.Module, loader: DataLoader, loss_fn: nn.Module, device: torch.device) -> Tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total = 0
    correct = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            loss = loss_fn(logits, labels)

            total_loss += float(loss.item()) * labels.size(0)
            preds = logits.argmax(dim=1)
            correct += int((preds == labels).sum().item())
            total += labels.size(0)

    return total_loss / max(1, total), correct / max(1, total)


def run() -> None:
    args = parse_args()

    torch.manual_seed(args.seed)

    train_dir = args.dataset / "train"
    test_dir = args.dataset / "test"
    if not train_dir.exists() or not test_dir.exists():
        raise FileNotFoundError("Dataset must contain train/ and test/ directories.")

    train_tfms, eval_tfms = make_transforms(args.image_size)
    train_ds, val_ds, class_names = split_train_val(train_dir, train_tfms, eval_tfms, args.val_ratio, args.seed)
    test_ds = datasets.ImageFolder(test_dir, transform=eval_tfms)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MudraCNN(num_classes=len(class_names)).to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)

    best_val_acc = 0.0
    best_state: Dict[str, torch.Tensor] | None = None
    patience = 6
    no_improve = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        running_correct = 0
        seen = 0

        progress = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", leave=False)
        for images, labels in progress:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += float(loss.item()) * labels.size(0)
            preds = logits.argmax(dim=1)
            running_correct += int((preds == labels).sum().item())
            seen += labels.size(0)

            progress.set_postfix(loss=f"{running_loss / max(1, seen):.4f}", acc=f"{running_correct / max(1, seen):.3f}")

        train_loss = running_loss / max(1, seen)
        train_acc = running_correct / max(1, seen)
        val_loss, val_acc = evaluate(model, val_loader, loss_fn, device)

        scheduler.step(val_acc)

        print(
            f"Epoch {epoch:02d} | train_loss={train_loss:.4f} train_acc={train_acc:.3f} "
            f"| val_loss={val_loss:.4f} val_acc={val_acc:.3f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= patience:
            print(f"Early stopping after {epoch} epochs (best val_acc={best_val_acc:.3f}).")
            break

    if best_state is None:
        best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    test_loss, test_acc = evaluate(model, test_loader, loss_fn, device)
    print(f"Test  | loss={test_loss:.4f} acc={test_acc:.3f}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "state_dict": best_state,
        "class_names": class_names,
        "image_size": args.image_size,
        "val_acc": best_val_acc,
        "test_acc": test_acc,
    }
    torch.save(payload, args.output)

    meta_path = args.output.with_suffix(".json")
    meta = {
        "model_path": str(args.output),
        "class_names": class_names,
        "image_size": args.image_size,
        "val_acc": best_val_acc,
        "test_acc": test_acc,
        "dataset": str(args.dataset),
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"Saved model: {args.output}")
    print(f"Saved metadata: {meta_path}")


if __name__ == "__main__":
    run()
