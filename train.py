"""
PyTorch Training Script for Plant & Fruit Disease and Ripeness Classification.
Implements the ResNet50 Architecture specified in the Project Documentation:
- Pre-trained ImageNet Backbone (ResNet-50)
- Custom Classification Head: Linear(2048, 512) -> ReLU -> Dropout(0.3) -> Linear(512, num_classes)
- Adam Optimizer (lr=0.0001) & CrossEntropyLoss
- Generates weights at models/plant_fruit_resnet50_best.pth
"""

import os
import sys
import json
import time
import argparse
from typing import Tuple, Dict, Any

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from PIL import Image


def get_data_transforms(img_size: int = 224) -> Tuple[transforms.Compose, transforms.Compose]:
    """Data augmentation and normalization pipeline according to Section 3.2.2."""
    train_transforms = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transforms = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    return train_transforms, val_transforms


def build_model(arch: str, num_classes: int) -> nn.Module:
    """Build model backbone with custom classification head matching report Section 3.4.1."""
    print(f"[Training] Initializing model architecture: {arch} for {num_classes} classes...")

    if arch == "resnet50":
        # ResNet50 Transfer Learning with custom classifier head
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        num_ftrs = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Linear(num_ftrs, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
    elif arch == "efficientnet_b0":
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Sequential(
            nn.Dropout(p=0.3, inplace=True),
            nn.Linear(in_features, num_classes)
        )
    else:  # mobilenet_v3_small
        model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
        in_features = model.classifier[3].in_features
        model.classifier[3] = nn.Sequential(
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(in_features, num_classes)
        )

    return model


def train_model(
    data_dir: str = "dataset",
    arch: str = "resnet50",
    epochs: int = 5,
    batch_size: int = 16,
    lr: float = 0.0001,
    output_dir: str = "models"
) -> Dict[str, Any]:
    """Train the model with PyTorch."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[Training] Using computing device: {device}")
    if torch.cuda.is_available():
        print(f"[Training] GPU Acceleration: {torch.cuda.get_device_name(0)}")

    os.makedirs(output_dir, exist_ok=True)
    train_dir = os.path.join(data_dir, "train")
    val_dir = os.path.join(data_dir, "val")

    if not os.path.exists(train_dir) or not os.path.exists(val_dir):
        raise FileNotFoundError(f"Dataset directories '{train_dir}' or '{val_dir}' not found.")

    train_transforms, val_transforms = get_data_transforms()
    train_dataset = datasets.ImageFolder(train_dir, transform=train_transforms)
    val_dataset = datasets.ImageFolder(val_dir, transform=val_transforms)

    class_names = train_dataset.classes
    num_classes = len(class_names)
    print(f"[Training] Loaded {len(train_dataset)} training images and {len(val_dataset)} validation images")
    print(f"[Training] Target Classes ({num_classes}): {class_names}")

    # Save classes.json
    classes_path = os.path.join(output_dir, "classes.json")
    with open(classes_path, "w", encoding="utf-8") as f:
        json.dump(class_names, f, ensure_ascii=False, indent=2)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    model = build_model(arch, num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_val_acc = 0.0
    best_weights_path = os.path.join(output_dir, "plant_fruit_resnet50_best.pth")
    training_history = []

    print("\n" + "=" * 65)
    print(f"[STARTING TRAINING]: {arch.upper()} | {epochs} Epochs | Batch Size: {batch_size}")
    print("=" * 65)
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()

        # 1. Training Phase
        model.train()
        running_train_loss = 0.0
        running_train_corrects = 0

        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            _, preds = torch.max(outputs, 1)
            running_train_loss += loss.item() * inputs.size(0)
            running_train_corrects += torch.sum(preds == labels.data).item()

        train_loss = running_train_loss / len(train_dataset)
        train_acc = running_train_corrects / len(train_dataset) * 100.0

        # 2. Validation Phase
        model.eval()
        running_val_loss = 0.0
        running_val_corrects = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                _, preds = torch.max(outputs, 1)
                running_val_loss += loss.item() * inputs.size(0)
                running_val_corrects += torch.sum(preds == labels.data).item()

        val_loss = running_val_loss / len(val_dataset)
        val_acc = running_val_corrects / len(val_dataset) * 100.0
        epoch_time = time.time() - epoch_start

        record = {
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "train_acc": round(train_acc, 2),
            "val_loss": round(val_loss, 4),
            "val_acc": round(val_acc, 2),
            "time_sec": round(epoch_time, 2)
        }
        training_history.append(record)

        print(f"Epoch [{epoch:02d}/{epochs:02d}] "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}% | "
              f"Time: {epoch_time:.1f}s")

        # Save best model
        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            # Save state dict and metadata
            save_payload = {
                "architecture": arch,
                "num_classes": num_classes,
                "classes": class_names,
                "best_val_acc": best_val_acc,
                "state_dict": model.state_dict(),
                "model_state_dict": model.state_dict()
            }
            torch.save(save_payload, best_weights_path)
            # Also save standard best_model.pth
            torch.save(save_payload, os.path.join(output_dir, "best_model.pth"))

    total_time = time.time() - start_time
    print("=" * 65)
    print(f"[OK] TRAINING COMPLETE in {total_time:.2f}s!")
    print(f"[*] Best Validation Accuracy: {best_val_acc:.2f}%")
    print(f"[+] Best Weights Saved to: {best_weights_path}")
    print("=" * 65)

    # Save summary report
    summary_path = os.path.join(output_dir, "training_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({
            "architecture": arch,
            "num_classes": num_classes,
            "classes": class_names,
            "best_val_acc": best_val_acc,
            "total_time_seconds": round(total_time, 2),
            "history": training_history
        }, f, ensure_ascii=False, indent=2)

    return {
        "best_val_acc": best_val_acc,
        "best_weights_path": best_weights_path,
        "history": training_history
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Plant and Fruit Disease Classifier")
    parser.add_argument("--data-dir", type=str, default="dataset", help="Path to dataset")
    parser.add_argument("--arch", type=str, default="resnet50", choices=["resnet50", "mobilenet_v3_small", "efficientnet_b0"])
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.0001, help="Learning rate")
    parser.add_argument("--output-dir", type=str, default="models", help="Output directory")

    args = parser.parse_args()
    train_model(
        data_dir=args.data_dir,
        arch=args.arch,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        output_dir=args.output_dir
    )
