"""
PyTorch Training Script for Plant & Fruit Disease Classification.
Supports Transfer Learning with MobileNetV3, EfficientNet, and ResNet.
"""

import os
import sys
import json
import argparse
import time
from typing import Tuple
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models


def get_data_transforms(img_size: int = 224) -> Tuple[transforms.Compose, transforms.Compose]:
    """Data augmentation and normalization for training and validation."""
    train_transforms = transforms.Compose([
        transforms.Resize((img_size + 32, img_size + 32)),
        transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
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
    """Build model backbone with custom classification head."""
    print(f"[Training] Initializing model architecture: {arch} for {num_classes} classes...")
    
    if arch == "efficientnet_b0":
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Sequential(
            nn.Dropout(p=0.3, inplace=True),
            nn.Linear(in_features, num_classes)
        )
    elif arch == "resnet50":
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(p=0.4),
            nn.Linear(in_features, num_classes)
        )
    else:  # Default mobilenet_v3_small
        model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
        in_features = model.classifier[3].in_features
        model.classifier[3] = nn.Sequential(
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(in_features, num_classes)
        )

    return model


def create_sample_dataset_if_empty(data_dir: str):
    """Creates a sample dataset structure with generated placeholder images if data_dir is empty."""
    train_dir = os.path.join(data_dir, "train")
    val_dir = os.path.join(data_dir, "val")

    if os.path.exists(train_dir) and len(os.listdir(train_dir)) > 0:
        return

    print(f"[Training] No existing dataset found at '{data_dir}'. Creating starter mock classes...")
    sample_classes = ["tomato_healthy", "tomato_early_blight", "apple_scab", "apple_healthy"]
    
    for split, count in [(train_dir, 6), (val_dir, 2)]:
        for cls in sample_classes:
            folder = os.path.join(split, cls)
            os.makedirs(folder, exist_ok=True)
            for i in range(count):
                img_path = os.path.join(folder, f"sample_{i}.png")
                # Create a small sample image
                color = (40, 180, 40) if "healthy" in cls else (180, 80, 40)
                img = Image.new("RGB", (224, 224), color=color)
                img.save(img_path)

    print(f"[Training] Starter mock dataset created at: {data_dir}")


def train_model(
    data_dir: str = "dataset",
    arch: str = "mobilenet_v3_small",
    epochs: int = 5,
    batch_size: int = 16,
    lr: float = 0.001,
    output_dir: str = "models"
):
    """Fine-tune the model on dataset."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Training] Using computing device: {device}")
    if torch.cuda.is_available():
        print(f"[Training] GPU: {torch.cuda.get_device_name(0)}")

    os.makedirs(output_dir, exist_ok=True)
    create_sample_dataset_if_empty(data_dir)

    train_dir = os.path.join(data_dir, "train")
    val_dir = os.path.join(data_dir, "val")

    train_transforms, val_transforms = get_data_transforms()

    train_dataset = datasets.ImageFolder(train_dir, transform=train_transforms)
    val_dataset = datasets.ImageFolder(val_dir, transform=val_transforms)

    class_names = train_dataset.classes
    num_classes = len(class_names)
    print(f"[Training] Loaded {len(train_dataset)} training images across {num_classes} classes:")
    print(f"           Classes: {class_names}")

    # Save class names mapping
    with open(os.path.join(output_dir, "classes.json"), "w", encoding="utf-8") as f:
        json.dump(class_names, f, ensure_ascii=False, indent=2)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    model = build_model(arch, num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_acc = 0.0
    best_model_path = os.path.join(output_dir, "best_model.pth")

    print(f"\n[Training] Starting Training for {epochs} Epochs...\n" + "="*50)
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        # Training Phase
        model.train()
        running_loss = 0.0
        running_corrects = 0

        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            _, preds = torch.max(outputs, 1)
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        scheduler.step()
        train_loss = running_loss / len(train_dataset)
        train_acc = running_corrects.double() / len(train_dataset)

        # Validation Phase
        model.eval()
        val_loss = 0.0
        val_corrects = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                _, preds = torch.max(outputs, 1)
                val_loss += loss.item() * inputs.size(0)
                val_corrects += torch.sum(preds == labels.data)

        val_loss = val_loss / len(val_dataset)
        val_acc = val_corrects.double() / len(val_dataset)

        print(f"Epoch [{epoch:02d}/{epochs:02d}] "
              f"Train Loss: {train_loss:.4f} | Acc: {train_acc*100:.2f}%  ||  "
              f"Val Loss: {val_loss:.4f} | Acc: {val_acc*100:.2f}%")

        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_model_path)
            print(f"  --> Saved new best checkpoint to {best_model_path}")

    elapsed = time.time() - start_time
    print("="*50)
    print(f"[Training Complete] Total Time: {elapsed//60:.0f}m {elapsed%60:.1f}s | Best Val Accuracy: {best_val_acc*100:.2f}%\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Plant Disease Classifier")
    parser.add_argument("--data_dir", type=str, default="dataset", help="Path to dataset root folder")
    parser.add_argument("--arch", type=str, default="mobilenet_v3_small", choices=["mobilenet_v3_small", "efficientnet_b0", "resnet50"])
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    args = parser.parse_args()

    train_model(
        data_dir=args.data_dir,
        arch=args.arch,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr
    )
