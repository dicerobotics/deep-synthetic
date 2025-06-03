"""
Script Name: resnet18_mwir_train_val.py

Purpose:
    This script fine tunes a pre-trains a ResNet18 classifier on MWIR (mid-wave infrared) military vehicle images
    using PyTorch, with automatic validation after each epoch and final accuracy reporting.

Features:
    - Supports grayscale MWIR input by converting to 3-channel RGB.
    - Automatically loads data from split train/val folders.
    - Computes and prints loss and accuracy for both training and validation per epoch.
    - Saves the trained model to disk.

Dependencies:
    - PyTorch
    - Torchvision
    - PIL (Pillow)
    - OS
    - json

Assumptions:
    - Dataset is already split using the `split_mwir_dataset.py` script.
    - Folder structure is: 
        /your_dataset/
            ├── train/
            │   └── [class_name]/
            └── val/
                └── [class_name]/

Usage:
    Modify DATA_DIR to point to the base dataset directory containing `train/` and `val/`.
    Adjust NUM_CLASSES if your dataset changes.
"""


import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import os
import json

# ---- Config ----
IMAGE_SIZE = 224
BATCH_SIZE = 16
NUM_CLASSES = 8  # Update this based on your dataset
DATA_DIR = "../datasets/DSIAC_ATR_Resnet18_Split"  # should contain /train and /val
MODEL_DIR = "../models/classifiers"

# ---- Custom Dataset ----
class MWIRDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(sorted(os.listdir(root_dir)))}
        for cls in self.class_to_idx:
            class_dir = os.path.join(root_dir, cls)
            if not os.path.isdir(class_dir):
                continue
            for img_name in os.listdir(class_dir):
                self.samples.append((os.path.join(class_dir, img_name), self.class_to_idx[cls]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("L").convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label

# Optional: calculate mean and std of MWIR dataset
def compute_mean_std(dataset):
    loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=False)
    mean = 0.
    std = 0.
    total = 0
    for images, _ in loader:
        batch_samples = images.size(0)
        images = images.view(batch_samples, images.size(1), -1)
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)
        total += batch_samples
    mean /= total
    std /= total
    return mean, std

# ---- Transforms ----
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]), # ImageNet-trained models Mean and Std
    transforms.Normalize([0.2342, 0.2342, 0.2342], [0.0662, 0.0662, 0.0662]) # DSIAC_ATR_Resnet18 Mean and Std
])

# ---- Data Loaders ----
train_dataset = MWIRDataset(os.path.join(DATA_DIR, "train"), transform=transform)
val_dataset = MWIRDataset(os.path.join(DATA_DIR, "val"), transform=transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)


# Needed once to calculate mean and std of dataset
# mean, std = compute_mean_std(train_dataset)
# print(f"MWIR dataset mean: {mean}, std: {std}")



# ---- Model ----
model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
model = model.cuda()

# ---- Training Setup ----
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

# ---- Training + Validation Loop ----
for epoch in range(5):
    model.train()
    total_loss = 0
    for inputs, labels in train_loader:
        inputs, labels = inputs.cuda(), labels.cuda()
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        # if torch.isnan(loss):
        #     print("[Error] NaN loss detected! Batch skipped.")
        #     continue
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

    # ---- Validation ----
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.cuda(), labels.cuda()
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    acc = 100 * correct / total
    print(f"Epoch {epoch+1}, Validation Accuracy: {acc:.2f}%")


# ---- Save model ----
torch.save(model.state_dict(), os.path.join(MODEL_DIR, "resnet18_mwir_weights.pth"))
# torch.save(model, os.path.join(MODEL_DIR, "resnet18_mwir_full.pth"))  # Save full model (optional, Not recommended)
print("Model trained and saved.")

# ---- Save Class Index Mapping ----

with open(os.path.join(MODEL_DIR, "resnet18_mwir_class_to_idx.json"), "w") as f:
    json.dump(train_dataset.class_to_idx, f)
