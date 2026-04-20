import os
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms

# ------------------------ Config ------------------------ #
TRAIN_DIR = "./multilabel_dataset_2/train/images"
VALID_DIR = "./multilabel_dataset_2/valid/images"
TRAIN_CSV = "./multilabel_dataset_2/train/labels.csv"
VALID_CSV = "./multilabel_dataset_2/valid/labels.csv"

CLASS_NAMES = ["normal","spaghetti_and_stringing","warp"]
NUM_CLASSES = len(CLASS_NAMES)

BATCH_SIZE = 32
EPOCHS = 50
LR = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ------------------------ Dataset ------------------------ #
class MultiLabelDataset(Dataset):
    def __init__(self, image_dir, csv_file, class_names, transform=None):
        self.image_dir = image_dir
        self.df = pd.read_csv(csv_file)
        self.class_names = class_names
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        filename = str(row["image_name"]).strip()
        image_path = os.path.join(self.image_dir, filename)

        if filename == "" or filename.lower() == "nan":
            raise ValueError(f"Bad filename at row {idx}: {filename!r}")

        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Missing image at row {idx}: {image_path}")

        image = Image.open(image_path).convert("RGB")
        labels = torch.tensor(row[self.class_names].values.astype("float32"))

        if self.transform:
            image = self.transform(image)

        return image, labels


# ------------------------ Transforms ------------------------ #
train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])


# ------------------------ Train / Eval ------------------------ #
def train_one_epoch(model, loader, criterion, optimizer, device, threshold=0.5):
    model.train()
    total_loss = 0.0
    total_correct_labels = 0
    total_labels = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)                    # logits
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)

        probs = torch.sigmoid(outputs)
        preds = (probs > threshold).float()

        total_correct_labels += (preds == labels).sum().item()
        total_labels += labels.numel()

    avg_loss = total_loss / len(loader.dataset)
    label_acc = total_correct_labels / total_labels
    return avg_loss, label_acc


def evaluate(model, loader, criterion, device, threshold=0.5):
    model.eval()
    total_loss = 0.0
    total_correct_labels = 0
    total_labels = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)

            probs = torch.sigmoid(outputs)
            preds = (probs > threshold).float()

            total_correct_labels += (preds == labels).sum().item()
            total_labels += labels.numel()

    avg_loss = total_loss / len(loader.dataset)
    label_acc = total_correct_labels / total_labels
    return avg_loss, label_acc


# ------------------------ Main ------------------------ #
if __name__ == "__main__":
    train_dataset = MultiLabelDataset(
        image_dir=TRAIN_DIR,
        csv_file=TRAIN_CSV,
        class_names=CLASS_NAMES,
        transform=train_transforms
    )

    val_dataset = MultiLabelDataset(
        image_dir=VALID_DIR,
        csv_file=VALID_CSV,
        class_names=CLASS_NAMES,
        transform=val_transforms
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    model = model.to(DEVICE)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    for epoch in range(EPOCHS):
        train_loss, train_label_acc = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
        val_loss, val_label_acc = evaluate(model, val_loader, criterion, DEVICE)

        scheduler.step()

        print(
            f"Epoch {epoch+1:02d}/{EPOCHS} | "
            f"Train Loss: {train_loss:.4f} Label Acc: {train_label_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} Label Acc: {val_label_acc:.4f}"
        )

    torch.save({
        "model_state_dict": model.state_dict(),
        "class_names": CLASS_NAMES
    }, "resnet18_multilabel.pth")

    print("Model saved to resnet18_multilabel.pth")