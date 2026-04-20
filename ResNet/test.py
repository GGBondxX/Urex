import os
import pandas as pd
import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TEST_DIR = "./multilabel_dataset_2/test.v1i.multiclass/test"
TEST_CSV = "./multilabel_dataset_2/test.v1i.multiclass/test/_classes.csv"   # change this to your actual CSV file

# Load checkpoint once
checkpoint = torch.load("resnet18_multilabel.pth", map_location=DEVICE)
CLASS_NAMES = checkpoint["class_names"]
NUM_CLASSES = len(CLASS_NAMES)

# Rebuild model
model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
model.load_state_dict(checkpoint["model_state_dict"])
model = model.to(DEVICE)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

class MultiLabelTestDataset(Dataset):
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

        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Missing image: {image_path}")

        image = Image.open(image_path).convert("RGB")
        labels = torch.tensor(row[self.class_names].values.astype("float32"))

        if self.transform:
            image = self.transform(image)

        return image, labels, filename

def evaluate_test_with_confusion(model, loader, device, threshold=0.5):
    model.eval()

    # confusion[class_idx, actual, predicted]
    # actual: 0 or 1
    # predicted: 0 or 1
    confusion = torch.zeros(NUM_CLASSES, 2, 2, dtype=torch.int64)

    total_correct_labels = 0
    total_labels = 0

    with torch.no_grad():
        for images, labels, filenames in loader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            probs = torch.sigmoid(logits)
            preds = (probs > threshold).int()

            labels_int = labels.int()

            # Optional: print per-image predictions
            for b in range(len(filenames)):
                print(f"Image: {filenames[b]}")
                for i, name in enumerate(CLASS_NAMES):
                    print(f"  {name}: {probs[b, i].item():.2%}")

                predicted_labels = [
                    CLASS_NAMES[i]
                    for i in range(NUM_CLASSES)
                    if preds[b, i].item() == 1
                ]

                true_labels = [
                    CLASS_NAMES[i]
                    for i in range(NUM_CLASSES)
                    if labels_int[b, i].item() == 1
                ]

                print("→ Predicted labels:", ", ".join(predicted_labels) if predicted_labels else "None")
                print("→ True labels     :", ", ".join(true_labels) if true_labels else "None")
                print()

            total_correct_labels += (preds == labels_int).sum().item()
            total_labels += labels.numel()

            for c in range(NUM_CLASSES):
                y_true = labels_int[:, c].cpu()
                y_pred = preds[:, c].cpu()

                tn = ((y_true == 0) & (y_pred == 0)).sum().item()
                fp = ((y_true == 0) & (y_pred == 1)).sum().item()
                fn = ((y_true == 1) & (y_pred == 0)).sum().item()
                tp = ((y_true == 1) & (y_pred == 1)).sum().item()

                confusion[c, 0, 0] += tn
                confusion[c, 0, 1] += fp
                confusion[c, 1, 0] += fn
                confusion[c, 1, 1] += tp

    label_acc = total_correct_labels / total_labels
    return confusion, label_acc

def print_confusion_metrics(confusion, class_names):
    print("\n===== Confusion Matrices =====\n")

    for i, class_name in enumerate(class_names):
        tn = confusion[i, 0, 0].item()
        fp = confusion[i, 0, 1].item()
        fn = confusion[i, 1, 0].item()
        tp = confusion[i, 1, 1].item()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        print(f"{class_name}:")
        print(f"            Pred 0   Pred 1")
        print(f"Actual 0      {tn:5d}   {fp:5d}")
        print(f"Actual 1      {fn:5d}   {tp:5d}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1-score:  {f1:.4f}")
        print()

if __name__ == "__main__":
    test_dataset = MultiLabelTestDataset(
        image_dir=TEST_DIR,
        csv_file=TEST_CSV,
        class_names=CLASS_NAMES,
        transform=transform
    )

    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    confusion, label_acc = evaluate_test_with_confusion(
        model, test_loader, DEVICE, threshold=0.5
    )

    print(f"Overall label accuracy: {label_acc:.4f}")
    print_confusion_metrics(confusion, CLASS_NAMES)