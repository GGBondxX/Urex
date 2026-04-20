import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Must match training classes exactly
checkpoint = torch.load("resnet18_multilabel.pth", map_location=DEVICE)
CLASS_NAMES = checkpoint["class_names"]
NUM_CLASSES = len(CLASS_NAMES)

model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)

checkpoint = torch.load("resnet18_multilabel.pth", map_location=DEVICE)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval().to(DEVICE)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

def predict(image_path, threshold=0.5):
    img = Image.open(image_path).convert("RGB")
    tensor = transform(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(tensor)[0]
        probs = torch.sigmoid(logits)

    print(f"Image: {os.path.basename(image_path)}")
    for name, p in zip(CLASS_NAMES, probs):
        print(f"  {name}: {p:.2%}")

    predicted_labels = [CLASS_NAMES[i] for i, p in enumerate(probs) if p > threshold]

    if predicted_labels:
        print("→ Predicted labels:", ", ".join(predicted_labels))
    else:
        print("→ No labels above threshold")
    print()

TEST_FOLDER = "./dataset/test"

for filename in os.listdir(TEST_FOLDER):
    if filename.lower().endswith((".jpg", ".jpeg", ".png")):
        image_path = os.path.join(TEST_FOLDER, filename)
        predict(image_path, threshold=0.5)