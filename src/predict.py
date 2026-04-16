import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from torchvision import transforms
from PIL import Image

# ---------------------------
# MODEL (same as training)
# ---------------------------
class CNN(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        self.conv4 = nn.Conv2d(128, 256, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)

        self.pool = nn.MaxPool2d(2, 2)

        self.gap = nn.AdaptiveAvgPool2d((1,1))
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.pool(F.relu(self.bn4(self.conv4(x))))

        x = self.gap(x)
        x = x.view(x.size(0), -1)

        x = self.dropout(x)
        return self.fc(x)

# ---------------------------
# LOAD MODEL (SAFE PATH)
# ---------------------------
model = CNN()

model_path = os.path.join(os.path.dirname(__file__), "model2.pth")

if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model not found at {model_path}")

model.load_state_dict(torch.load(model_path, map_location="cpu"))
model.eval()

# ---------------------------
# TRANSFORM
# ---------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

classes = ['brain', 'chest', 'knee']

# ---------------------------
# PREDICTION FUNCTION
# ---------------------------
def predict_image(img_file):
    try:
        image = Image.open(img_file).convert("RGB")  # ensures compatibility
        image = transform(image).unsqueeze(0)

        with torch.no_grad():
            output = model(image)
            probs = F.softmax(output, dim=1)
            conf, pred = torch.max(probs, 1)

        return classes[pred.item()], conf.item()

    except Exception as e:
        print("Prediction Error:", str(e))
        return "unknown", 0.0