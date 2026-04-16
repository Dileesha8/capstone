import tensorflow as tf
import numpy as np
import cv2
from src.model_loader import load_legacy_h5
import os

BASE_DIR   = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
model_path = os.path.join(BASE_DIR, "models", "brain_model.h5")

model = load_legacy_h5(model_path, compile=False)

classes = ["glioma", "meningioma", "pituitary", "notumor"]

def predict(img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (224, 224))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    preds      = model.predict(img)[0]
    label      = classes[np.argmax(preds)]
    confidence = round(float(np.max(preds)) * 100, 2)

    score = confidence

    if score < 40:
        risk = "Low Risk (15%)"
    elif score < 70:
        risk = "Moderate Risk (45%)"
    else:
        risk = "High Risk (80%)"

    return label, score, risk