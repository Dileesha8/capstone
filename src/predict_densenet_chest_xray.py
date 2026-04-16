import tensorflow as tf
import numpy as np
import cv2
from src.severity_score_a import severity_score
from src.forecast_a import forecast_risk
from src.gradcam_a import make_gradcam_heatmap, generate_sidebyside_b64
from src.model_loader import load_legacy_h5
import os

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
model_path = os.path.join(BASE_DIR, "models", "severity_model.h5")

model = load_legacy_h5(model_path, compile=False)
classes = ["mild", "moderate", "severe"]

def predict(img_path):
    img = cv2.imread(img_path)

    if img is None:
        raise ValueError("Image not found or path incorrect")

    img = cv2.resize(img, (224, 224))
    img = img / 255.0
    img_array = np.expand_dims(img, 0)

    probs = model.predict(img_array)[0]
    label = classes[np.argmax(probs)]

    score = severity_score(probs)
    risk  = forecast_risk(score)

    return label, score, risk