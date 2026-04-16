import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import tensorflow as tf
import numpy as np
import cv2

# ── severity helpers ──────────────────────────────────────────────────────────

def severity_score(probabilities, class_names):
    weights = {}
    for i, name in enumerate(class_names):
        n = name.lower()
        if 'normal' in n or 'healthy' in n:
            weights[i] = 0
        elif 'doubtful' in n or 'minimal' in n:
            weights[i] = 25
        elif 'mild' in n:
            weights[i] = 50
        elif 'moderate' in n:
            weights[i] = 75
        elif 'severe' in n or 'advanced' in n:
            weights[i] = 100
        else:
            weights[i] = (i / max(len(class_names) - 1, 1)) * 100
    return round(sum(probabilities[i] * weights[i] for i in range(len(probabilities))), 2)


def forecast_risk(score):
    if score < 20:
        return "Low Risk (10-15%)", "Minimal joint space narrowing, regular monitoring recommended"
    elif score < 40:
        return "Mild Risk (25-30%)", "Early OA signs, lifestyle modifications advised"
    elif score < 60:
        return "Moderate Risk (50-60%)", "Definite OA, consider treatment options"
    elif score < 80:
        return "High Risk (70-80%)", "Advanced OA, medical intervention recommended"
    else:
        return "Severe Risk (90%+)", "Severe OA, surgical consultation recommended"


# ── KL grade mapping ──────────────────────────────────────────────────────────

# Maps class name → KL grade number (for the recommendation lookup in app.py)
CLASS_TO_KL = {
    'normal':   0,
    'doubtful': 1,
    'mild':     2,
    'moderate': 3,
    'severe':   4,
}

# ── model (loaded once at import time) ───────────────────────────────────────

_MODEL_PATH = os.path.join(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..")),
    "models", "knee_oa_best.h5"
)
import tf_keras
_model      = tf_keras.models.load_model(_MODEL_PATH)
_CLASS_NAMES = ['doubtful', 'mild', 'moderate', 'normal', 'severe']  # must match training order
_IMG_SIZE    = 224


# ── public predict() ─────────────────────────────────────────────────────────

def predict(image_path):
    """
    Returns (label, score, risk)
      label : predicted class name  e.g. "moderate"
      score : confidence %          e.g. 78.4
      risk  : risk string           e.g. "Moderate Risk (50-60%)"
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Cannot read image: {image_path}")

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (_IMG_SIZE, _IMG_SIZE))
    img = img / 255.0
    img_array = np.expand_dims(img, axis=0)

    predictions   = _model.predict(img_array, verbose=0)[0]
    predicted_idx = int(np.argmax(predictions))
    label         = _CLASS_NAMES[predicted_idx]
    confidence    = round(float(predictions[predicted_idx]) * 100, 2)

    sev            = severity_score(predictions, _CLASS_NAMES)
    risk_level, _  = forecast_risk(sev)

    return label, confidence, risk_level


# ── standalone test ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "upload.jpg"
    label, score, risk = predict(path)
    print(f"Class : {label}  (KL {CLASS_TO_KL.get(label, '?')})")
    print(f"Score : {score:.2f}%")
    print(f"Risk  : {risk}")