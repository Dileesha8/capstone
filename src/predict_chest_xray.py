import tensorflow as tf
import numpy as np
import cv2
from src.severity_score_a import severity_score
from src.forecast_a import forecast_risk
from src.gradcam_a import make_gradcam_heatmap, overlay_heatmap

model = tf.keras.models.load_model("models/severity_model.h5")
classes = ["mild", "moderate", "severe"]

def predict(img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (224,224))
    img = img/255.0
    img_array = np.expand_dims(img,0)

    probs = model.predict(img_array)[0]
    label = classes[np.argmax(probs)]

    score = severity_score(probs)
    risk = forecast_risk(score)

    heatmap = make_gradcam_heatmap(
        img_array, model,
        last_conv_layer_name="conv5_block3_out"  # for ResNet50
    )
    overlay_heatmap(img_path, heatmap)

    return label, score, risk
