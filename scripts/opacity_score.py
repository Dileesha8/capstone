import cv2
import numpy as np

def compute_opacity_score(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (224, 224))
    image = image / 255.0

    _, thresholded = cv2.threshold(image, 0.6, 1.0, cv2.THRESH_BINARY)
    opacity_ratio = np.sum(thresholded) / (224 * 224)

    return opacity_ratio
