import os
import shutil
from opacity_score import compute_opacity_score
from severity_rules import assign_severity

SOURCE = r"C:\Users\91701\project__\dataset\chest_xray\train\PNEUMONIA"
DEST = "dataset/severity/train"

os.makedirs(DEST, exist_ok=True)

for img_name in os.listdir(SOURCE):
    img_path = os.path.join(SOURCE, img_name)

    if not img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
        continue

    opacity = compute_opacity_score(img_path)
    severity = assign_severity(opacity)

    target_dir = os.path.join(DEST, severity)
    os.makedirs(target_dir, exist_ok=True)

    shutil.copy(img_path, os.path.join(target_dir, img_name))

print("✅ Severity dataset created successfully!")
