import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from sklearn.preprocessing import label_binarize

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import cv2

# =========================
# PATH
# =========================
BASE_DIR = r"C:\Users\<username>\Downloads\project"
train_dir = os.path.join(BASE_DIR, "chest_xray_dataset", "train")
val_dir   = os.path.join(BASE_DIR, "chest_xray_dataset", "val")

# =========================
# CONFIG
# =========================
IMG_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 15

# =========================
# DATA
# =========================
train_gen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    zoom_range=0.15,
    horizontal_flip=True,
    brightness_range=[0.8,1.2],
    shear_range=0.1
)

val_gen = ImageDataGenerator(rescale=1./255)

train_data = train_gen.flow_from_directory(
    train_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical"
)

val_data = val_gen.flow_from_directory(
    val_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False
)

# =========================
# CLASS WEIGHTS
# =========================
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_data.classes),
    y=train_data.classes
)
class_weights = dict(enumerate(class_weights))

# =========================
# MODEL
# =========================
base = DenseNet121(weights="imagenet", include_top=False, input_shape=(224,224,3))

base.trainable = True
for layer in base.layers[:100]:
    layer.trainable = False

x = GlobalAveragePooling2D()(base.output)
x = Dense(256, activation="relu")(x)
x = Dropout(0.5)(x)
out = Dense(train_data.num_classes, activation="softmax")(x)

model = Model(base.input, out)

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-4),
    loss="categorical_crossentropy",
    metrics=[
        "accuracy",
        tf.keras.metrics.Precision(),
        tf.keras.metrics.Recall(),
        tf.keras.metrics.AUC()
    ]
)

# =========================
# CALLBACKS
# =========================
#callbacks = [
#    EarlyStopping(patience=5, restore_best_weights=True),
#   ReduceLROnPlateau(patience=3)
#]

# =========================
# TRAIN
# =========================
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS,
    class_weight=class_weights,
    #callbacks=callbacks
)

# =========================
# PLOTS: ACCURACY & LOSS
# =========================
plt.figure()
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Accuracy - CHEST MODEL')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Validation'])
plt.show()

plt.figure()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Loss - CHEST MODEL')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Train', 'Validation'])
plt.show()

# =========================
# PREDICTIONS
# =========================
val_data.reset()
preds = model.predict(val_data)
y_pred = np.argmax(preds, axis=1)
y_true = val_data.classes

print("\nClassification Report:\n")
print(classification_report(y_true, y_pred))

# =========================
# CONFUSION MATRIX
# =========================
cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix - CHEST MODEL')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# =========================
# ROC CURVE + AUC
# =========================
n_classes = train_data.num_classes
y_true_bin = label_binarize(y_true, classes=range(n_classes))

plt.figure()

for i in range(n_classes):
    fpr, tpr, _ = roc_curve(y_true_bin[:, i], preds[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'Class {i} (AUC = {roc_auc:.2f})')

plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - CHEST MODEL')
plt.legend()
plt.show()

# =========================
# PRECISION-RECALL CURVE
# =========================
plt.figure()

for i in range(n_classes):
    precision, recall, _ = precision_recall_curve(y_true_bin[:, i], preds[:, i])
    plt.plot(recall, precision, label=f'Class {i}')

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve - CHEST MODEL')
plt.legend()
plt.show()

# =========================
# FALSE POSITIVE / NEGATIVE
# =========================
fp = []
fn = []

for i in range(len(y_true)):
    if y_true[i] != y_pred[i]:
        if y_pred[i] > y_true[i]:
            fp.append(i)
        else:
            fn.append(i)

print("False Positives:", len(fp))
print("False Negatives:", len(fn))

# =========================
# SHOW MISCLASSIFIED IMAGES
# =========================


def show_misclassified(indices, title):
    plt.figure(figsize=(10,5))

    for i, idx in enumerate(indices[:6]):  # show max 6
        img_path = val_data.filepaths[idx]

        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        plt.subplot(2,3,i+1)
        plt.imshow(img)
        plt.title(f"Pred: {y_pred[idx]} | True: {y_true[idx]}")
        plt.axis('off')

    plt.suptitle(title)
    plt.show()

# =========================
# SAVE MODEL
# =========================
model_dir = os.path.join(BASE_DIR, "models")
os.makedirs(model_dir, exist_ok=True)

model.save(os.path.join(model_dir, "chest_model15.keras"))

print("✅ Chest model saved successfully!")
