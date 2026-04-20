import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
import seaborn as sns

# =========================
# PATH
# =========================
BASE_DIR = r"C:\Users\<username>\Downloads\project"
train_dir = os.path.join(BASE_DIR, "brain_tumor_dataset", "Training")
val_dir   = os.path.join(BASE_DIR, "brain_tumor_dataset", "Testing")

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
    rotation_range=20,
    zoom_range=0.15,
    horizontal_flip=True,
    brightness_range=[0.8,1.2]
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

# fine-tune
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
callbacks = [
    EarlyStopping(patience=5, restore_best_weights=True),
    ReduceLROnPlateau(patience=3)
]

# =========================
# TRAIN
# =========================
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS,
    class_weight=class_weights,
    callbacks=callbacks
)

# =========================
# EVALUATION
# =========================
val_data.reset()
preds = model.predict(val_data)
y_pred = np.argmax(preds, axis=1)
y_true = val_data.classes

print(classification_report(y_true, y_pred))
print(confusion_matrix(y_true, y_pred))


# Accuracy
plt.figure()
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.legend()
plt.title("Accuracy - BRAIN MODEL")
plt.show()

# Loss
plt.figure()
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.legend()
plt.title("Loss - BRAIN MODEL")
plt.show()

# Precision
plt.figure()
plt.plot(history.history['precision'], label='Train Precision')
plt.plot(history.history['val_precision'], label='Val Precision')
plt.legend()
plt.title("Precision - BRAIN MODEL")
plt.show()

# Recall
plt.figure()
plt.plot(history.history['recall'], label='Train Recall')
plt.plot(history.history['val_recall'], label='Val Recall')
plt.legend()
plt.title("Recall - BRAIN MODEL")
plt.show()

# AUC
plt.figure()
plt.plot(history.history['auc'], label='Train AUC')
plt.plot(history.history['val_auc'], label='Val AUC')
plt.legend()
plt.title("AUC - BRAIN MODEL")
plt.show()


# Binarize labels
n_classes = train_data.num_classes
y_true_bin = label_binarize(y_true, classes=range(n_classes))

fpr = dict()
tpr = dict()
roc_auc = dict()

for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], preds[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Plot ROC
plt.figure()
for i in range(n_classes):
    plt.plot(fpr[i], tpr[i], label=f'Class {i} (AUC = {roc_auc[i]:.2f})')

plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - BRAIN MODEL")
plt.legend()
plt.show()


cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - BRAIN MODEL")
plt.show()
# =========================
# SAVE (SAFE)
# =========================
try:
    model_dir = os.path.join(BASE_DIR, "models")
    os.makedirs(model_dir, exist_ok=True)

    save_path = os.path.join(model_dir, "brain_model.keras")
    model.save(save_path)
    model.save_weights(os.path.join(model_dir, "brain_weights.weights.h5"))

    print(f"✅ Model saved at: {save_path}")

except Exception as e:
    print("❌ Error saving model:", e)
