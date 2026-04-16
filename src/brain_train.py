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

# =========================
# PATH
# =========================
BASE_DIR = r"C:\Users\91701\capstone"
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
model.fit(
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

# =========================
# SAVE
# =========================
model.save(os.path.join(BASE_DIR, "models", "brain_model.keras"))
print("✅ Brain model saved")