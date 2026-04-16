import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import numpy as np
import os
from sklearn.metrics import classification_report, confusion_matrix

# =========================
# CONFIG
# =========================
IMG_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 5

# =========================
# PATHS (FIX IF NEEDED)
# =========================
BASE_DIR = r"C:\Users\91701\capstone"

TRAIN_DIR = os.path.join(BASE_DIR, "knee_dataset", "train")
VAL_DIR   = os.path.join(BASE_DIR, "knee_dataset", "val")

# =========================
# DATA GENERATORS
# =========================
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    horizontal_flip=True,
    brightness_range=[0.8, 1.2],
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=True
)

val_generator = val_datagen.flow_from_directory(
    VAL_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False  # IMPORTANT
)

class_names = list(train_generator.class_indices.keys())
NUM_CLASSES = len(class_names)

print("Classes:", class_names)

# =========================
# MODEL
# =========================
base_model = DenseNet121(
    weights='imagenet',
    include_top=False,
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)

base_model.trainable = False

x = GlobalAveragePooling2D()(base_model.output)
x = Dense(512, activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.3)(x)
output = Dense(NUM_CLASSES, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=output)

# =========================
# METRICS (FULL SET)
# =========================
metrics = [
    "accuracy",
    tf.keras.metrics.Precision(name="precision"),
    tf.keras.metrics.Recall(name="recall"),
    tf.keras.metrics.AUC(name="auc")
]

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=metrics
)

# =========================
# CALLBACKS
# =========================
callbacks = [
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
    ModelCheckpoint(
        os.path.join(BASE_DIR, "models", "knee_oa_best.h5"),
        monitor='val_accuracy',
        save_best_only=True
    ),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)
]

# =========================
# TRAIN (PHASE 1)
# =========================
model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=val_generator,
    callbacks=callbacks
)

# =========================
# FINE-TUNING
# =========================
base_model.trainable = True

for layer in base_model.layers[:100]:
    layer.trainable = False

model.compile(
    optimizer=Adam(learning_rate=1e-5),
    loss='categorical_crossentropy',
    metrics=metrics
)

model.fit(
    train_generator,
    epochs=10,
    validation_data=val_generator,
    callbacks=callbacks
)

# =========================
# EVALUATION (VERY IMPORTANT)
# =========================
print("\n📊 Evaluating model...")

val_generator.reset()

preds = model.predict(val_generator)
y_pred = np.argmax(preds, axis=1)
y_true = val_generator.classes

print("\n📌 Classification Report:")
print(classification_report(y_true, y_pred, target_names=class_names))

print("\n📌 Confusion Matrix:")
print(confusion_matrix(y_true, y_pred))

# =========================
# SAVE FINAL MODEL
# =========================
final_path = os.path.join(BASE_DIR, "models", "knee_oa_final.h5")
model.save(final_path)

print("\n✅ Model saved at:", final_path)