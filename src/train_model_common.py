import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model

train_gen =  ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    zoom_range=0.1,
    horizontal_flip=True
)
val_gen = ImageDataGenerator(rescale=1./255)

train_data = train_gen.flow_from_directory(
    "dataset/train",
    target_size=(224,224),
    class_mode="categorical"
)

val_data = val_gen.flow_from_directory(
    "dataset/val",
    target_size=(224,224),
    class_mode="categorical"
)
base = DenseNet121(
    weights="imagenet",
    include_top=False,
    input_shape=(224,224,3)
)

base.trainable = False

x = GlobalAveragePooling2D()(base.output)
x = Dense(128, activation="relu")(x)
out = Dense(3, activation="softmax")(x)

model = Model(base.input, out)

model.compile(optimizer="adam",
              loss="categorical_crossentropy",
              metrics=["accuracy"])

model.fit(train_data, validation_data=val_data, epochs=15)

model.save("models/severity_model.h5")
