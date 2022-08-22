"""
Keras Deeplabv3+ model.
"""
from typing import Union
import matplotlib.pyplot as plt

from tensorflow import keras
from constants import IMAGE_SIZE, MODEL_DIR, NUM_CLASSES
from create_model import deeplab_v3_plus

from load_data import train_dataset, val_dataset

# build model

KerasTensor = Union[keras.layers.Layer, keras.layers.InputLayer]

# pylint: disable=too-many-arguments

model: keras.Model = deeplab_v3_plus(
    image_size=IMAGE_SIZE, num_classes=NUM_CLASSES)

loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss=loss,
    metrics=["accuracy"]
)

# training

history: keras.callbacks.History = model.fit(
    train_dataset, validation_data=val_dataset, epochs=25)

plt.plot(history.history["loss"])
plt.title("Training Loss")
plt.ylabel("loss")
plt.xlabel("epoch")
plt.show()

plt.plot(history.history["accuracy"])
plt.title("Training Accuracy")
plt.ylabel("accuracy")
plt.xlabel("epoch")
plt.show()

plt.plot(history.history["val_loss"])
plt.title("Validation Loss")
plt.ylabel("val_loss")
plt.xlabel("epoch")
plt.show()

plt.plot(history.history["val_accuracy"])
plt.title("Validation Accuracy")
plt.ylabel("val_accuracy")
plt.xlabel("epoch")
plt.show()

# saving the model

model.save_weights(MODEL_DIR)
