"""
Save model to tensorflowjs format.
"""
from typing import Union

from tensorflow import keras
from constants import IMAGE_SIZE, NUM_CLASSES, MODEL_WEIGHT_DIR, MODEL_DIR
from create_model import deeplab_v3_plus

import tensorflowjs as tfjs

KerasTensor = Union[keras.layers.Layer, keras.layers.InputLayer]

model: keras.Model = deeplab_v3_plus(
    image_size=IMAGE_SIZE, num_classes=NUM_CLASSES)

loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss=loss,
    metrics=["accuracy"]
)

model.load_weights(MODEL_WEIGHT_DIR)

model.save(MODEL_DIR)

tfjs.converters.save_keras_model(model, "../webapp/public/tfjs-model")
