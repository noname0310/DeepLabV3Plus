"""
Save model to tensorflowjs format.
"""
from typing import Union

from tensorflow import keras
from constants import IMAGE_SIZE, NUM_CLASSES, MODEL_DIR
from create_model import deeplab_v3_plus

import tensorflowjs as tfjs

KerasTensor = Union[keras.layers.Layer, keras.layers.InputLayer]

model: keras.Model = deeplab_v3_plus(
    image_size=IMAGE_SIZE, num_classes=NUM_CLASSES)

model.load_weights(MODEL_DIR)

tfjs.converters.save_keras_model(model, "../webapp/public/tfjs-model")
