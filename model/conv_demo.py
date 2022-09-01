"""
convolutional layer demo
"""

import os
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

from constants import DATA_DIR

def plot_images(images: list[tf.Tensor]) -> None:
    """
    Plot the images.
    Args:
        images: The images to plot.
    Returns:
        None.
    """
    for i, image in enumerate(images):
        plt.subplot(1, len(images), i + 1)
        plt.imshow(keras.preprocessing.image.array_to_img(image))
    plt.show()

image_tensor: tf.Tensor = tf.io.read_file(
    os.path.join(DATA_DIR, "Images", "NewLevelSequence.FinalImage.0002.png"))

image_tensor: tf.Tensor = tf.image.decode_png(image_tensor, channels=3)
image_tensor = tf.image.convert_image_dtype(image_tensor, tf.float32)
image_tensor = image_tensor / 255.0

image_conv_1: tf.Tensor = keras.layers.Conv2D(
    filters=1,
    kernel_size=3,
    dilation_rate=1,
    padding="same",
    use_bias=False,
    kernel_initializer= keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=1)
)(tf.expand_dims(image_tensor, axis=0))
image_conv_1 = tf.squeeze(image_conv_1, axis=0)

image_conv_6: tf.Tensor = keras.layers.Conv2D(
    filters=1,
    kernel_size=3,
    dilation_rate=6,
    padding="same",
    use_bias=False,
    kernel_initializer= keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=1)
)(tf.expand_dims(image_tensor, axis=0))
image_conv_6 = tf.squeeze(image_conv_6, axis=0)

image_conv_12: tf.Tensor = keras.layers.Conv2D(
    filters=1,
    kernel_size=3,
    dilation_rate=12,
    padding="same",
    use_bias=False,
    kernel_initializer= keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=1)
)(tf.expand_dims(image_tensor, axis=0))
image_conv_12 = tf.squeeze(image_conv_12, axis=0)

image_conv_18: tf.Tensor = keras.layers.Conv2D(
    filters=1,
    kernel_size=3,
    dilation_rate=18,
    padding="same",
    use_bias=False,
    kernel_initializer= keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=1)
)(tf.expand_dims(image_tensor, axis=0))
image_conv_18 = tf.squeeze(image_conv_18, axis=0)

image_conv_concat: tf.Tensor = keras.layers.Concatenate(axis=-1)([
    image_conv_1, image_conv_6, image_conv_12, image_conv_18
])

image_conv_1_k_100: tf.Tensor = keras.layers.Conv2D(
    filters=1,
    kernel_size=30,
    dilation_rate=1,
    padding="same",
    use_bias=False,
    kernel_initializer= keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=1)
)(tf.expand_dims(image_tensor, axis=0))
image_conv_1_k_100 = tf.squeeze(image_conv_1_k_100, axis=0)

plot_images([
    image_tensor, image_conv_1_k_100, image_conv_18, image_conv_concat])
