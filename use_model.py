"""
Keras Deeplabv3+ model.
"""
from typing import Tuple, Union
import cv2
import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
from PIL import Image

import tensorflow as tf
from tensorflow import keras

from constants import COLORMAP_DIR, IMAGE_SIZE, MODEL_DIR, NUM_CLASSES
from create_model import deeplab_v3_plus
from load_data import read_image, train_images, val_images

# load the model

deeplabv3plus = deeplab_v3_plus(image_size=IMAGE_SIZE, num_classes=NUM_CLASSES)
deeplabv3plus.load_weights(MODEL_DIR)

# Loading the Colormap
human_colormap: np.ndarray = loadmat(COLORMAP_DIR)["colormap"]
human_colormap = human_colormap * 100
human_colormap = human_colormap.astype(np.uint8)

def infer(model: keras.Model, image_tensor: np.ndarray) -> np.ndarray:
    """
    Infer the segmentation mask for an image.

    Args:
        model: The model to use for inference.
        image_tensor: The image to infer the segmentation mask for.
    Returns:
        The segmentation mask for the image.
    """
    predictions: np.ndarray = model.predict(
        np.expand_dims(image_tensor, axis=0))
    predictions = np.squeeze(predictions)
    predictions = np.argmax(predictions, axis=2)
    return predictions


def decode_segmentation_masks(mask: np.ndarray, colormap: np.ndarray, n_classes: int) -> np.ndarray:
    """
    Decode the segmentation mask.
    Args:
        mask: The segmentation mask to decode.
        colormap: The colormap to use for decoding.
        n_classes: The number of classes in the dataset.
    Returns:
        The decoded segmentation mask.
    """
    red: np.ndarray = np.zeros_like(mask).astype(np.uint8)
    green: np.ndarray = np.zeros_like(mask).astype(np.uint8)
    blue: np.ndarray = np.zeros_like(mask).astype(np.uint8)
    for i in range(0, n_classes):
        idx = mask == i
        red[idx] = colormap[i, 0]
        green[idx] = colormap[i, 1]
        blue[idx] = colormap[i, 2]
    rgb = np.stack([red, green, blue], axis=2)
    return rgb


def get_overlay(image: tf.Tensor, colored_mask: np.ndarray) -> np.ndarray:
    """
    Get the overlay for an image.
    Args:
        image: The image to overlay.
        colored_mask: The colored mask to overlay.
    Returns:
        The overlayed image.
    """
    image: Image.Image = tf.keras.preprocessing.image.array_to_img(image)
    image: np.ndarray = np.array(image).astype(np.uint8)
    # pylint: disable=no-member
    overlay = cv2.addWeighted(image, 0.35, colored_mask, 0.65, 0)
    return overlay


def plot_samples_matplotlib(
    display_list: list[Union[tf.Tensor, np.ndarray]],
    figsize: Tuple[float, float] = (5, 3)
) -> None:
    """
    Plot the samples in a matplotlib figure.
    Args:
        display_list: The samples to plot.
        figsize: The size of the figure.
    Returns:
        None.
    """
    _, axes = plt.subplots(nrows=1, ncols=len(display_list), figsize=figsize)
    for i, display in enumerate(display_list):
        if display.shape[-1] == 3:
            axes[i].imshow(tf.keras.preprocessing.image.array_to_img(display))
        else:
            axes[i].imshow(display)
    plt.show()


def plot_predictions(images_list: list[str], colormap: np.ndarray, model: keras.Model) -> None:
    """
    Plot the predictions for a list of images.
    Args:
        images_list: The list of images to plot.
        colormap: The colormap to use for decoding.
        model: The model to use for inference.
    Returns:
        None.
    """
    for image_file in images_list:
        image_tensor: tf.Tensor = read_image(image_file)
        prediction_mask = infer(image_tensor=image_tensor, model=model)
        prediction_colormap = decode_segmentation_masks(
            prediction_mask, colormap, 20)
        overlay = get_overlay(image_tensor, prediction_colormap)
        plot_samples_matplotlib(
            [image_tensor, overlay, prediction_colormap], figsize=(18, 14)
        )

plot_predictions(train_images[:4], human_colormap, model=deeplabv3plus)
plot_predictions(val_images[:4], human_colormap, model=deeplabv3plus)
