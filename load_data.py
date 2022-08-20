"""
Keras Deeplabv3+ model.
"""
import os
from glob import glob
import tensorflow as tf
from constants import BATCH_SIZE, DATA_DIR, IMAGE_SIZE, NUM_TRAIN_IMAGES, NUM_VAL_IMAGES

# create dataset

train_images: list[str] = sorted(
    glob(os.path.join(DATA_DIR, "Images/*")))[:NUM_TRAIN_IMAGES]
train_masks: list[str] = sorted(
    glob(os.path.join(DATA_DIR, "Category_ids/*")))[:NUM_TRAIN_IMAGES]
val_images: list[str] = sorted(glob(os.path.join(DATA_DIR, "Images/*")))[
    NUM_TRAIN_IMAGES: NUM_VAL_IMAGES + NUM_TRAIN_IMAGES
]
val_masks: list[str] = sorted(glob(os.path.join(DATA_DIR, "Category_ids/*")))[
    NUM_TRAIN_IMAGES: NUM_VAL_IMAGES + NUM_TRAIN_IMAGES
]


def read_image(image_path: tf.Tensor, mask: bool = False) -> tf.Tensor:
    """
    Reads an image from a file path and returns a float32 tensor
    containing the image.

    Args:
        image_path: A string tensor containing the file path to the image.
        mask: A boolean indicating whether the image is a mask.
    Returns:
        A float32 tensor containing the image.
    """
    image: tf.Tensor = tf.io.read_file(image_path)
    if mask:
        image = tf.image.decode_png(image, channels=1)
        image.set_shape([None, None, 1])
        image = tf.image.resize(images=image, size=[IMAGE_SIZE, IMAGE_SIZE])
    else:
        image = tf.image.decode_png(image, channels=3)
        image.set_shape([None, None, 3])
        image = tf.image.resize(images=image, size=[IMAGE_SIZE, IMAGE_SIZE])
        image = image / 127.5 - 1
    return image


def load_data(image_list: tf.Tensor, mask_list: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
    """
    Loads an image and its mask from a list of file paths.
    Args:
        image_list: A list of file paths to the image.
        mask_list: A list of file paths to the mask.
    Returns:
        A tuple containing the image and mask.
    """
    image = read_image(image_list)
    mask = read_image(mask_list, mask=True)
    return image, mask


def data_generator(image_list: list[str], mask_list: list[str]) -> tf.data.Dataset:
    """
    Creates a tf.data.Dataset from a list of image and mask paths.
    Args:
        image_list: A list of file paths to the image.
        mask_list: A list of file paths to the mask.
    Returns:
        A tf.data.Dataset object.
    """
    dataset = tf.data.Dataset.from_tensor_slices((image_list, mask_list))
    dataset = dataset.map(load_data, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)
    return dataset


train_dataset: tf.data.Dataset = data_generator(train_images, train_masks)
val_dataset: tf.data.Dataset = data_generator(val_images, val_masks)
