"""
Keras Deeplabv3+ model.
"""
import os
from typing import Tuple, Union, Literal
from glob import glob
import cv2
import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
from PIL import Image

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Tensorflow type check fix code
# pylint: disable=unknown-option-value

# # Explicitly import lazy-loaded modules to support autocompletion.
# # pylint: disable=g-import-not-at-top
# if _typing.TYPE_CHECKING:
#   from tensorflow_estimator.python.estimator.api._v2 import estimator as estimator
#   from keras.api._v2 import keras
#   from keras.api._v2.keras import losses
#   from keras.api._v2.keras import metrics
#   from keras.api._v2.keras import optimizers
#   from keras.api._v2.keras import initializers
# # pylint: enable=g-import-not-at-top

# pylint: enable=unknown-option-value

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

IMAGE_SIZE = 512 // 2
BATCH_SIZE = 4
NUM_CLASSES = 20
DATA_DIR = "./instance-level_human_parsing/instance-level_human_parsing/Training"
NUM_TRAIN_IMAGES = 1000
NUM_VAL_IMAGES = 50

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

# print("Train Dataset:", train_dataset)
# print("Val Dataset:", val_dataset)

# build model

KerasTensor = Union[keras.layers.Layer, keras.layers.InputLayer]

# pylint: disable=too-many-arguments


def convolution_block(
    block_input: KerasTensor,
    num_filters: int = 256,
    kernel_size: int = 3,
    dilation_rate: int = 1,
    padding: Literal["valid", "same"] = "same",
    use_bias: bool = False,
) -> KerasTensor:
    """
    Creates a convolution block with batch normalization and relu activation.
    Args:
        block_input: The input to the convolution block.
        num_filters: The number of filters in the convolution block.
        kernel_size: The size of the convolution kernel.
        dilation_rate: The dilation rate of the convolution kernel.
        padding: The padding type of the convolution kernel.
        use_bias: Whether to use a bias in the convolution block.
    Returns:
        The output of the convolution block.
    """
    tensor: KerasTensor = layers.Conv2D(
        num_filters,
        kernel_size=kernel_size,
        dilation_rate=dilation_rate,
        padding=padding,
        use_bias=use_bias,
        kernel_initializer=keras.initializers.HeNormal(),
    )(block_input)
    tensor = layers.BatchNormalization()(tensor)
    return tf.nn.relu(tensor)


def dilated_spatial_pyramid_pooling(dspp_input: KerasTensor) -> KerasTensor:
    """
    Creates a Dilated Spatial Pyramid Pooling layer.
    Args:
        dspp_input: The input to the Dilated Spatial Pyramid Pooling layer.
    Returns:
        The output of the Dilated Spatial Pyramid Pooling layer.
    """
    dims: tf.TensorShape = dspp_input.shape
    tensor: KerasTensor = layers.AveragePooling2D(
        pool_size=(dims[-3], dims[-2]))(dspp_input)
    tensor = convolution_block(tensor, kernel_size=1, use_bias=True)
    out_pool: KerasTensor = layers.UpSampling2D(
        size=(dims[-3] // tensor.shape[1], dims[-2] // tensor.shape[2]), interpolation="bilinear",
    )(tensor)

    out_1 = convolution_block(dspp_input, kernel_size=1, dilation_rate=1)
    out_6 = convolution_block(dspp_input, kernel_size=3, dilation_rate=6)
    out_12 = convolution_block(dspp_input, kernel_size=3, dilation_rate=12)
    out_18 = convolution_block(dspp_input, kernel_size=3, dilation_rate=18)

    tensor = layers.Concatenate(
        axis=-1)([out_pool, out_1, out_6, out_12, out_18])
    output = convolution_block(tensor, kernel_size=1)
    return output


def deeplab_v3_plus(image_size: int, num_classes: int) -> keras.Model:
    """
    Creates a DeeplabV3Plus model.
    Args:
        image_size: The size of the input images.
        num_classes: The number of classes in the dataset.
    Returns:
        A Keras model.
    """
    model_input: KerasTensor = keras.Input(shape=(image_size, image_size, 3))
    resnet50 = keras.applications.ResNet50(
        weights="imagenet", include_top=False, input_tensor=model_input
    )
    tensor: KerasTensor = resnet50.get_layer("conv4_block6_2_relu").output
    tensor = dilated_spatial_pyramid_pooling(tensor)

    input_a: KerasTensor = layers.UpSampling2D(
        size=(image_size // 4 // tensor.shape[1],
              image_size // 4 // tensor.shape[2]),
        interpolation="bilinear",
    )(tensor)
    input_b: KerasTensor = resnet50.get_layer("conv2_block3_2_relu").output
    input_b = convolution_block(input_b, num_filters=48, kernel_size=1)

    tensor = layers.Concatenate(axis=-1)([input_a, input_b])
    tensor = convolution_block(tensor)
    tensor = convolution_block(tensor)
    tensor = layers.UpSampling2D(
        size=(image_size // tensor.shape[1], image_size // tensor.shape[2]),
        interpolation="bilinear",
    )(tensor)
    model_output: KerasTensor = layers.Conv2D(
        num_classes, kernel_size=(1, 1), padding="same")(tensor)
    return keras.Model(inputs=model_input, outputs=model_output)


model: keras.Model = deeplab_v3_plus(
    image_size=IMAGE_SIZE, num_classes=NUM_CLASSES)
#model.summary()

# training

loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss=loss,
    metrics=["accuracy"],
)

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

# Loading the Colormap
colormap: np.ndarray = loadmat(
    "./instance-level_human_parsing/instance-level_human_parsing/human_colormap.mat"
)["colormap"]
colormap = colormap * 100
colormap = colormap.astype(np.uint8)

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
    overlay = cv2.addWeighted(image, 0.35, colored_mask, 0.65, 0)
    return overlay


def plot_samples_matplotlib(display_list: list[Union[tf.Tensor, np.ndarray]], figsize: Tuple[float, float] = (5, 3)) -> None:
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

plot_predictions(train_images[:4], colormap, model=model)
plot_predictions(val_images[:4], colormap, model=model)
