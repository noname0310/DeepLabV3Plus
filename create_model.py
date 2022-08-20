"""
Keras Deeplabv3+ model.
"""
from typing import Union, Literal

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

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
        A Keras model. that compiled with the Adam optimizer.
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
