"""Convert supported Keras models to scratch forward models."""

from __future__ import annotations

from .layers import (
    AveragePooling2D,
    Conv2D,
    Dense,
    Flatten,
    GlobalAveragePooling2D,
    GlobalMaxPooling2D,
    LocallyConnected2D,
    MaxPooling2D,
)
from .model import SequentialScratchModel


def keras_to_scratch_model(keras_model) -> SequentialScratchModel:
    """Convert supported Keras layers to a NumPy forward model.

    Supported layers: Conv2D, MaxPooling2D, AveragePooling2D,
    GlobalAveragePooling2D, GlobalMaxPooling2D, Flatten, Dense.
    """
    scratch_layers = []
    for layer in keras_model.layers:
        class_name = layer.__class__.__name__
        if class_name == "InputLayer":
            continue
        if class_name == "Conv2D":
            kernel, bias = layer.get_weights()
            scratch_layers.append(
                Conv2D(
                    kernel=kernel,
                    bias=bias,
                    strides=tuple(layer.strides),
                    padding=layer.padding,
                    activation=layer.activation.__name__,
                )
            )
        elif class_name == "MaxPooling2D":
            scratch_layers.append(MaxPooling2D(tuple(layer.pool_size), tuple(layer.strides), layer.padding))
        elif class_name == "AveragePooling2D":
            scratch_layers.append(AveragePooling2D(tuple(layer.pool_size), tuple(layer.strides), layer.padding))
        elif class_name == "GlobalAveragePooling2D":
            scratch_layers.append(GlobalAveragePooling2D())
        elif class_name == "GlobalMaxPooling2D":
            scratch_layers.append(GlobalMaxPooling2D())
        elif class_name == "Flatten":
            scratch_layers.append(Flatten())
        elif class_name == "Dense":
            kernel, bias = layer.get_weights()
            scratch_layers.append(Dense(kernel=kernel, bias=bias, activation=layer.activation.__name__))
        elif class_name in {"LocallyConnected2D", "SimpleLocallyConnected2D"}:
            kernel, bias = layer.get_weights()
            activation = layer.activation.__name__ if layer.activation is not None else None
            scratch_layers.append(
                LocallyConnected2D(
                    kernel=kernel,
                    bias=bias,
                    kernel_size=tuple(layer.kernel_size),
                    strides=tuple(layer.strides),
                    padding=getattr(layer, "padding", "valid"),
                    activation=activation,
                )
            )
        elif class_name == "Dropout":
            continue
        else:
            raise ValueError(f"Unsupported Keras layer for scratch conversion: {class_name}")
    return SequentialScratchModel(scratch_layers)

