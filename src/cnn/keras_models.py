from __future__ import annotations

import tensorflow as tf

from .experiments import CNNExperimentConfig


@tf.keras.utils.register_keras_serializable(package="SpreiElsa")
class SimpleLocallyConnected2D(tf.keras.layers.Layer):
    def __init__(self, filters: int, kernel_size: int | tuple[int, int], strides: int | tuple[int, int] = 1, activation: str | None = None, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.strides = strides if isinstance(strides, tuple) else (strides, strides)
        self.activation = tf.keras.activations.get(activation)
        self.activation_name = activation

    def build(self, input_shape):
        _, height, width, channels = input_shape
        kh, kw = self.kernel_size
        sh, sw = self.strides
        self.out_h = (int(height) - kh) // sh + 1
        self.out_w = (int(width) - kw) // sw + 1
        patch_dim = kh * kw * int(channels)
        positions = self.out_h * self.out_w
        self.kernel = self.add_weight(
            name="kernel",
            shape=(positions, patch_dim, self.filters),
            initializer="glorot_uniform",
            trainable=True,
        )
        self.bias = self.add_weight(
            name="bias",
            shape=(positions, self.filters),
            initializer="zeros",
            trainable=True,
        )

    def call(self, inputs):
        patches = tf.image.extract_patches(
            inputs,
            sizes=[1, self.kernel_size[0], self.kernel_size[1], 1],
            strides=[1, self.strides[0], self.strides[1], 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        batch = tf.shape(inputs)[0]
        patches = tf.reshape(patches, (batch, self.out_h * self.out_w, -1))
        output = tf.einsum("bpd,pdf->bpf", patches, self.kernel) + self.bias
        output = tf.reshape(output, (batch, self.out_h, self.out_w, self.filters))
        if self.activation is not None:
            output = self.activation(output)
        return output

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "filters": self.filters,
                "kernel_size": self.kernel_size,
                "strides": self.strides,
                "activation": self.activation_name,
            }
        )
        return config


def _pooling_layer(pooling: str, index: int):
    if pooling == "max":
        return tf.keras.layers.MaxPooling2D(pool_size=(2, 2), name=f"pool_{index}")
    if pooling == "avg":
        return tf.keras.layers.AveragePooling2D(pool_size=(2, 2), name=f"pool_{index}")
    raise ValueError(f"Unsupported pooling: {pooling}")


def build_shared_cnn(config: CNNExperimentConfig) -> tf.keras.Model:
    inputs = tf.keras.Input(shape=(config.image_size, config.image_size, 3), name="image")
    x = inputs
    for index, filters in enumerate(config.filters):
        x = tf.keras.layers.Conv2D(
            filters,
            (config.kernel_size, config.kernel_size),
            padding="valid",
            activation="relu",
            name=f"conv_{index + 1}",
        )(x)
        x = _pooling_layer(config.pooling, index + 1)(x)
    x = tf.keras.layers.Flatten(name="flatten")(x)
    x = tf.keras.layers.Dense(config.dense_units, activation="relu", name="dense_hidden")(x)
    outputs = tf.keras.layers.Dense(config.num_classes, activation="softmax", name="classifier")(x)
    model = tf.keras.Model(inputs, outputs, name=f"shared_{config.experiment_id}")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=config.learning_rate),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )
    return model


def build_non_shared_cnn(config: CNNExperimentConfig) -> tf.keras.Model:
    inputs = tf.keras.Input(shape=(config.image_size, config.image_size, 3), name="image")
    x = inputs
    layer_cls = getattr(tf.keras.layers, "LocallyConnected2D", SimpleLocallyConnected2D)
    for index, filters in enumerate(config.filters):
        x = layer_cls(
            filters,
            (config.kernel_size, config.kernel_size),
            activation="relu",
            name=f"local_{index + 1}",
        )(x)
        x = _pooling_layer(config.pooling, index + 1)(x)
    x = tf.keras.layers.Flatten(name="flatten")(x)
    x = tf.keras.layers.Dense(config.dense_units, activation="relu", name="dense_hidden")(x)
    outputs = tf.keras.layers.Dense(config.num_classes, activation="softmax", name="classifier")(x)
    model = tf.keras.Model(inputs, outputs, name=f"non_shared_{config.experiment_id}")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=config.learning_rate),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )
    return model

