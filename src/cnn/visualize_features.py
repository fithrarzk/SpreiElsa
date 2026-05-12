"""Visualize CNN feature maps and Grad-CAM for a single image."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt

from src.utils.image_utils import load_image


def _find_last_conv_layer(model: tf.keras.Model) -> tf.keras.layers.Layer:
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer
    raise ValueError("No Conv2D layer found in model")


def _save_feature_map_grid(feature_maps: np.ndarray, output_path: Path, max_channels: int = 32) -> None:
    if feature_maps.ndim != 3:
        raise ValueError("feature_maps must have shape (H, W, C)")
    h, w, channels = feature_maps.shape
    channels = min(channels, max_channels)

    cols = int(np.ceil(np.sqrt(channels)))
    rows = int(np.ceil(channels / cols))
    plt.figure(figsize=(cols * 2, rows * 2))
    for idx in range(channels):
        ax = plt.subplot(rows, cols, idx + 1)
        fmap = feature_maps[:, :, idx]
        fmap = (fmap - fmap.min()) / (fmap.max() - fmap.min() + 1e-8)
        ax.imshow(fmap, cmap="viridis")
        ax.axis("off")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.close()


def _grad_cam_heatmap(model: tf.keras.Model, image_tensor: tf.Tensor, conv_layer: tf.keras.layers.Layer, class_index: int | None) -> np.ndarray:
    grad_model = tf.keras.Model(model.inputs, [conv_layer.output, model.output])
    with tf.GradientTape() as tape:
        conv_out, preds = grad_model(image_tensor)
        if class_index is None:
            class_index = int(tf.argmax(preds[0]))
        loss = preds[:, class_index]
    grads = tape.gradient(loss, conv_out)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_out = conv_out[0]
    heatmap = tf.reduce_sum(conv_out * pooled_grads, axis=-1)
    heatmap = tf.maximum(heatmap, 0)
    heatmap = heatmap / (tf.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()


def _overlay_heatmap(image: np.ndarray, heatmap: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    heatmap_resized = np.array(Image.fromarray(np.uint8(255 * heatmap)).resize((image.shape[1], image.shape[0])))
    heatmap_color = plt.get_cmap("jet")(heatmap_resized / 255.0)[:, :, :3]
    overlay = heatmap_color * alpha + (image / 255.0) * (1 - alpha)
    return np.uint8(255 * overlay)


def visualize(
    model_path: Path,
    image_path: Path,
    output_dir: Path,
    image_size: int,
    layer_name: str | None,
    class_index: int | None,
    max_channels: int,
    alpha: float,
) -> None:
    model = tf.keras.models.load_model(model_path)
    layer = model.get_layer(layer_name) if layer_name else _find_last_conv_layer(model)

    image = load_image(image_path, image_size=(image_size, image_size), normalize=False)
    image_tensor = tf.convert_to_tensor(image[np.newaxis] / 255.0, dtype=tf.float32)

    feature_model = tf.keras.Model(model.inputs, layer.output)
    feature_maps = feature_model.predict(image_tensor, verbose=0)[0]
    output_dir.mkdir(parents=True, exist_ok=True)
    _save_feature_map_grid(feature_maps, output_dir / "feature_maps.png", max_channels=max_channels)

    heatmap = _grad_cam_heatmap(model, image_tensor, layer, class_index)
    overlay = _overlay_heatmap(image, heatmap, alpha=alpha)
    Image.fromarray(np.uint8(255 * heatmap)).save(output_dir / "grad_cam_heatmap.png")
    Image.fromarray(overlay).save(output_dir / "grad_cam_overlay.png")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--image-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/cnn/visuals"))
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--layer-name")
    parser.add_argument("--class-index", type=int)
    parser.add_argument("--max-channels", type=int, default=32)
    parser.add_argument("--alpha", type=float, default=0.4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    visualize(
        model_path=args.model_path,
        image_path=args.image_path,
        output_dir=args.output_dir,
        image_size=args.image_size,
        layer_name=args.layer_name,
        class_index=args.class_index,
        max_channels=args.max_channels,
        alpha=args.alpha,
    )


if __name__ == "__main__":
    main()
