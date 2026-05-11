import os
import numpy as np
from image_utils import load_image, load_batch  # dari Timur


def build_inception_encoder():
    from tensorflow.keras.applications import InceptionV3
    from tensorflow.keras.applications.inception_v3 import preprocess_input
    base = InceptionV3(weights="imagenet", include_top=False, pooling="avg")
    for layer in base.layers:
        layer.trainable = False
    return base, preprocess_input, (299, 299)


def build_vgg16_encoder():
    from tensorflow.keras.applications import VGG16
    from tensorflow.keras.applications.vgg16 import preprocess_input
    base = VGG16(weights="imagenet", include_top=False, pooling="avg")
    for layer in base.layers:
        layer.trainable = False
    return base, preprocess_input, (224, 224)


def extract_features(image_paths, encoder, preprocess_fn, target_size=(299, 299), batch_size=32):
    features = {}
    n = len(image_paths)
    for start in range(0, n, batch_size):
        batch_paths = image_paths[start:start + batch_size]
        batch = (load_batch(batch_paths, target_size) * 255.0).astype(np.float32)
        batch_feats = encoder.predict(preprocess_fn(batch), verbose=0)
        for path, feat in zip(batch_paths, batch_feats):
            features[os.path.basename(path)] = feat.astype(np.float32)
        print(f"  [{min(start + batch_size, n)}/{n}]", end="\r")
    print()
    return features


def save_features(features, path):
    np.save(path, features)


def load_features(path):
    return np.load(path, allow_pickle=True).item()


def extract_flickr8k_features(images_dir, output_path, encoder_name="inceptionv3", batch_size=32):
    if encoder_name == "vgg16":
        encoder, preprocess_fn, target_size = build_vgg16_encoder()
    else:
        encoder, preprocess_fn, target_size = build_inception_encoder()

    image_files = sorted(
        os.path.join(images_dir, f)
        for f in os.listdir(images_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    )
    features = extract_features(image_files, encoder, preprocess_fn, target_size, batch_size)
    save_features(features, output_path)
    return features
