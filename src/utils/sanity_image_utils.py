from __future__ import annotations

import argparse

from src.utils.image_utils import load_classification_split


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="src/dataset/train")
    parser.add_argument("--image-size", type=int, default=64)
    args = parser.parse_args()

    x, y, classes = load_classification_split(
        args.split,
        image_size=(args.image_size, args.image_size),
    )
    print(x.shape, y.shape, classes)


if __name__ == "__main__":
    main()

