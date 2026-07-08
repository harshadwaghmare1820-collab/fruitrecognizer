"""
Classify a single fruit image using a trained model.

Usage:
    python predict.py --image path/to/fruit.jpg
"""

import argparse
import json
import os

import numpy as np
from tensorflow import keras

from model import IMG_SIZE


def load_class_names(models_dir: str):
    with open(os.path.join(models_dir, "class_names.json")) as f:
        return json.load(f)


def predict(image_path: str, models_dir: str = "models", top_k: int = 3):
    model_path = os.path.join(models_dir, "fruit_classifier.keras")
    model = keras.models.load_model(model_path)
    class_names = load_class_names(models_dir)

    img = keras.utils.load_img(image_path, target_size=IMG_SIZE)
    img_array = keras.utils.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)

    preds = model.predict(img_array, verbose=0)[0]
    top_indices = preds.argsort()[::-1][:top_k]

    print(f"\nPredictions for {image_path}:")
    for idx in top_indices:
        print(f"  {class_names[idx]:<15} {preds[idx] * 100:.2f}%")

    best_idx = top_indices[0]
    print(f"\n=> Best guess: {class_names[best_idx]} ({preds[best_idx] * 100:.2f}% confidence)")
    return class_names[best_idx], float(preds[best_idx])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Path to the fruit image")
    parser.add_argument("--models_dir", default="models")
    parser.add_argument("--top_k", type=int, default=3)
    args = parser.parse_args()

    predict(args.image, args.models_dir, args.top_k)
