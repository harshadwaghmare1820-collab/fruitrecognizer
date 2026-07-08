"""
Train a fruit image classifier CNN.

Usage:
    python train.py --data_dir data --arch mobilenet --epochs 15
    python train.py --data_dir data --arch custom --epochs 30

Expects:
    data/train/<class_name>/*.jpg
    data/val/<class_name>/*.jpg      (optional; auto-split from train if absent)
"""

import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow import keras
from tensorflow.keras import layers

from model import IMG_SIZE, build_model


def get_datasets(data_dir: str, img_size, batch_size: int, val_split: float = 0.2):
    train_dir = os.path.join(data_dir, "train")
    val_dir = os.path.join(data_dir, "val")

    has_separate_val = os.path.isdir(val_dir) and any(os.scandir(val_dir))

    if has_separate_val:
        train_ds = keras.utils.image_dataset_from_directory(
            train_dir, image_size=img_size, batch_size=batch_size, seed=42
        )
        val_ds = keras.utils.image_dataset_from_directory(
            val_dir, image_size=img_size, batch_size=batch_size, seed=42
        )
    else:
        train_ds = keras.utils.image_dataset_from_directory(
            train_dir,
            image_size=img_size,
            batch_size=batch_size,
            validation_split=val_split,
            subset="training",
            seed=42,
        )
        val_ds = keras.utils.image_dataset_from_directory(
            train_dir,
            image_size=img_size,
            batch_size=batch_size,
            validation_split=val_split,
            subset="validation",
            seed=42,
        )

    class_names = train_ds.class_names
    return train_ds, val_ds, class_names


def build_augmentation():
    """Data augmentation pipeline -- crucial for making a small dataset
    behave like a bigger, more varied one."""
    return keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.15),
            layers.RandomZoom(0.15),
            layers.RandomTranslation(0.1, 0.1),
            layers.RandomContrast(0.15),
        ],
        name="augmentation",
    )


def prepare(ds, augment: bool, augmenter=None):
    ds = ds.cache()
    if augment and augmenter is not None:
        ds = ds.map(
            lambda x, y: (augmenter(x, training=True), y),
            num_parallel_calls=tf.data.AUTOTUNE,
        )
    return ds.prefetch(tf.data.AUTOTUNE)


def plot_history(history, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    axes[0].plot(history.history["accuracy"], label="train")
    axes[0].plot(history.history["val_accuracy"], label="val")
    axes[0].set_title("Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(history.history["loss"], label="train")
    axes[1].plot(history.history["val_loss"], label="val")
    axes[1].set_title("Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Saved training curves to {out_path}")


def evaluate(model, val_ds, class_names):
    y_true, y_pred = [], []
    for images, labels in val_ds:
        preds = model.predict(images, verbose=0)
        y_true.extend(labels.numpy())
        y_pred.extend(np.argmax(preds, axis=1))

    print("\nClassification report:")
    print(classification_report(y_true, y_pred, target_names=class_names))

    print("Confusion matrix (rows=true, cols=predicted):")
    print(confusion_matrix(y_true, y_pred))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--arch", choices=["custom", "mobilenet"], default="mobilenet")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--models_dir", default="models")
    parser.add_argument("--outputs_dir", default="outputs")
    args = parser.parse_args()

    os.makedirs(args.models_dir, exist_ok=True)
    os.makedirs(args.outputs_dir, exist_ok=True)

    print("Loading datasets...")
    train_ds, val_ds, class_names = get_datasets(
        args.data_dir, IMG_SIZE, args.batch_size
    )
    print(f"Classes found: {class_names}")

    augmenter = build_augmentation()
    train_ds = prepare(train_ds, augment=True, augmenter=augmenter)
    val_ds = prepare(val_ds, augment=False)

    print(f"Building model (arch={args.arch})...")
    model = build_model(args.arch, num_classes=len(class_names))

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=args.lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()

    model_path = os.path.join(args.models_dir, "fruit_classifier.keras")
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            model_path, save_best_only=True, monitor="val_accuracy", mode="max"
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=5, restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6
        ),
    ]

    print("Training...")
    history = model.fit(
        train_ds, validation_data=val_ds, epochs=args.epochs, callbacks=callbacks
    )

    # Save class names alongside the model so predict.py can map indices -> labels
    with open(os.path.join(args.models_dir, "class_names.json"), "w") as f:
        json.dump(class_names, f)

    plot_history(history, os.path.join(args.outputs_dir, "training_curves.png"))
    evaluate(model, val_ds, class_names)

    print(f"\nDone. Best model saved to: {model_path}")


if __name__ == "__main__":
    main()
