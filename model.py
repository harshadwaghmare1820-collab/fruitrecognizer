"""
CNN model architectures for fruit classification.

Two options are provided:
  1. build_custom_cnn   -> a small CNN trained from scratch.
                           Simple and fast, but needs more data per class
                           (roughly 200+ images) to generalize well.
  2. build_mobilenet_tl -> transfer learning on top of MobileNetV2
                           pretrained on ImageNet. Recommended when you
                           only have a small dataset (tens to ~100 images
                           per class), since the backbone already knows
                           general visual features.
"""

from tensorflow import keras
from tensorflow.keras import layers


IMG_SIZE = (128, 128)


def build_custom_cnn(num_classes: int, img_size=IMG_SIZE) -> keras.Model:
    """A compact CNN built from scratch.

    Architecture: 4 conv blocks (Conv2D -> BatchNorm -> ReLU -> MaxPool),
    increasing filter counts, followed by GlobalAveragePooling and a
    dense classifier head with dropout for regularization.
    """
    inputs = keras.Input(shape=(*img_size, 3))

    x = layers.Rescaling(1.0 / 255)(inputs)

    for filters in (32, 64, 128, 256):
        x = layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.MaxPooling2D()(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = keras.Model(inputs, outputs, name="fruit_custom_cnn")
    return model


def build_mobilenet_tl(
    num_classes: int, img_size=IMG_SIZE, fine_tune_at: int | None = 100
) -> keras.Model:
    """MobileNetV2 transfer-learning model.

    The ImageNet-pretrained backbone is frozen except for the last
    `fine_tune_at`-th layer onward (if provided), which lets the network
    adapt higher-level features to fruits while keeping low-level filters
    (edges, textures, colors) intact -- this is what makes it work well
    even with a small dataset.
    """
    base_model = keras.applications.MobileNetV2(
        input_shape=(*img_size, 3), include_top=False, weights="imagenet"
    )
    base_model.trainable = False

    if fine_tune_at is not None:
        base_model.trainable = True
        for layer in base_model.layers[:fine_tune_at]:
            layer.trainable = False

    inputs = keras.Input(shape=(*img_size, 3))
    x = keras.applications.mobilenet_v2.preprocess_input(inputs)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = keras.Model(inputs, outputs, name="fruit_mobilenet_tl")
    return model


def build_model(arch: str, num_classes: int, img_size=IMG_SIZE) -> keras.Model:
    if arch == "custom":
        return build_custom_cnn(num_classes, img_size)
    elif arch == "mobilenet":
        return build_mobilenet_tl(num_classes, img_size)
    else:
        raise ValueError(f"Unknown arch '{arch}'. Choose 'custom' or 'mobilenet'.")
