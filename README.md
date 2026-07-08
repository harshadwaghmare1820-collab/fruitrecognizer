# Fruit Image Classifier (CNN, small-dataset friendly)

Classifies fruit images (e.g., apple, banana, orange) using a Convolutional
Neural Network built with TensorFlow/Keras. Designed to work well even with
a small dataset by using:

- Heavy data augmentation (rotation, flip, zoom, brightness, etc.)
- Transfer learning option (MobileNetV2) for when you have very few images
- A lightweight custom CNN option for when you want to train from scratch
- Dropout + batch norm for regularization
- Early stopping to avoid overfitting on small data

## 1. Folder structure

Organize your images like this (one folder per class):

```
data/
  train/
    apple/
      img001.jpg
      img002.jpg
      ...
    banana/
      img001.jpg
      ...
    orange/
      img001.jpg
      ...
  val/
    apple/
    banana/
    orange/
```

You only need `train/` — the script will automatically carve out a
validation split from it if `val/` is empty or missing. 30–100 images per
class is enough to get started thanks to augmentation + transfer learning.

Good free sources for a small starter dataset:
- Fruits-360 dataset (Kaggle) — just take ~50 images per class you want.
- Your own phone photos (works great, and it's fun to test the model on
  your own kitchen fruit afterwards).

## 2. Install dependencies

```bash
pip install -r requirements.txt
```

## 3. Train

Custom lightweight CNN (good if you have 200+ images per class):
```bash
python train.py --data_dir data --arch custom --epochs 30
```

Transfer learning with MobileNetV2 (recommended for small datasets,
e.g. under 200 images per class):
```bash
python train.py --data_dir data --arch mobilenet --epochs 15
```

This will:
1. Load and augment your images
2. Train the model
3. Save the best model to `models/fruit_classifier.keras`
4. Save the class names to `models/class_names.json`
5. Plot accuracy/loss curves to `outputs/training_curves.png`
6. Print a classification report + confusion matrix on the validation set

## 4. Predict on a new image

```bash
python predict.py --image path/to/some_fruit.jpg
```

## 5. Files

| File | Purpose |
|---|---|
| `model.py` | CNN architecture definitions (custom CNN + MobileNetV2 transfer learning) |
| `train.py` | Data loading, augmentation, training loop, evaluation, plots |
| `predict.py` | Load a trained model and classify a single image |
| `requirements.txt` | Python dependencies |

## Tips for small datasets

- **Use transfer learning** (`--arch mobilenet`). A model pretrained on
  ImageNet already knows general shapes/textures/colors, so it needs far
  fewer fruit images to specialize.
- **Augment aggressively** — already configured in `train.py` (rotation,
  zoom, shear, flips, brightness jitter).
- **Keep classes balanced** — try to have a similar number of images per
  fruit class.
- **Watch validation loss** — `train.py` uses early stopping so it won't
  keep training once validation loss stops improving.
