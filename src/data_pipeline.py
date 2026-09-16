"""
Loads the chest X-ray dataset into tf.data.Dataset pipelines.

Key decisions (see README "Concepts" section for the why):
- Augmentation is applied to the TRAINING set only.
- DenseNet's own preprocess_input is used for pixel scaling, not plain /255.
- Class weights are computed from the train folder's image counts because
  this dataset is imbalanced (~3x more PNEUMONIA than NORMAL).
"""
import os
import tensorflow as tf
from tensorflow.keras.applications.densenet import preprocess_input

from src import config


def _count_images_per_class(directory):
    """Count images in each class subfolder. Returns {class_name: count}."""
    counts = {}
    for class_name in config.CLASS_NAMES:
        class_dir = os.path.join(directory, class_name)
        counts[class_name] = len(
            [f for f in os.listdir(class_dir) if not f.startswith(".")]
        )
    return counts


def compute_class_weights():
    """
    Compute class weights inversely proportional to class frequency, so the
    loss penalizes mistakes on the minority class (NORMAL) more heavily.
    Standard formula: weight_i = n_samples / (n_classes * n_samples_i)
    """
    counts = _count_images_per_class(config.TRAIN_DIR)
    total = sum(counts.values())
    n_classes = len(counts)
    weights = {
        i: total / (n_classes * counts[class_name])
        for i, class_name in enumerate(config.CLASS_NAMES)
    }
    print(f"Class counts: {counts}")
    print(f"Class weights: {weights}")
    return weights


def _augment(image, label):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, max_delta=0.1)  # proxy for contrast/zoom jitter
    image = tf.image.random_contrast(image, lower=0.9, upper=1.1)
    # Small random zoom via central crop + resize back
    image = tf.image.central_crop(image, central_fraction=tf.random.uniform([], 0.85, 1.0))
    image = tf.image.resize(image, config.IMG_SIZE)
    # Small random rotation isn't natively in tf.image; keras layer used in build_datasets instead.
    return image, label


def _preprocess(image, label):
    image = preprocess_input(image)
    return image, label


def build_datasets():
    """
    Returns (train_ds, val_ds, test_ds), all batched, prefetched, and
    preprocessed with DenseNet's preprocess_input. Only train_ds is augmented.
    """
    train_ds = tf.keras.utils.image_dataset_from_directory(
        config.TRAIN_DIR,
        labels="inferred",
        label_mode="binary",
        class_names=config.CLASS_NAMES,
        image_size=config.IMG_SIZE,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        seed=config.SEED,
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        config.VAL_DIR,
        labels="inferred",
        label_mode="binary",
        class_names=config.CLASS_NAMES,
        image_size=config.IMG_SIZE,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
    )
    test_ds = tf.keras.utils.image_dataset_from_directory(
        config.TEST_DIR,
        labels="inferred",
        label_mode="binary",
        class_names=config.CLASS_NAMES,
        image_size=config.IMG_SIZE,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
    )

    # Random rotation as a Keras layer (applied only within the train pipeline)
    rotation_layer = tf.keras.layers.RandomRotation(0.05)  # ~±18 degrees

    train_ds = (
        train_ds.map(lambda x, y: (rotation_layer(x, training=True), y),
                     num_parallel_calls=tf.data.AUTOTUNE)
        .map(_augment, num_parallel_calls=tf.data.AUTOTUNE)
        .map(_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
        .prefetch(tf.data.AUTOTUNE)
    )
    val_ds = val_ds.map(_preprocess, num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)
    test_ds = test_ds.map(_preprocess, num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)

    return train_ds, val_ds, test_ds


if __name__ == "__main__":
    # Quick smoke test: run `python -m src.data_pipeline` after downloading data.
    weights = compute_class_weights()
    train_ds, val_ds, test_ds = build_datasets()
    for images, labels in train_ds.take(1):
        print("Batch shape:", images.shape, "Label shape:", labels.shape)
