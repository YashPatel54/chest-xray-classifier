"""
Two model builders:
- build_baseline_cnn(): small CNN from scratch, comparison baseline (~85-88% acc expected).
- build_transfer_model(): DenseNet121 (ImageNet weights) + custom head, base frozen initially.
- unfreeze_for_fine_tuning(): unfreezes the top N layers and recompiles at a low LR.
"""
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import DenseNet121

from src import config

METRICS = [
    "accuracy",
    tf.keras.metrics.AUC(name="auc"),
    tf.keras.metrics.Precision(name="precision"),
    tf.keras.metrics.Recall(name="recall"),
]


def build_baseline_cnn():
    """Small CNN from scratch: 4 conv blocks -> dense head. Sanity-check baseline."""
    inputs = layers.Input(shape=(*config.IMG_SIZE, 3))
    x = inputs
    for filters in (32, 64, 128, 256):
        x = layers.Conv2D(filters, 3, padding="same", activation="relu")(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D()(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)

    model = models.Model(inputs, outputs, name="baseline_cnn")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(config.BASELINE_LR),
        loss="binary_crossentropy",
        metrics=METRICS,
    )
    return model


def build_transfer_model():
    """DenseNet121 backbone (frozen) + custom classification head."""
    base_model = DenseNet121(
        weights="imagenet", include_top=False, input_shape=(*config.IMG_SIZE, 3)
    )
    base_model.trainable = False

    inputs = layers.Input(shape=(*config.IMG_SIZE, 3))
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)

    model = models.Model(inputs, outputs, name="densenet121_transfer")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(config.FROZEN_LR),
        loss="binary_crossentropy",
        metrics=METRICS,
    )
    return model, base_model


def unfreeze_for_fine_tuning(model, base_model, num_layers=config.FINETUNE_UNFREEZE_LAYERS):
    """
    Unfreeze the top `num_layers` of the DenseNet121 base and recompile at a
    much lower LR. Must be called after the frozen phase has already trained
    the new head -- otherwise large random-init gradients from the head would
    wreck the pretrained backbone weights.
    """
    base_model.trainable = True
    for layer in base_model.layers[:-num_layers]:
        layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(config.FINETUNE_LR),
        loss="binary_crossentropy",
        metrics=METRICS,
    )
    return model


if __name__ == "__main__":
    baseline = build_baseline_cnn()
    baseline.summary()
    transfer_model, base = build_transfer_model()
    transfer_model.summary()
