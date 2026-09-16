"""
Smoke tests for model.py. These check shapes and compilation, not accuracy --
the point is to catch a broken architecture (wrong output shape, uncompiled
model, mismatched layer count) before burning an hour of GPU time on it.

The DenseNet121 transfer-model tests download ImageNet weights on first call,
so they're skipped unless RUN_SLOW_TESTS=1 is set (e.g. in CI with network,
or locally once you're set up).
"""
import os

import numpy as np
import pytest
import tensorflow as tf

from src import config
from src.model import build_baseline_cnn, build_transfer_model, unfreeze_for_fine_tuning

needs_network = pytest.mark.skipif(
    not os.environ.get("RUN_SLOW_TESTS"),
    reason="downloads ImageNet weights; set RUN_SLOW_TESTS=1 to run",
)


def _dummy_batch(batch_size=2):
    return np.random.uniform(-1, 1, size=(batch_size, *config.IMG_SIZE, 3)).astype("float32")


def test_baseline_cnn_output_shape():
    model = build_baseline_cnn()
    preds = model.predict(_dummy_batch(), verbose=0)
    assert preds.shape == (2, 1)
    assert np.all((preds >= 0) & (preds <= 1)), "sigmoid output must be in [0, 1]"


def test_baseline_cnn_is_compiled_with_required_metrics():
    model = build_baseline_cnn()
    dummy_labels = np.random.randint(0, 2, size=(2, 1)).astype("float32")
    # return_dict=True surfaces each metric by name; model.metrics_names
    # collapses them all under one "compile_metrics" entry in this Keras version.
    results = model.evaluate(_dummy_batch(), dummy_labels, verbose=0, return_dict=True)

    for expected in ("auc", "precision", "recall", "accuracy"):
        assert expected in results, f"missing metric: {expected}"


@needs_network
def test_transfer_model_output_shape():
    model, base_model = build_transfer_model()
    preds = model.predict(_dummy_batch(), verbose=0)
    assert preds.shape == (2, 1)
    assert base_model.trainable is False, "base model must start frozen"


@needs_network
def test_unfreeze_for_fine_tuning_unfreezes_only_top_layers():
    model, base_model = build_transfer_model()
    unfreeze_for_fine_tuning(model, base_model, num_layers=10)

    trainable_count = sum(1 for layer in base_model.layers if layer.trainable)
    assert trainable_count == 10
    # optimizer LR should have dropped for fine-tuning
    assert float(model.optimizer.learning_rate.numpy()) == pytest.approx(config.FINETUNE_LR)
