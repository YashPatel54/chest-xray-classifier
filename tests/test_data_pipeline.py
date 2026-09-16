"""
Smoke tests for the data pipeline's non-tf.data logic (class weighting).
Builds a tiny fake train/ folder instead of touching the real dataset, so
this runs in seconds and doesn't require the dataset to be downloaded.
"""
import os

import pytest

from src import config
from src.data_pipeline import compute_class_weights


@pytest.fixture
def fake_train_dir(tmp_path, monkeypatch):
    """
    Build a fake train/ dir with an imbalanced class split (mirrors the real
    dataset's ~3x PNEUMONIA:NORMAL ratio) and point config.TRAIN_DIR at it.
    """
    train_dir = tmp_path / "train"
    counts = {"NORMAL": 10, "PNEUMONIA": 30}
    for class_name, n in counts.items():
        class_dir = train_dir / class_name
        class_dir.mkdir(parents=True)
        for i in range(n):
            (class_dir / f"img_{i}.jpeg").write_bytes(b"fake image bytes")

    monkeypatch.setattr(config, "TRAIN_DIR", str(train_dir))
    return counts


def test_class_weights_favor_minority_class(fake_train_dir):
    weights = compute_class_weights()

    normal_idx = config.CLASS_NAMES.index("NORMAL")
    pneumonia_idx = config.CLASS_NAMES.index("PNEUMONIA")

    # NORMAL is the minority class (10 vs 30) -> it must get the larger weight,
    # or the imbalance correction is backwards.
    assert weights[normal_idx] > weights[pneumonia_idx]


def test_class_weights_are_positive_floats(fake_train_dir):
    weights = compute_class_weights()
    assert all(w > 0 for w in weights.values())
    assert set(weights.keys()) == {0, 1}


def test_class_weights_exact_formula(fake_train_dir):
    # weight_i = n_samples / (n_classes * n_samples_i)
    counts = fake_train_dir
    total = sum(counts.values())
    n_classes = len(counts)
    weights = compute_class_weights()

    for i, class_name in enumerate(config.CLASS_NAMES):
        expected = total / (n_classes * counts[class_name])
        assert weights[i] == pytest.approx(expected)
