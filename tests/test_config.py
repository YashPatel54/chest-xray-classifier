"""
Smoke tests for config.py. These just guard against silly mistakes (typos in
paths, mismatched class counts) that would otherwise only surface hours into
a training run.
"""
import os

from src import config


def test_img_size_is_square_tuple():
    assert isinstance(config.IMG_SIZE, tuple)
    assert len(config.IMG_SIZE) == 2
    assert config.IMG_SIZE[0] == config.IMG_SIZE[1] == 224


def test_class_names_match_binary_setup():
    # Binary sigmoid output assumes exactly 2 classes, index 0 / 1.
    assert config.CLASS_NAMES == ["NORMAL", "PNEUMONIA"]


def test_output_dirs_exist():
    # config.py creates these on import -- verify that actually happened.
    for d in (config.MODEL_DIR, config.PLOT_DIR, config.GRADCAM_DIR):
        assert os.path.isdir(d), f"{d} should be created on import"


def test_learning_rates_are_sane():
    # Fine-tune LR must be much smaller than the frozen-phase LR, or fine-
    # tuning will wreck the pretrained backbone weights.
    assert config.FINETUNE_LR < config.FROZEN_LR
    assert config.FINETUNE_LR <= 1e-4


def test_data_dir_paths_nest_correctly():
    assert config.TRAIN_DIR.startswith(config.DATA_DIR)
    assert config.VAL_DIR.startswith(config.DATA_DIR)
    assert config.TEST_DIR.startswith(config.DATA_DIR)
