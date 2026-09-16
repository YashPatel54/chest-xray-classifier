"""
Trains the baseline CNN, then the DenseNet121 transfer model in two phases
(frozen -> fine-tune). Saves best checkpoints, curves, and combined history.

Run: python -m src.train
"""
import json
import os

import matplotlib.pyplot as plt
import tensorflow as tf

from src import config
from src.data_pipeline import build_datasets, compute_class_weights
from src.model import build_baseline_cnn, build_transfer_model, unfreeze_for_fine_tuning


def _callbacks(checkpoint_path):
    return [
        tf.keras.callbacks.ModelCheckpoint(
            checkpoint_path, monitor="val_auc", mode="max", save_best_only=True, verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_auc", mode="max", patience=5, restore_best_weights=True, verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=3, min_lr=1e-7, verbose=1
        ),
    ]


def _plot_history(history_dict, title, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history_dict["loss"], label="train")
    axes[0].plot(history_dict["val_loss"], label="val")
    axes[0].set_title(f"{title} — Loss")
    axes[0].legend()

    axes[1].plot(history_dict["auc"], label="train")
    axes[1].plot(history_dict["val_auc"], label="val")
    axes[1].set_title(f"{title} — AUC")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def main():
    train_ds, val_ds, test_ds = build_datasets()
    class_weights = compute_class_weights()

    combined_history = {}

    # --- Baseline CNN ---
    print("\n=== Training baseline CNN ===")
    baseline = build_baseline_cnn()
    baseline_hist = baseline.fit(
        train_ds,
        validation_data=val_ds,
        epochs=config.BASELINE_EPOCHS,
        class_weight=class_weights,
        callbacks=_callbacks(config.BASELINE_MODEL_PATH),
    )
    combined_history["baseline"] = baseline_hist.history
    _plot_history(baseline_hist.history, "Baseline CNN", os.path.join(config.PLOT_DIR, "baseline_curves.png"))

    # --- Transfer model: Phase 1 (frozen base) ---
    print("\n=== Training DenseNet121 (frozen base) ===")
    transfer_model, base_model = build_transfer_model()
    frozen_hist = transfer_model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=config.FROZEN_EPOCHS,
        class_weight=class_weights,
        callbacks=_callbacks(config.TRANSFER_MODEL_PATH),
    )
    combined_history["frozen"] = frozen_hist.history
    _plot_history(frozen_hist.history, "DenseNet121 Frozen", os.path.join(config.PLOT_DIR, "frozen_curves.png"))

    # --- Transfer model: Phase 2 (fine-tune) ---
    print("\n=== Fine-tuning DenseNet121 (top layers unfrozen) ===")
    transfer_model = unfreeze_for_fine_tuning(transfer_model, base_model)
    finetune_hist = transfer_model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=config.FINETUNE_EPOCHS,
        class_weight=class_weights,
        callbacks=_callbacks(config.FINETUNED_MODEL_PATH),
    )
    combined_history["finetune"] = finetune_hist.history
    _plot_history(finetune_hist.history, "DenseNet121 Fine-tuned", os.path.join(config.PLOT_DIR, "finetune_curves.png"))

    with open(config.HISTORY_PATH, "w") as f:
        json.dump(combined_history, f, indent=2)

    print(f"\nDone. Models saved to {config.MODEL_DIR}, curves to {config.PLOT_DIR}.")


if __name__ == "__main__":
    main()
