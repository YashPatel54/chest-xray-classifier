"""
Loads a trained model and runs inference on the TEST set only (touched once,
here). Reports classification report, confusion matrix, and ROC curve/AUC.

Run: python -m src.evaluate --model outputs/models/densenet121_finetuned.keras
"""
import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_auc_score,
    roc_curve,
)

from src import config
from src.data_pipeline import build_datasets


def evaluate(model_path):
    model = tf.keras.models.load_model(model_path)
    _, _, test_ds = build_datasets()

    y_true, y_prob = [], []
    for images, labels in test_ds:
        preds = model.predict(images, verbose=0).flatten()
        y_prob.extend(preds)
        y_true.extend(labels.numpy().flatten())

    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    y_pred = (y_prob >= 0.5).astype(int)

    print("\n=== Classification Report ===")
    report = classification_report(y_true, y_pred, target_names=config.CLASS_NAMES)
    print(report)
    print(
        "Note: recall on PNEUMONIA matters most here -- a false negative "
        "(missed pneumonia) is worse than a false alarm in a screening context."
    )

    # --- Confusion matrix ---
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=config.CLASS_NAMES)
    disp.plot(cmap="Blues")
    plt.title("Confusion Matrix — Test Set")
    plt.savefig(os.path.join(config.PLOT_DIR, "confusion_matrix.png"))
    plt.close()

    # --- ROC curve ---
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)
    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve — Test Set")
    plt.legend()
    plt.savefig(os.path.join(config.PLOT_DIR, "roc_curve.png"))
    plt.close()

    print(f"\nTest AUC: {auc:.4f}")
    print(f"Plots saved to {config.PLOT_DIR}")

    with open(os.path.join(config.PLOT_DIR, "classification_report.txt"), "w") as f:
        f.write(report)
        f.write(f"\nTest AUC: {auc:.4f}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to a trained .keras model")
    args = parser.parse_args()
    evaluate(args.model)
