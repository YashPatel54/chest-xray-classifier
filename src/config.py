"""
Single source of truth for paths and hyperparameters.
Every other module imports from here instead of hardcoding values.
"""
import os

# --- Paths ---
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "chest_xray")
TRAIN_DIR = os.path.join(DATA_DIR, "train")
VAL_DIR = os.path.join(DATA_DIR, "val")
TEST_DIR = os.path.join(DATA_DIR, "test")

OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")
MODEL_DIR = os.path.join(OUTPUT_DIR, "models")
PLOT_DIR = os.path.join(OUTPUT_DIR, "plots")
GRADCAM_DIR = os.path.join(OUTPUT_DIR, "gradcam")

BASELINE_MODEL_PATH = os.path.join(MODEL_DIR, "baseline_cnn.keras")
TRANSFER_MODEL_PATH = os.path.join(MODEL_DIR, "densenet121_frozen.keras")
FINETUNED_MODEL_PATH = os.path.join(MODEL_DIR, "densenet121_finetuned.keras")
HISTORY_PATH = os.path.join(OUTPUT_DIR, "training_history.json")

# --- Data ---
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 42
CLASS_NAMES = ["NORMAL", "PNEUMONIA"]  # index 0 / 1 -- matches sigmoid output

# --- Training: Phase 1 (frozen base) ---
FROZEN_EPOCHS = 10
FROZEN_LR = 1e-3

# --- Training: Phase 2 (fine-tune) ---
FINETUNE_EPOCHS = 15
FINETUNE_LR = 1e-5
FINETUNE_UNFREEZE_LAYERS = 30  # unfreeze the top N layers of DenseNet121

# --- Baseline CNN training ---
BASELINE_EPOCHS = 15
BASELINE_LR = 1e-3

# --- Grad-CAM ---
GRADCAM_LAYER_NAME = "conv5_block16_concat"  # last conv block in DenseNet121

# --- Misc ---
RANDOM_SAMPLES_FOR_GRADCAM = 6

for _dir in (MODEL_DIR, PLOT_DIR, GRADCAM_DIR):
    os.makedirs(_dir, exist_ok=True)
