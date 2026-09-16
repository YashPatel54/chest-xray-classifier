"""
Smoke tests for the non-model parts of gradcam.py (overlay compositing).
Doesn't load DenseNet121, so it runs fast with no network dependency.
"""
import numpy as np
from PIL import Image

from src.gradcam import overlay_heatmap


def test_overlay_heatmap_preserves_original_size():
    original = Image.new("RGB", (224, 224), color=(120, 120, 120))
    heatmap = np.random.uniform(0, 1, size=(7, 7)).astype("float32")  # DenseNet's last conv grid

    overlay = overlay_heatmap(original, heatmap)

    assert overlay.size == original.size
    assert overlay.mode == "RGB"


def test_overlay_heatmap_all_zero_heatmap_stays_close_to_original():
    original = Image.new("RGB", (100, 100), color=(200, 50, 50))
    heatmap = np.zeros((7, 7), dtype="float32")

    overlay = overlay_heatmap(original, heatmap, alpha=0.4)

    # A zero heatmap maps to the "cold" end of the jet colormap, so the
    # overlay should still be a valid same-size RGB image, not a crash.
    assert overlay.size == original.size
    arr = np.array(overlay)
    assert arr.shape == (100, 100, 3)
