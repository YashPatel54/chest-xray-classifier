"""
Grad-CAM on DenseNet121's last conv layer (conv5_block16_concat).

Mechanics: run a forward pass capturing that layer's feature maps, take the
gradient of the predicted class score w.r.t. those feature maps, global-average
-pool the gradients per channel to get "importance weights," weight the
feature maps by them, sum, ReLU, and resize the result over the original image.

Warm regions should land on the lungs, not image borders/text artifacts --
that's a useful sanity check on whether the model learned something real.
"""
import argparse
import os
import random

import matplotlib
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.densenet import preprocess_input
from PIL import Image

from src import config


def make_gradcam_heatmap(img_array, model, layer_name=config.GRADCAM_LAYER_NAME):
    """
    img_array: preprocessed batch of shape (1, H, W, 3). Returns a 2D heatmap in [0,1].

    DenseNet121 is embedded as a single nested layer (named "densenet121")
    inside the full model, rather than being flattened into it. Keras cannot
    trace a Functional model whose output is an intermediate tensor *inside*
    a nested sub-model -- that tensor isn't considered "connected" to the
    outer model's inputs. So this is done in two stages instead:
      1. Build a small model entirely within the backbone's own graph that
         outputs both the target conv layer and the backbone's final output.
      2. Manually replay the remaining head layers (GAP -> Dropout -> Dense ->
         Dropout -> Dense) on top of that, inside the GradientTape, so
         gradients still flow from the prediction back to the target layer.
    """
    base_model = model.get_layer("densenet121")
    base_grad_model = tf.keras.models.Model(
        inputs=base_model.input,
        outputs=[base_model.get_layer(layer_name).output, base_model.output],
    )

    backbone_index = model.layers.index(base_model)
    head_layers = model.layers[backbone_index + 1:]

    with tf.GradientTape() as tape:
        conv_output, backbone_output = base_grad_model(img_array)
        h = backbone_output
        for layer in head_layers:
            h = layer(h, training=False)
        predictions = h
        class_score = predictions[:, 0]  # sigmoid output, single unit

    grads = tape.gradient(class_score, conv_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))  # per-channel importance

    conv_output = conv_output[0]
    heatmap = conv_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy(), float(predictions[0][0])


def overlay_heatmap(original_img, heatmap, alpha=0.4):
    """original_img: PIL Image (RGB). Returns a PIL Image with heatmap overlaid."""
    heatmap_resized = np.array(
        Image.fromarray(np.uint8(255 * heatmap)).resize(original_img.size)
    )
    jet = matplotlib.colormaps["jet"]
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap_resized]
    jet_heatmap = Image.fromarray(np.uint8(jet_heatmap * 255)).resize(original_img.size)

    overlay = Image.blend(original_img.convert("RGB"), jet_heatmap, alpha)
    return overlay


def generate_for_random_samples(model_path, n_samples=config.RANDOM_SAMPLES_FOR_GRADCAM):
    model = tf.keras.models.load_model(model_path)

    samples = []
    for class_name in config.CLASS_NAMES:
        class_dir = os.path.join(config.TEST_DIR, class_name)
        files = [f for f in os.listdir(class_dir) if not f.startswith(".")]
        chosen = random.sample(files, min(n_samples // len(config.CLASS_NAMES) + 1, len(files)))
        samples.extend([(os.path.join(class_dir, f), class_name) for f in chosen])
    samples = samples[:n_samples]

    for path, true_class in samples:
        original = Image.open(path).convert("RGB").resize(config.IMG_SIZE)
        img_array = np.expand_dims(np.array(original), axis=0).astype("float32")
        preprocessed = preprocess_input(img_array.copy())

        heatmap, pred_prob = make_gradcam_heatmap(preprocessed, model)
        overlay = overlay_heatmap(original, heatmap)

        pred_class = config.CLASS_NAMES[1] if pred_prob >= 0.5 else config.CLASS_NAMES[0]
        out_name = f"{true_class}_true__{pred_class}_pred_{os.path.basename(path)}"
        overlay.save(os.path.join(config.GRADCAM_DIR, out_name))
        print(f"{path} -> true={true_class}, pred={pred_class} ({pred_prob:.3f})")

    print(f"\nGrad-CAM overlays saved to {config.GRADCAM_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=config.FINETUNED_MODEL_PATH)
    parser.add_argument("--random-samples", type=int, default=config.RANDOM_SAMPLES_FOR_GRADCAM)
    args = parser.parse_args()
    generate_for_random_samples(args.model, args.random_samples)
