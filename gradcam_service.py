"""
Grad-CAM Explainable AI (XAI) Service for PCOS AI Web Application.
Generates genuine visual attribution heatmaps highlighting spatial regions
(follicles, ovarian stroma) that influenced deep learning model predictions.
"""

import os
import uuid
from pathlib import Path
from typing import Dict, Any, Tuple, Optional

import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import Model

WEB_DIR = Path(__file__).resolve().parent.parent
GENERATED_DIR = WEB_DIR / "static" / "generated"
GENERATED_DIR.mkdir(parents=True, exist_ok=True)

TARGET_CONV_LAYERS = {
    "efficientnet": ["top_conv", "block7a_project_conv"],
    "densenet": ["conv5_block16_2_conv", "relu"]
}

def find_conv_layer(model: Model, preferred_names: list) -> str:
    """Finds target convolutional layer in the loaded functional model."""
    layer_names = [l.name for l in model.layers]
    for name in preferred_names:
        if name in layer_names:
            return name

    # Search in reverse
    for l in reversed(model.layers):
        if isinstance(l, tf.keras.layers.Conv2D):
            return l.name
        if hasattr(l, "layers"):
            for sub in reversed(l.layers):
                if isinstance(sub, tf.keras.layers.Conv2D):
                    return f"{l.name}/{sub.name}"

    raise ValueError(f"Could not locate a convolutional layer in {model.name}")

def compute_heatmap(model: Model, img_tensor: tf.Tensor, layer_name: str) -> np.ndarray:
    """Computes Grad-CAM salience heatmap using tf.GradientTape."""
    target_layer = model.get_layer(layer_name)
    grad_model = Model(
        inputs=model.inputs,
        outputs=[target_layer.output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_tensor)
        class_score = predictions[:, 0]

    grads = tape.gradient(class_score, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0.0)
    max_val = tf.math.reduce_max(heatmap)
    if max_val > 0:
        heatmap = heatmap / max_val

    return heatmap.numpy()

def generate_gradcam_visualizations(
    model: Model,
    model_type: str,
    raw_img_pil: Image.Image,
    prep_tensor: tf.Tensor,
    request_id: Optional[str] = None
) -> Dict[str, str]:
    """
    Generates genuine Grad-CAM original, heatmap, overlay, and 3-panel composite.
    Returns dictionary with relative web URLs for static serving.
    """
    req_id = request_id or str(uuid.uuid4())[:8]
    model_type_clean = model_type.lower()
    
    # 1. Identify Target Convolutional Layer
    preferred = TARGET_CONV_LAYERS.get("efficientnet" if "eff" in model_type_clean else "densenet", ["top_conv"])
    target_layer_name = find_conv_layer(model, preferred)

    # 2. Compute Heatmap
    heatmap = compute_heatmap(model, prep_tensor, target_layer_name)
    
    # Resize to 224x224
    heatmap_resized = cv2.resize(heatmap, (224, 224))
    heatmap_uint8 = np.uint8(255 * heatmap_resized)

    # 3. Apply JET Colormap
    jet = plt.colormaps["jet"]
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = np.uint8(jet_colors[heatmap_uint8] * 255)

    # 4. Superimpose with Original Image
    orig_np = np.array(raw_img_pil.resize((224, 224)))
    overlay = cv2.addWeighted(orig_np, 0.60, jet_heatmap, 0.40, 0)

    # 5. Save Artifacts
    base_fn = f"gradcam_{model_type_clean}_{req_id}"
    orig_path = GENERATED_DIR / f"{base_fn}_orig.png"
    heat_path = GENERATED_DIR / f"{base_fn}_heat.png"
    over_path = GENERATED_DIR / f"{base_fn}_overlay.png"
    comp_path = GENERATED_DIR / f"{base_fn}_composite.png"

    Image.fromarray(orig_np).save(orig_path)
    Image.fromarray(jet_heatmap).save(heat_path)
    Image.fromarray(overlay).save(over_path)

    # 6. Generate 3-Panel Composite
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), dpi=200)
    fig.patch.set_facecolor('#F8FAFC')
    
    axes[0].imshow(orig_np)
    axes[0].set_title("1. Original Ultrasound Scan", fontsize=11, fontweight="bold", color="#1E293B", pad=8)
    axes[0].axis("off")

    im1 = axes[1].imshow(heatmap_resized, cmap="jet")
    axes[1].set_title(f"2. Grad-CAM Salience Heatmap\n(Layer: {target_layer_name})", fontsize=11, fontweight="bold", color="#1E293B", pad=8)
    axes[1].axis("off")
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    axes[2].imshow(overlay)
    axes[2].set_title("3. Explainability Overlay", fontsize=11, fontweight="bold", color="#1E293B", pad=8)
    axes[2].axis("off")

    plt.suptitle(f"{model.name.upper()} — Spatial Feature Attribution Analysis", fontsize=12, fontweight="bold", color="#334155", y=0.98)
    plt.tight_layout()
    plt.savefig(comp_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    return {
        "original_url": f"/static/generated/{orig_path.name}",
        "heatmap_url": f"/static/generated/{heat_path.name}",
        "overlay_url": f"/static/generated/{over_path.name}",
        "composite_url": f"/static/generated/{comp_path.name}",
        "target_layer": target_layer_name
    }
