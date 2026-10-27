"""
Prediction Service for PCOS AI Web Application.
Preprocesses ultrasound images and executes inference on EfficientNetB0, DenseNet121,
and the soft-voting ensemble. Integrates risk stratification and Grad-CAM generation.
"""

import os
import uuid
import time
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from PIL import Image
import numpy as np
import tensorflow as tf

from .model_loader import ModelManager
from .risk_service import get_risk_tier
from .gradcam_service import generate_gradcam_visualizations

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png"}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB

def validate_uploaded_image(file_storage) -> Tuple[bool, str]:
    """Validates file existence, extension, and decodability."""
    if not file_storage or file_storage.filename == "":
        return False, "No file was selected for upload."

    ext = Path(file_storage.filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        return False, f"Unsupported file extension '{ext}'. Allowed formats: JPG, JPEG, PNG."

    # Check decodability
    try:
        file_storage.stream.seek(0)
        img = Image.open(file_storage.stream)
        img.verify()
        file_storage.stream.seek(0)
    except Exception as e:
        return False, f"Corrupted or invalid image file: {str(e)}"

    return True, ""

def process_and_predict(
    image_input,
    filename: str = "scan.jpg",
    generate_xai: bool = True
) -> Dict[str, Any]:
    """
    Executes end-to-end inference on an ultrasound image using verified saved models.
    Args:
        image_input: FileStorage, PIL Image, or Path to image.
        filename: Original file name.
        generate_xai: Whether to compute Grad-CAM heatmaps.
    """
    t_start = time.time()
    manager = ModelManager.get_instance()
    
    if manager.eff_model is None or manager.dense_model is None:
        raise RuntimeError("Deep learning models are not initialized or failed to load. Check server logs.")

    req_id = str(uuid.uuid4())[:8]

    # 1. Load Image as RGB
    if isinstance(image_input, (str, Path)):
        pil_img = Image.open(str(image_input)).convert("RGB")
    elif hasattr(image_input, "stream"):
        image_input.stream.seek(0)
        pil_img = Image.open(image_input.stream).convert("RGB")
    else:
        pil_img = image_input.convert("RGB")

    orig_width, orig_height = pil_img.size

    # 2. Resize to Standardized (224, 224)
    img_resized = pil_img.resize((224, 224), Image.Resampling.BILINEAR)
    img_array = np.array(img_resized, dtype=np.float32)

    # 3. Architecture-Specific Normalizations
    in_eff = tf.keras.applications.efficientnet.preprocess_input(img_array.copy())
    in_eff = np.expand_dims(in_eff, axis=0)

    in_dense = tf.keras.applications.densenet.preprocess_input(img_array.copy())
    in_dense = np.expand_dims(in_dense, axis=0)

    # 4. Model Predictions
    p_eff_raw = float(manager.eff_model.predict(in_eff, verbose=0)[0][0])
    p_dense_raw = float(manager.dense_model.predict(in_dense, verbose=0)[0][0])

    w1 = manager.ensemble_weights.get("w_eff", 0.50)
    w2 = manager.ensemble_weights.get("w_dense", 0.50)
    threshold = manager.decision_threshold

    # 5. Soft-Voting Ensemble Probability
    p_ensemble_raw = float(w1 * p_eff_raw + w2 * p_dense_raw)

    # 6. Classification & Risk Stratification
    pred_class_id = 1 if p_ensemble_raw >= threshold else 0
    pred_class_name = manager.class_names.get(pred_class_id, "PCOS" if pred_class_id == 1 else "Healthy / Normal")
    
    score_pct = round(p_ensemble_raw * 100.0, 2)
    risk_info = get_risk_tier(score_pct)

    # Individual model percentage scores
    eff_pct = round(p_eff_raw * 100.0, 2)
    dense_pct = round(p_dense_raw * 100.0, 2)

    # 7. Generate Grad-CAM Explainability Visualizations
    xai_data = {}
    if generate_xai:
        try:
            eff_xai = generate_gradcam_visualizations(
                manager.eff_model, "efficientnet", pil_img, in_eff, request_id=req_id
            )
            dense_xai = generate_gradcam_visualizations(
                manager.dense_model, "densenet", pil_img, in_dense, request_id=req_id
            )
            xai_data = {
                "efficientnet": eff_xai,
                "densenet": dense_xai,
                "available": True
            }
        except Exception as e:
            print(f"Warning: Grad-CAM generation encountered an issue: {e}")
            xai_data = {"available": False, "error": str(e)}
    else:
        xai_data = {"available": False}

    elapsed = round(time.time() - t_start, 3)

    return {
        "request_id": req_id,
        "filename": filename,
        "dimensions": f"{orig_width} × {orig_height}",
        "elapsed_seconds": elapsed,
        "scores": {
            "efficientnet_score": p_eff_raw,
            "efficientnet_pct": eff_pct,
            "densenet_score": p_dense_raw,
            "densenet_pct": dense_pct,
            "ensemble_score": p_ensemble_raw,
            "ensemble_pct": score_pct
        },
        "ensemble_config": {
            "weight_efficientnet": w1,
            "weight_densenet": w2,
            "threshold": threshold
        },
        "predicted_class_id": pred_class_id,
        "predicted_class": pred_class_name,
        "confidence_score": round(max(p_ensemble_raw, 1.0 - p_ensemble_raw) * 100.0, 2),
        "risk_tier": risk_info,
        "xai": xai_data,
        "models_used": {
            "efficientnet": manager.eff_model_name,
            "densenet": manager.dense_model_name
        }
    }
