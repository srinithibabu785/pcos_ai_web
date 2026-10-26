"""
Model Loader Service for PCOS AI Web Application.
Discovers and loads verified trained deep learning and tabular models on startup.
Caches models in memory to avoid reloading on individual requests.
"""

import os
import sys
import json
import time
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
import tensorflow as tf
import joblib

# Paths
WEB_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = WEB_DIR.parent
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
RESULTS_DIR = PROJECT_ROOT / "results"

class ModelManager:
    _instance = None

    def __init__(self):
        self.eff_model: Optional[tf.keras.Model] = None
        self.dense_model: Optional[tf.keras.Model] = None
        self.tabular_model = None
        self.tabular_preprocessor = None
        
        self.eff_model_name = ""
        self.dense_model_name = ""
        
        self.ensemble_weights = {"w_eff": 0.50, "w_dense": 0.50}
        self.decision_threshold = 0.60  # Default verified from reports/optimal_threshold.json
        self.class_names = {0: "Healthy / Normal", 1: "PCOS"}
        
        self.model_status: Dict[str, Any] = {
            "efficientnet": {"loaded": False, "name": "", "input_shape": None, "params": 0, "error": None},
            "densenet": {"loaded": False, "name": "", "input_shape": None, "params": 0, "error": None},
            "ensemble": {"ready": False, "weights": None, "threshold": 0.60},
            "tabular": {"loaded": False, "name": "", "features_count": 0, "error": None},
            "initialization_time": None
        }

    @classmethod
    def get_instance(cls) -> "ModelManager":
        if cls._instance is None:
            cls._instance = ModelManager()
            cls._instance.initialize_models()
        return cls._instance

    def initialize_models(self):
        """Loads models, preprocessors, and configuration files into memory."""
        start_time = time.time()
        print("\n========================================================")
        print("PCOS AI WEB: INITIALIZING VERIFIED MACHINE LEARNING MODELS")
        print("========================================================")

        # 1. Load Configurations
        self._load_configurations()

        # 2. Discover & Load EfficientNetB0
        self._load_efficientnet()

        # 3. Discover & Load DenseNet121
        self._load_densenet()

        # 4. Check Ensemble Readiness
        if self.eff_model is not None and self.dense_model is not None:
            self.model_status["ensemble"]["ready"] = True
            self.model_status["ensemble"]["weights"] = self.ensemble_weights
            self.model_status["ensemble"]["threshold"] = self.decision_threshold
            print(f"Ensemble Pipeline Ready: w1(Eff)={self.ensemble_weights['w_eff']}, w2(Dense)={self.ensemble_weights['w_dense']}, Threshold={self.decision_threshold}")

        # 5. Discover & Load Tabular Model & Preprocessor
        self._load_tabular()

        elapsed = time.time() - start_time
        self.model_status["initialization_time"] = round(elapsed, 2)
        print(f"Model initialization completed in {elapsed:.2f} seconds.\n")

    def _load_configurations(self):
        """Loads verified threshold, ensemble, and class mapping configs."""
        # Check reports/optimal_threshold.json first (from high-acc pipeline)
        thresh_path = REPORTS_DIR / "optimal_threshold.json"
        if thresh_path.exists():
            try:
                with open(thresh_path, "r", encoding="utf-8") as f:
                    cfg = json.load(f)
                    self.decision_threshold = float(cfg.get("optimal_threshold", 0.60))
                    print(f"Loaded optimal threshold from reports/optimal_threshold.json: {self.decision_threshold}")
            except Exception as e:
                print(f"Warning reading optimal_threshold.json: {e}")

        # Fallback to models/threshold_config.json
        fallback_thresh = MODELS_DIR / "threshold_config.json"
        if not thresh_path.exists() and fallback_thresh.exists():
            try:
                with open(fallback_thresh, "r", encoding="utf-8") as f:
                    cfg = json.load(f)
                    self.decision_threshold = float(cfg.get("selected_threshold", 0.50))
                    print(f"Loaded threshold from models/threshold_config.json: {self.decision_threshold}")
            except Exception as e:
                pass

        # Ensemble weights
        ens_cfg_path = MODELS_DIR / "ensemble_config.json"
        if ens_cfg_path.exists():
            try:
                with open(ens_cfg_path, "r", encoding="utf-8") as f:
                    cfg = json.load(f)
                    w1 = float(cfg.get("efficientnet_weight", 0.50))
                    w2 = float(cfg.get("densenet_weight", 0.50))
                    self.ensemble_weights = {"w_eff": w1, "w_dense": w2}
            except Exception as e:
                pass

        # Class names
        cls_path = MODELS_DIR / "class_names.json"
        if cls_path.exists():
            try:
                with open(cls_path, "r", encoding="utf-8") as f:
                    raw_cls = json.load(f)
                    self.class_names = {int(k): str(v) for k, v in raw_cls.items()}
            except Exception as e:
                pass

    def _load_efficientnet(self):
        """Discovers and loads verified EfficientNetB0 .keras model."""
        candidates = [
            MODELS_DIR / "EfficientNetB0_best.keras",
            MODELS_DIR / "efficientnetb0_pcos.keras",
            MODELS_DIR / "EfficientNetB0_stage1_best.keras"
        ]
        chosen_path = next((p for p in candidates if p.exists()), None)
        if chosen_path:
            try:
                print(f"Loading EfficientNetB0 model from: {chosen_path.name}...")
                self.eff_model = tf.keras.models.load_model(str(chosen_path))
                self.eff_model_name = chosen_path.name
                param_count = self.eff_model.count_params()
                self.model_status["efficientnet"] = {
                    "loaded": True,
                    "name": chosen_path.name,
                    "input_shape": list(self.eff_model.input_shape),
                    "params": param_count,
                    "error": None
                }
                print(f"EfficientNetB0 loaded successfully! (Parameters: {param_count:,})")
            except Exception as e:
                print(f"Error loading EfficientNetB0: {e}")
                self.model_status["efficientnet"]["error"] = str(e)
        else:
            self.model_status["efficientnet"]["error"] = "No compatible EfficientNetB0 model file located in models/"

    def _load_densenet(self):
        """Discovers and loads verified DenseNet121 .keras model."""
        candidates = [
            MODELS_DIR / "DenseNet121_best.keras",
            MODELS_DIR / "densenet121_pcos.keras"
        ]
        chosen_path = next((p for p in candidates if p.exists()), None)
        if chosen_path:
            try:
                print(f"Loading DenseNet121 model from: {chosen_path.name}...")
                self.dense_model = tf.keras.models.load_model(str(chosen_path))
                self.dense_model_name = chosen_path.name
                param_count = self.dense_model.count_params()
                self.model_status["densenet"] = {
                    "loaded": True,
                    "name": chosen_path.name,
                    "input_shape": list(self.dense_model.input_shape),
                    "params": param_count,
                    "error": None
                }
                print(f"DenseNet121 loaded successfully! (Parameters: {param_count:,})")
            except Exception as e:
                print(f"Error loading DenseNet121: {e}")
                self.model_status["densenet"]["error"] = str(e)
        else:
            self.model_status["densenet"]["error"] = "No compatible DenseNet121 model file located in models/"

    def _load_tabular(self):
        """Loads verified tabular Random Forest model and preprocessor."""
        tab_model_path = MODELS_DIR / "tabular_model.pkl"
        tab_prep_path = MODELS_DIR / "tabular_preprocessor.pkl"

        if tab_model_path.exists() and tab_prep_path.exists():
            try:
                self.tabular_model = joblib.load(str(tab_model_path))
                self.tabular_preprocessor = joblib.load(str(tab_prep_path))
                
                # Check feature count
                n_features = getattr(self.tabular_model, "n_features_in_", 41)
                self.model_status["tabular"] = {
                    "loaded": True,
                    "name": "RandomForest (41 Clinical Features)",
                    "features_count": n_features,
                    "error": None
                }
                print(f"Tabular Clinical Model loaded successfully! (Type: {type(self.tabular_model).__name__}, Features: {n_features})")
            except Exception as e:
                print(f"Error loading tabular model: {e}")
                self.model_status["tabular"]["error"] = str(e)
        else:
            self.model_status["tabular"]["error"] = "Tabular model or preprocessor artifact missing in models/"
