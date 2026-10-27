"""
What-If Clinical Lab Service for PCOS AI Web Application.
Uses the verified tabular clinical Random Forest model and ColumnTransformer
to simulate how adjusting clinical, metabolic, and hormonal parameters
alters the AI-predicted PCOS likelihood score.
"""

from pathlib import Path
from typing import Dict, Any, List
import pandas as pd
import numpy as np

from .model_loader import ModelManager
from .risk_service import get_risk_tier

WEB_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = WEB_DIR.parent
DATA_PATH = PROJECT_ROOT / "data_processed" / "tabular_test.csv"

# Pre-computed baseline median patient profile from tabular dataset
DEFAULT_PROFILE = {
    'Sl. No': 1,
    'Patient File No.': 1001,
    'Age (yrs)': 28,
    'Weight (Kg)': 62.0,
    'Height(Cm)': 160.0,
    'BMI': 24.2,
    'Blood Group': 13,
    'Pulse rate(bpm)': 72.0,
    'RR (breaths/min)': 18.0,
    'Hb(g/dl)': 12.0,
    'Cycle(R/I)': 2,              # 2 = Regular, 4 = Irregular
    'Cycle length(days)': 28.0,
    'Marraige Status (Yrs)': 3.0,
    'Pregnant(Y/N)': 0,
    'No. of abortions': 0,
    'I   beta-HCG(mIU/mL)': 1.99,
    'II    beta-HCG(mIU/mL)': 1.99,
    'FSH(mIU/mL)': 6.0,
    'LH(mIU/mL)': 4.5,
    'FSH/LH': 1.33,
    'Hip(inch)': 38.0,
    'Waist(inch)': 32.0,
    'Waist:Hip Ratio': 0.84,
    'TSH (mIU/L)': 2.5,
    'AMH(ng/mL)': 3.2,
    'PRL(ng/mL)': 18.0,
    'Vit D3 (ng/mL)': 22.0,
    'PRG(ng/mL)': 0.5,
    'RBS(mg/dl)': 92.0,
    'Weight gain(Y/N)': 0,
    'hair growth(Y/N)': 0,
    'Skin darkening (Y/N)': 0,
    'Hair loss(Y/N)': 0,
    'Pimples(Y/N)': 0,
    'Fast food (Y/N)': 0,
    'Reg.Exercise(Y/N)': 1,
    'BP _Systolic (mmHg)': 115.0,
    'BP _Diastolic (mmHg)': 75.0,
    'Follicle No. (L)': 5.0,
    'Follicle No. (R)': 6.0,
    'Avg. F size (L) (mm)': 14.0,
    'Avg. F size (R) (mm)': 15.0,
    'Endometrium (mm)': 8.0
}

def simulate_whatif(user_adjustments: Dict[str, Any]) -> Dict[str, Any]:
    """
    Executes What-If inference using the trained tabular model.
    Merges user overrides into the standardized clinical feature schema.
    """
    manager = ModelManager.get_instance()
    
    if manager.tabular_model is None or manager.tabular_preprocessor is None:
        return {
            "available": False,
            "error": "Tabular clinical model or preprocessor is not loaded."
        }

    # Start with default profile
    profile = DEFAULT_PROFILE.copy()

    # Apply user overrides
    for k, v in user_adjustments.items():
        if k in profile:
            try:
                profile[k] = float(v)
            except (ValueError, TypeError):
                profile[k] = v

    # Recalculate derived clinical features
    weight = profile.get('Weight (Kg)', 62.0)
    height_m = profile.get('Height(Cm)', 160.0) / 100.0
    if height_m > 0:
        profile['BMI'] = round(weight / (height_m ** 2), 2)

    waist = profile.get('Waist(inch)', 32.0)
    hip = profile.get('Hip(inch)', 38.0)
    if hip > 0:
        profile['Waist:Hip Ratio'] = round(waist / hip, 2)

    fsh = profile.get('FSH(mIU/mL)', 6.0)
    lh = profile.get('LH(mIU/mL)', 4.5)
    if lh > 0:
        profile['FSH/LH'] = round(fsh / lh, 2)

    # Prepare DataFrame matching preprocessor schema
    df_input = pd.DataFrame([profile])

    # Transform through ColumnTransformer
    X_trans = manager.tabular_preprocessor.transform(df_input)

    # Predict Probability
    prob_pcos = float(manager.tabular_model.predict_proba(X_trans)[0, 1])
    score_pct = round(prob_pcos * 100.0, 2)
    risk_info = get_risk_tier(score_pct)

    # Baseline comparison (with default profile)
    df_base = pd.DataFrame([DEFAULT_PROFILE])
    X_base = manager.tabular_preprocessor.transform(df_base)
    base_prob = float(manager.tabular_model.predict_proba(X_base)[0, 1])
    base_pct = round(base_prob * 100.0, 2)
    delta_pct = round(score_pct - base_pct, 2)

    return {
        "available": True,
        "score_pct": score_pct,
        "prob_raw": prob_pcos,
        "predicted_class": "PCOS" if prob_pcos >= 0.50 else "Healthy / Normal",
        "risk_tier": risk_info,
        "baseline_pct": base_pct,
        "delta_pct": delta_pct,
        "active_parameters": {
            "age": profile['Age (yrs)'],
            "bmi": profile['BMI'],
            "weight": profile['Weight (Kg)'],
            "height": profile['Height(Cm)'],
            "cycle_type": "Irregular" if profile['Cycle(R/I)'] > 2 else "Regular",
            "cycle_length": profile['Cycle length(days)'],
            "amh": profile['AMH(ng/mL)'],
            "lh": profile['LH(mIU/mL)'],
            "fsh": profile['FSH(mIU/mL)'],
            "follicles_left": profile['Follicle No. (L)'],
            "follicles_right": profile['Follicle No. (R)'],
            "weight_gain": bool(profile['Weight gain(Y/N)']),
            "hair_growth": bool(profile['hair growth(Y/N)']),
            "skin_darkening": bool(profile['Skin darkening (Y/N)'])
        }
    }
