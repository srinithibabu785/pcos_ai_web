"""
Risk Stratification Service for PCOS AI Web Application.
Maps model-predicted likelihood scores into standardized presentation tiers.
Enforces responsible healthcare AI disclaimers and clear presentation guidelines.
"""

from typing import Dict, Any

def get_risk_tier(score_pct: float) -> Dict[str, Any]:
    """
    Standardizes model likelihood percentage [0 - 100] into 5 project presentation bands.
    NOTE: These bands are presentation categories for clinical decision support,
    not standalone medical diagnostic criteria.
    """
    if score_pct <= 20.0:
        return {
            "tier": 1,
            "category": "Very Low",
            "badge_class": "badge-very-low",
            "color": "#10B981",       # Mint / Green
            "color_name": "Green",
            "band": "0–20%",
            "clinical_guidance": "Ultrasound features are consistent with normal ovarian parenchymal patterns. Low model-predicted morphological evidence for polycystic appearance.",
            "next_steps": "Continue routine preventive health checkups. If menstrual irregularities or hormonal concerns persist, consult a gynecologist for a complete clinical assessment."
        }
    elif score_pct <= 40.0:
        return {
            "tier": 2,
            "category": "Low",
            "badge_class": "badge-low",
            "color": "#EAB308",       # Yellow
            "color_name": "Yellow",
            "band": "21–40%",
            "clinical_guidance": "Mild or non-specific follicular variations detected by transfer-learning filters. Sub-threshold pattern for typical polycystic ovarian morphology.",
            "next_steps": "Correlate with clinical history, cycle tracking, and basic metabolic panels (e.g., fasting glucose, lipid profile) during scheduled physician visits."
        }
    elif score_pct <= 60.0:
        return {
            "tier": 3,
            "category": "Moderate (Borderline)",
            "badge_class": "badge-moderate",
            "color": "#F97316",       # Orange
            "color_name": "Orange",
            "band": "41–60%",
            "clinical_guidance": "Equivocal model score. Imaging features show intermediate follicular distribution between normal and multicystic patterns.",
            "next_steps": "Biochemical laboratory correlation is strongly recommended (Total/Free Testosterone, LH/FSH ratio, AMH, Fasting Insulin) to verify endocrine status under Rotterdam criteria."
        }
    elif score_pct <= 80.0:
        return {
            "tier": 4,
            "category": "High",
            "badge_class": "badge-high",
            "color": "#EF4444",       # Red
            "color_name": "Red",
            "band": "61–80%",
            "clinical_guidance": "Elevated AI-predicted likelihood. The ensemble model identified prominent peripheral follicular clustering or increased stromal density typical of PCO morphology.",
            "next_steps": "Schedule a comprehensive clinical consultation with an endocrinologist or gynecologist. Formal review of transvaginal ultrasound imaging (>= 20 follicles per ovary or ovarian volume >= 10 mL) is indicated."
        }
    else:
        return {
            "tier": 5,
            "category": "Very High",
            "badge_class": "badge-very-high",
            "color": "#991B1B",       # Dark Red
            "color_name": "Dark Red",
            "band": "81–100%",
            "clinical_guidance": "Strong visual feature attribution corresponding to classic 'necklace sign' follicular distribution and expanded ovarian stroma.",
            "next_steps": "High priority for specialized clinical evaluation. Discuss personalized lifestyle, metabolic, and ovulatory management strategies with a licensed healthcare professional."
        }

RESPONSIBLE_AI_DISCLAIMER = (
    "This AI output is intended for academic research, medical machine learning demonstration, "
    "and clinical decision support only. It is NOT a medical diagnosis and should never replace "
    "evaluation, physical examination, or treatment planning by a qualified healthcare professional."
)
