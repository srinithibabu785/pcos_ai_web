"""
Healthcare Recommendations and Clinical Care Compass Service.
Provides evidence-based educational guidance, diagnostic pathway descriptions,
lifestyle support, and clinician consultation checklists.
"""

from typing import Dict, Any, List

def get_clinical_guidance(risk_category: str) -> Dict[str, Any]:
    """Generates stratified clinical recommendations based on AI risk tier."""
    base_disclaimer = (
        "Educational Guidance: These suggestions are based on international clinical guidelines "
        "(Monash University / ESHRE / ASRM 2023). They do not constitute medical orders or diagnoses."
    )

    diagnostic_panels = [
        {
            "name": "Endocrine & Androgen Panel",
            "tests": ["Total Testosterone", "Free Androgen Index (FAI)", "SHBG", "DHEAS", "17-OHP (to rule out CAH)"],
            "purpose": "Evaluates biochemical hyperandrogenism, a hallmark criterion of PCOS."
        },
        {
            "name": "Gonadotropin & Ovarian Reserve",
            "tests": ["LH (Luteinizing Hormone)", "FSH (Follicle-Stimulating Hormone)", "LH/FSH Ratio", "AMH (Anti-Müllerian Hormone)"],
            "purpose": "Assesses hypothalamic-pituitary-ovarian axis signaling and granulosa cell follicle reserve."
        },
        {
            "name": "Cardiometabolic & Glucose Regulation",
            "tests": ["Fasting Insulin", "Fasting Glucose & 2-Hour OGTT", "HbA1c", "Lipid Profile (HDL, LDL, Triglycerides)"],
            "purpose": "Identifies insulin resistance and compensatory hyperinsulinemia present in 70%+ of PCOS phenotypes."
        },
        {
            "name": "Thyroid & Pituitary Screen",
            "tests": ["TSH (Thyroid-Stimulating Hormone)", "Free T4", "Prolactin (PRL)"],
            "purpose": "Rules out primary hypothyroidism and hyperprolactinemia which mimic PCOS oligomenorrhea."
        }
    ]

    lifestyle_pillars = [
        {
            "title": "Low Glycemic Index & Balanced Nutrition",
            "icon": "apple-alt",
            "advice": "Prioritize high-fiber, low-glycemic complex carbohydrates (legumes, leafy greens, quinoa) combined with high-quality protein and healthy monounsaturated fats (olive oil, avocados) to stabilize postprandial glycemic spikes."
        },
        {
            "title": "Progressive Resistance & Aerobic Training",
            "icon": "dumbbell",
            "advice": "Incorporate 150 minutes of moderate-intensity cardiovascular exercise plus 2-3 sessions of progressive resistance training weekly to enhance peripheral insulin receptor sensitivity and GLUT-4 translocation in skeletal muscle."
        },
        {
            "title": "Circadian Rhythm & Sleep Architecture",
            "icon": "moon",
            "advice": "Aim for 7–9 hours of uninterrupted sleep. Obstructive sleep apnea (OSA) is up to 5-10x more prevalent in PCOS populations; screening is advised if daytime somnolence or snoring is reported."
        },
        {
            "title": "Symptom & Ovulation Tracking",
            "icon": "calendar-check",
            "advice": "Track cycle frequency, bleeding duration, and premenstrual symptoms using a mobile journal or basal body temperature monitoring to provide longitudinal data for your physician."
        }
    ]

    questions_for_doctor = [
        "Based on my ultrasound scan and symptoms, do my ovaries show classic polycystic morphology (>= 20 follicles or volume >= 10 mL)?",
        "Which hormonal panels (Testosterone, DHEAS, LH/FSH, AMH) would you recommend to evaluate Rotterdam criteria?",
        "Should we test for metabolic markers such as insulin resistance (OGTT, fasting insulin) or dyslipidemia?",
        "Are my menstrual cycles considered ovulatory, and what lifestyle modifications best support regular menses?",
        "If I am planning pregnancy in the future, what steps should I take now to protect ovarian reserve and endometrial health?"
    ]

    return {
        "disclaimer": base_disclaimer,
        "diagnostic_panels": diagnostic_panels,
        "lifestyle_pillars": lifestyle_pillars,
        "doctor_checklist": questions_for_doctor
    }
