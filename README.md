# PCOS AI Web Application

A modern, responsive healthcare AI web dashboard for **Explainable PCOS Prediction and Clinical Decision Support**.

---

## 1. Features
1. **Health Hub (`/`):** Central dashboard displaying system readiness, loaded model parameters, and quick test scan library.
2. **AI Check (`/ai-check`):** Ultrasound image upload and automated inference via EfficientNetB0, DenseNet121, and the soft-voting ensemble.
3. **AI Vision (`/ai-vision`):** Genuine Grad-CAM visual heatmaps and overlays showing spatial follicular activation.
4. **Wellness Guide (`/wellness`):** Evidence-based supportive nutrition, exercise, circadian sleep, and lifestyle advice.
5. **Care Compass (`/care-compass`):** Detailed Rotterdam 2003 / 2023 guideline explanation, diagnostic lab panels, and physician consultation checklists.
6. **What-If Lab (`/what-if`):** Interactive clinical parameter simulator powered by the verified 41-feature Random Forest model (`tabular_model.pkl`).
7. **Model Lab (`/model-lab`):** Verified empirical benchmark tables, confusion matrices, ROC curves, and reliability calibration curves.
8. **My AI Report (`/report/<id>`):** Interactive summary and downloadable research PDF report with ReportLab.

---

## 2. Quick Start on Windows

### Step 1: Open Terminal in Project Directory
```powershell
cd C:\Users\srini\Documents\pcos_project
```

### Step 2: Run Application
```powershell
python pcos_ai_web/run.py
```
Or:
```powershell
python -m pcos_ai_web.app
```

### Step 3: Open in Browser
Visit: **[http://127.0.0.1:5000](http://127.0.0.1:5000)**

---

## 3. Architecture & Verified Models Reused
- **EfficientNetB0:** `models/EfficientNetB0_best.keras` (Input: 224×224×3 RGB, Target layer: `top_conv`)
- **DenseNet121:** `models/DenseNet121_best.keras` (Input: 224×224×3 RGB, Target layer: `conv5_block16_2_conv`)
- **Ensemble:** Weighted Soft-Voting with validation-locked weights and threshold $\tau = 0.60$
- **Tabular Model:** `models/tabular_model.pkl` + `models/tabular_preprocessor.pkl` (Random Forest, 41 clinical features)

---

## 4. Responsible AI Notice
This application is an **academic research prototype** and **clinical decision-support tool**. It does not produce a definitive medical diagnosis. All model outputs must be clinically correlated by a licensed medical practitioner.
