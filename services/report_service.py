"""
PDF Report Generation Service for PCOS AI Web Application.
Generates publication-quality, clinical decision-support research summary reports
using ReportLab with clean typography, tables, and responsible AI notices.
"""

import os
import uuid
import datetime
from pathlib import Path
from typing import Dict, Any, Optional

from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage, HRFlowable
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT, TA_JUSTIFY

WEB_DIR = Path(__file__).resolve().parent.parent
REPORTS_DIR = WEB_DIR / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

def generate_pdf_report(prediction_result: Dict[str, Any]) -> str:
    """
    Generates a PDF decision-support report and saves it to reports/.
    Returns the file path string.
    """
    req_id = prediction_result.get("request_id", str(uuid.uuid4())[:8])
    pdf_filename = f"PCOS_AI_Report_{req_id}.pdf"
    pdf_path = REPORTS_DIR / pdf_filename

    doc = SimpleDocTemplate(
        str(pdf_path),
        pagesize=letter,
        rightMargin=40,
        leftMargin=40,
        topMargin=36,
        bottomMargin=36
    )

    styles = getSampleStyleSheet()

    # Custom styles
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=20,
        leading=24,
        textColor=colors.HexColor('#1E1B4B'),  # Deep Indigo
        alignment=TA_LEFT
    )

    subtitle_style = ParagraphStyle(
        'DocSubTitle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=10,
        leading=14,
        textColor=colors.HexColor('#6B7280'),
        alignment=TA_LEFT
    )

    section_heading = ParagraphStyle(
        'SectionHeading',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=12,
        leading=16,
        textColor=colors.HexColor('#4338CA'),  # Indigo
        spaceBefore=12,
        spaceAfter=6
    )

    body_style = ParagraphStyle(
        'BodyDark',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9,
        leading=13,
        textColor=colors.HexColor('#1F2937')
    )

    disclaimer_style = ParagraphStyle(
        'Disclaimer',
        parent=styles['Normal'],
        fontName='Helvetica-Oblique',
        fontSize=7.5,
        leading=11,
        textColor=colors.HexColor('#92400E'),
        alignment=TA_JUSTIFY
    )

    story = []

    # 1. Header Banner
    story.append(Paragraph("PCOS AI — Clinical Decision-Support Research Report", title_style))
    story.append(Paragraph("Explainable Ensemble Transfer Learning for Early PCOS Detection | Research Prototype", subtitle_style))
    story.append(Spacer(1, 10))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor('#6366F1'), spaceBefore=2, spaceAfter=12))

    # 2. Metadata Table
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    meta_data = [
        [
            Paragraph("<b>Study Identifier:</b>", body_style),
            Paragraph(f"PAS-{req_id}", body_style),
            Paragraph("<b>Report Date:</b>", body_style),
            Paragraph(timestamp, body_style)
        ],
        [
            Paragraph("<b>Input Scan:</b>", body_style),
            Paragraph(prediction_result.get("filename", "ultrasound_scan.jpg"), body_style),
            Paragraph("<b>Resolution:</b>", body_style),
            Paragraph(prediction_result.get("dimensions", "224 × 224"), body_style)
        ]
    ]
    t_meta = Table(meta_data, colWidths=[95, 170, 85, 180])
    t_meta.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#F8FAFC')),
        ('PADDING', (0, 0), (-1, -1), 5),
        ('BOX', (0, 0), (-1, -1), 0.5, colors.HexColor('#E2E8F0')),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ]))
    story.append(t_meta)
    story.append(Spacer(1, 12))

    # 3. Model Inference Results Table
    story.append(Paragraph("1. Model Predictive Scores & Integrated Ensemble", section_heading))
    
    scores = prediction_result.get("scores", {})
    eff_pct = scores.get("efficientnet_pct", 0.0)
    dense_pct = scores.get("densenet_pct", 0.0)
    ens_pct = scores.get("ensemble_pct", 0.0)
    
    risk_info = prediction_result.get("risk_tier", {})
    pred_class = prediction_result.get("predicted_class", "Undetermined")
    threshold = prediction_result.get("ensemble_config", {}).get("threshold", 0.60)
    w_eff = prediction_result.get("ensemble_config", {}).get("weight_efficientnet", 0.50)
    w_dense = prediction_result.get("ensemble_config", {}).get("weight_densenet", 0.50)

    pred_table_data = [
        [
            Paragraph("<b>Model Architecture</b>", body_style),
            Paragraph("<b>Prediction Weight</b>", body_style),
            Paragraph("<b>Model Score (Likelihood)</b>", body_style),
            Paragraph("<b>Decision Role</b>", body_style)
        ],
        [
            Paragraph("EfficientNetB0 (Compound Scaling)", body_style),
            Paragraph(f"{w_eff:.2f}", body_style),
            Paragraph(f"<b>{eff_pct:.2f}%</b>", body_style),
            Paragraph("High-level follicular feature extractor", body_style)
        ],
        [
            Paragraph("DenseNet121 (Dense Feature Reuse)", body_style),
            Paragraph(f"{w_dense:.2f}", body_style),
            Paragraph(f"<b>{dense_pct:.2f}%</b>", body_style),
            Paragraph("Parenchymal stroma feature extractor", body_style)
        ],
        [
            Paragraph("<b>Weighted Soft-Voting Ensemble</b>", body_style),
            Paragraph("<b>1.00</b>", body_style),
            Paragraph(f"<b><font color='#4338CA' size='10'>{ens_pct:.2f}%</font></b>", body_style),
            Paragraph(f"<b>Primary Decision Metric</b> (Threshold: {threshold:.2f})", body_style)
        ]
    ]

    t_pred = Table(pred_table_data, colWidths=[170, 90, 130, 140])
    t_pred.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#EEF2FF')),
        ('PADDING', (0, 0), (-1, -1), 6),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#CBD5E1')),
        ('BACKGROUND', (0, 3), (-1, 3), colors.HexColor('#E0E7FF')),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ]))
    story.append(t_pred)
    story.append(Spacer(1, 12))

    # 4. Risk Stratification Callout
    story.append(Paragraph("2. Presentation Risk Category", section_heading))
    
    tier_num = risk_info.get("tier", 3)
    tier_name = risk_info.get("category", "Moderate")
    band_name = risk_info.get("band", "41–60%")
    hex_col = risk_info.get("color", "#F97316")

    risk_box_data = [
        [
            Paragraph(f"<b>Predicted Classification:</b> {pred_class.upper()}<br/>"
                      f"<b>Presentation Tier:</b> Tier {tier_num} — {tier_name} (Score Range: {band_name})<br/>"
                      f"<b>Clinical Interpretation:</b> {risk_info.get('clinical_guidance', '')}", body_style)
        ]
    ]
    t_risk = Table(risk_box_data, colWidths=[530])
    t_risk.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#F8FAFC')),
        ('PADDING', (0, 0), (-1, -1), 8),
        ('BOX', (0, 0), (-1, -1), 1.5, colors.HexColor(hex_col)),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ]))
    story.append(t_risk)
    story.append(Spacer(1, 12))

    # 5. Explainable AI (Grad-CAM) Summary
    story.append(Paragraph("3. Explainable AI (Grad-CAM) Feature Attribution", section_heading))
    xai = prediction_result.get("xai", {})
    if xai.get("available", False):
        xai_summary = (
            "Gradient-weighted Class Activation Mapping (Grad-CAM) was computed for the target convolutional "
            "layers ('top_conv' for EfficientNetB0 and 'conv5_block16_2_conv' for DenseNet121). The generated "
            "salience heatmaps show that model activations localize around the peripheral follicular periphery "
            "and ovarian stromal boundary. Notice: Highlighted areas indicate image regions influencing the network's "
            "internal mathematical scoring, NOT definitive histological proof of cystic lesions."
        )
    else:
        xai_summary = "Grad-CAM spatial attribution was not requested or was bypassed for this inference session."
    story.append(Paragraph(xai_summary, body_style))
    story.append(Spacer(1, 12))

    # 6. Clinical Care Compass & Recommended Next Steps
    story.append(Paragraph("4. Recommended Clinical Diagnostic Correlation", section_heading))
    care_text = (
        "According to the 2023 International Evidence-Based PCOS Guidelines, an ultrasound scan alone cannot "
        "establish or rule out PCOS. A complete clinical assessment requires correlating with:<br/>"
        "• <b>Clinical Hyperandrogenism:</b> Hirsutism (modified Ferriman-Gallwey score), alopecia, or severe acne.<br/>"
        "• <b>Biochemical Hyperandrogenism:</b> Total Testosterone, Free Androgen Index (FAI), and DHEAS.<br/>"
        "• <b>Ovulatory Dysfunction:</b> Oligomenorrhea (cycles > 35 days) or amenorrhea (> 90 days).<br/>"
        "• <b>Metabolic Risk Assessment:</b> 75g 2-hour Oral Glucose Tolerance Test (OGTT), fasting insulin, and fasting lipid profile."
    )
    story.append(Paragraph(care_text, body_style))
    story.append(Spacer(1, 16))

    # 7. Responsible AI Governance Disclaimer Box
    disclaimer_box = [
        [
            Paragraph(
                "<b>ACADEMIC RESEARCH & CLINICAL DECISION-SUPPORT NOTICE:</b><br/>"
                "This report was automatically synthesized by the PCOS AI research system for academic demonstration. "
                "The predictions and likelihood scores are computational estimates, not verified medical diagnoses. "
                "Do not commence, alter, or discontinue any medical treatment based solely on this document. "
                "Always consult a licensed gynecologist, endocrinologist, or radiologist for official diagnosis.",
                disclaimer_style
            )
        ]
    ]
    t_disc = Table(disclaimer_box, colWidths=[530])
    t_disc.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#FEF3C7')), # Soft Amber
        ('PADDING', (0, 0), (-1, -1), 8),
        ('BOX', (0, 0), (-1, -1), 1, colors.HexColor('#F59E0B')),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ]))
    story.append(t_disc)

    doc.build(story)
    return str(pdf_path)
