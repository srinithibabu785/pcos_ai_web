"""
Flask Web Application for PCOS AI — Explainable Prediction & Decision Support.
Connects verified trained models (EfficientNetB0, DenseNet121, Ensemble, and Tabular RF)
to a modern, accessible healthcare AI dashboard.
"""

import os
import sys
import json
import uuid
from pathlib import Path
from flask import (
    Flask, render_template, request, jsonify, redirect, url_for,
    send_from_directory, send_file, flash
)
from werkzeug.utils import secure_filename

# Add project root to sys.path
WEB_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = WEB_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pcos_ai_web.services.model_loader import ModelManager
from pcos_ai_web.services.prediction_service import process_and_predict, validate_uploaded_image
from pcos_ai_web.services.whatif_service import simulate_whatif, DEFAULT_PROFILE
from pcos_ai_web.services.recommendation_service import get_clinical_guidance
from pcos_ai_web.services.report_service import generate_pdf_report
from pcos_ai_web.services.risk_service import RESPONSIBLE_AI_DISCLAIMER

app = Flask(
    __name__,
    template_folder=str(WEB_DIR / "templates"),
    static_folder=str(WEB_DIR / "static")
)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "pcos_ai_secure_academic_key_2026")
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB max

UPLOAD_DIR = WEB_DIR / "uploads"
REPORT_DIR = WEB_DIR / "reports"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

# In-memory store for recent session reports
SESSION_STORE = {}

# Initialize models once when app starts
with app.app_context():
    manager = ModelManager.get_instance()

# ------------------------------------------------------------------------------
# 1. Health Hub (Dashboard)
# ------------------------------------------------------------------------------
@app.route("/")
def index():
    status = manager.model_status
    sample_images = [
        {"name": "Normal Ovary Scan (Sample 1)", "path": "/static/images/samples/sample_normal_1.jpg", "type": "Healthy"},
        {"name": "Normal Ovary Scan (Sample 2)", "path": "/static/images/samples/sample_normal_2.jpg", "type": "Healthy"},
        {"name": "Polycystic Ovary Scan (Sample 1)", "path": "/static/images/samples/sample_pcos_1.jpg", "type": "PCOS"},
        {"name": "Polycystic Ovary Scan (Sample 2)", "path": "/static/images/samples/sample_pcos_2.jpg", "type": "PCOS"},
    ]
    return render_template(
        "index.html",
        status=status,
        samples=sample_images,
        disclaimer=RESPONSIBLE_AI_DISCLAIMER
    )

# ------------------------------------------------------------------------------
# 2. AI Check (Ultrasound Prediction Interface)
# ------------------------------------------------------------------------------
@app.route("/ai-check", methods=["GET", "POST"])
def ai_check():
    if request.method == "POST":
        # Check if sample image was selected or custom file uploaded
        sample_path = request.form.get("sample_path")
        file = request.files.get("ultrasound_image")

        try:
            if sample_path:
                full_sample_path = WEB_DIR / sample_path.lstrip("/")
                if not full_sample_path.exists():
                    flash("Selected sample image file could not be found.", "error")
                    return redirect(url_for("ai_check"))
                filename = full_sample_path.name
                result = process_and_predict(full_sample_path, filename=filename, generate_xai=True)
            else:
                is_valid, err_msg = validate_uploaded_image(file)
                if not is_valid:
                    flash(err_msg, "error")
                    return redirect(url_for("ai_check"))

                filename = secure_filename(file.filename) or f"scan_{uuid.uuid4().hex[:6]}.jpg"
                save_path = UPLOAD_DIR / f"{uuid.uuid4().hex[:8]}_{filename}"
                file.save(str(save_path))
                result = process_and_predict(save_path, filename=filename, generate_xai=True)

            # Store in session store
            req_id = result["request_id"]
            SESSION_STORE[req_id] = result

            return render_template("prediction.html", result=result, disclaimer=RESPONSIBLE_AI_DISCLAIMER)

        except Exception as e:
            flash(f"Inference pipeline encountered an error: {str(e)}", "error")
            return redirect(url_for("ai_check"))

    # GET request
    sample_images = [
        {"name": "Normal Ultrasound #1", "path": "static/images/samples/sample_normal_1.jpg", "type": "Healthy"},
        {"name": "Normal Ultrasound #2", "path": "static/images/samples/sample_normal_2.jpg", "type": "Healthy"},
        {"name": "PCOS Ultrasound #1", "path": "static/images/samples/sample_pcos_1.jpg", "type": "PCOS"},
        {"name": "PCOS Ultrasound #2", "path": "static/images/samples/sample_pcos_2.jpg", "type": "PCOS"}
    ]
    return render_template(
        "prediction.html",
        result=None,
        samples=sample_images,
        disclaimer=RESPONSIBLE_AI_DISCLAIMER
    )

# ------------------------------------------------------------------------------
# 3. AI Vision (Grad-CAM Explainability)
# ------------------------------------------------------------------------------
@app.route("/ai-vision")
def ai_vision():
    req_id = request.args.get("id")
    result = SESSION_STORE.get(req_id) if req_id else None
    
    # If no recent prediction, get latest or show educational overview
    if not result and SESSION_STORE:
        result = list(SESSION_STORE.values())[-1]

    return render_template("explainability.html", result=result, disclaimer=RESPONSIBLE_AI_DISCLAIMER)

# ------------------------------------------------------------------------------
# 4. Wellness Guide
# ------------------------------------------------------------------------------
@app.route("/wellness")
def wellness():
    guidance = get_clinical_guidance("Moderate")
    return render_template("wellness.html", guidance=guidance, disclaimer=RESPONSIBLE_AI_DISCLAIMER)

# ------------------------------------------------------------------------------
# 5. Care Compass (Clinical Evaluation Guide)
# ------------------------------------------------------------------------------
@app.route("/care-compass")
def care_compass():
    guidance = get_clinical_guidance("High")
    return render_template("care_compass.html", guidance=guidance, disclaimer=RESPONSIBLE_AI_DISCLAIMER)

# ------------------------------------------------------------------------------
# 6. What-If Lab (Tabular Model Simulation)
# ------------------------------------------------------------------------------
@app.route("/what-if", methods=["GET", "POST"])
def what_if():
    if request.method == "POST":
        # Form submission or JSON
        data = request.form.to_dict() if request.form else (request.get_json() or {})
        sim_result = simulate_whatif(data)
        if request.is_json:
            return jsonify(sim_result)
        return render_template("whatif.html", sim=sim_result, defaults=DEFAULT_PROFILE, disclaimer=RESPONSIBLE_AI_DISCLAIMER)

    # Initial GET with default simulation
    initial_sim = simulate_whatif({})
    return render_template("whatif.html", sim=initial_sim, defaults=DEFAULT_PROFILE, disclaimer=RESPONSIBLE_AI_DISCLAIMER)

@app.route("/api/whatif", methods=["POST"])
def api_whatif():
    data = request.get_json() or {}
    sim_result = simulate_whatif(data)
    return jsonify(sim_result)

# ------------------------------------------------------------------------------
# 7. Model Lab (Empirical Comparison & Benchmarks)
# ------------------------------------------------------------------------------
@app.route("/model-lab")
def model_lab():
    # Load verified model comparison metrics
    comparison_csv = PROJECT_ROOT / "reports" / "final_results.csv"
    fallback_csv = PROJECT_ROOT / "results" / "model_comparison.csv"
    
    metrics_list = []
    if comparison_csv.exists():
        import pandas as pd
        df = pd.read_csv(comparison_csv)
        metrics_list = df.to_dict(orient="records")
    elif fallback_csv.exists():
        import pandas as pd
        df = pd.read_csv(fallback_csv)
        metrics_list = df.to_dict(orient="records")

    status = manager.model_status
    return render_template(
        "model_lab.html",
        metrics=metrics_list,
        status=status,
        disclaimer=RESPONSIBLE_AI_DISCLAIMER
    )

# ------------------------------------------------------------------------------
# 8. My AI Report (View & Download PDF)
# ------------------------------------------------------------------------------
@app.route("/report/<request_id>")
def view_report(request_id):
    result = SESSION_STORE.get(request_id)
    if not result:
        flash("Report session expired or identifier not found.", "warning")
        return redirect(url_for("ai_check"))
    return render_template("report.html", result=result, disclaimer=RESPONSIBLE_AI_DISCLAIMER)

@app.route("/download-report/<request_id>")
def download_report(request_id):
    result = SESSION_STORE.get(request_id)
    if not result:
        flash("Report record not found. Please run an ultrasound prediction first.", "warning")
        return redirect(url_for("ai_check"))

    try:
        pdf_file = generate_pdf_report(result)
        return send_file(
            pdf_file,
            as_attachment=True,
            download_name=f"PCOS_AI_Report_{request_id}.pdf",
            mimetype="application/pdf"
        )
    except Exception as e:
        flash(f"Failed to generate PDF report: {str(e)}", "error")
        return redirect(url_for("view_report", request_id=request_id))

# ------------------------------------------------------------------------------
# 9. API Status Endpoint
# ------------------------------------------------------------------------------
@app.route("/api/status")
def api_status():
    return jsonify({
        "status": "healthy",
        "models": manager.model_status,
        "class_mapping": manager.class_names,
        "disclaimer": RESPONSIBLE_AI_DISCLAIMER
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"\n========================================================")
    print(f"PCOS AI WEB SERVER RUNNING AT: http://127.0.0.1:{port}")
    print(f"========================================================\n")
    app.run(host="127.0.0.1", port=port, debug=False)
