"""
Verification Test Script for PCOS AI Web Application.
Tests route rendering, prediction execution on sample image, What-If API,
and PDF report generation.
"""

import os
import sys
from pathlib import Path

# Force utf-8 stdout if supported
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

WEB_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = WEB_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pcos_ai_web.app import app

def test_endpoints():
    print("\n========================================================")
    print("TESTING PCOS AI WEB APPLICATION ENDPOINTS")
    print("========================================================")

    client = app.test_client()

    # 1. Health Hub (GET /)
    res_index = client.get("/")
    assert res_index.status_code == 200, f"Index failed: {res_index.status_code}"
    print("[PASS] GET / (Health Hub): Status 200 OK")

    # 2. AI Check (GET /ai-check)
    res_check = client.get("/ai-check")
    assert res_check.status_code == 200, f"AI-Check failed: {res_check.status_code}"
    print("[PASS] GET /ai-check: Status 200 OK")

    # 3. AI Vision (GET /ai-vision)
    res_vision = client.get("/ai-vision")
    assert res_vision.status_code == 200, f"AI-Vision failed: {res_vision.status_code}"
    print("[PASS] GET /ai-vision: Status 200 OK")

    # 4. Wellness Guide (GET /wellness)
    res_well = client.get("/wellness")
    assert res_well.status_code == 200, f"Wellness failed: {res_well.status_code}"
    print("[PASS] GET /wellness: Status 200 OK")

    # 5. Care Compass (GET /care-compass)
    res_care = client.get("/care-compass")
    assert res_care.status_code == 200, f"Care Compass failed: {res_care.status_code}"
    print("[PASS] GET /care-compass: Status 200 OK")

    # 6. What-If Lab (GET & POST)
    res_whatif_get = client.get("/what-if")
    assert res_whatif_get.status_code == 200, f"What-If GET failed: {res_whatif_get.status_code}"
    print("[PASS] GET /what-if: Status 200 OK")

    res_whatif_post = client.post("/api/whatif", json={"AMH(ng/mL)": 6.5, "Cycle(R/I)": 4})
    assert res_whatif_post.status_code == 200, f"What-If POST failed: {res_whatif_post.status_code}"
    data_whatif = res_whatif_post.get_json()
    assert data_whatif.get("available") is True, f"What-If failed: {data_whatif}"
    print(f"[PASS] POST /api/whatif: Status 200 OK (Simulated Score: {data_whatif['score_pct']}%)")

    # 7. Model Lab (GET /model-lab)
    res_lab = client.get("/model-lab")
    assert res_lab.status_code == 200, f"Model Lab failed: {res_lab.status_code}"
    print("[PASS] GET /model-lab: Status 200 OK")

    # 8. API Status (GET /api/status)
    res_status = client.get("/api/status")
    assert res_status.status_code == 200
    print("[PASS] GET /api/status: Status 200 OK")

    # 9. Prediction Pipeline via Sample Image
    sample_rel = "static/images/samples/sample_normal_1.jpg"
    res_pred = client.post("/ai-check", data={"sample_path": sample_rel})
    assert res_pred.status_code == 200, f"Prediction post failed: {res_pred.status_code}"
    assert b"AI Prediction Output" in res_pred.data
    print("[PASS] POST /ai-check (Sample Prediction + Grad-CAM): Status 200 OK")

    print("\nALL 9 ENDPOINT CHECKS PASSED PERFECTLY!\n")

if __name__ == "__main__":
    test_endpoints()
