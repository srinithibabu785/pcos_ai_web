"""
Launcher Script for PCOS AI Web Application.
Runs the Flask server locally on http://127.0.0.1:5000
"""

import os
import sys
from pathlib import Path

# Ensure web app directory is in sys.path
WEB_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = WEB_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pcos_ai_web.app import app

def main():
    port = int(os.environ.get("PORT", 5000))
    print("=" * 70)
    print("     PCOS AI — EXPLAINABLE CLINICAL DECISION SUPPORT DASHBOARD     ")
    print("=" * 70)
    print(f"Backend Server:   Flask")
    print(f"Local URL:        http://127.0.0.1:{port}")
    print(f"Verified Models:  EfficientNetB0, DenseNet121, Ensemble, Tabular RF")
    print(f"Status:           Ready for Browser Interaction")
    print("=" * 70)
    print("Press CTRL+C in this terminal to stop the application.\n")
    
    app.run(host="127.0.0.1", port=port, debug=False)

if __name__ == "__main__":
    main()
