"""MPLADS AI Monitoring System - Application Entry Point
"""

import sys
from pathlib import Path

# Add workspace directory to path
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from app import create_app

app = create_app()

if __name__ == "__main__":
    print("=" * 70)
    print("  MPLADS AI - Data Anomaly and Fraud Risk Detection System")
    print("  Academic Prototype (B.Tech CSE Data Science)")
    print("  Server starting on: http://127.0.0.1:5000")
    print("=" * 70)
    app.run(host="127.0.0.1", port=5000, debug=True)
