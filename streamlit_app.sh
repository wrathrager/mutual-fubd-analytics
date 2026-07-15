#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
source .venv/bin/activate
export PYTHONUNBUFFERED=1
export REQUEST_MIN_INTERVAL_SECONDS=${REQUEST_MIN_INTERVAL_SECONDS:-1.5}
streamlit run app.py --server.port 8501 --server.address 0.0.0.0 --server.headless true
