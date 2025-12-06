#!/bin/bash
# scripts/setup_and_run.sh
# One-click startup script for Adam v23.5

set -e

echo "=================================================="
echo "   ADAM v23.5 | Autonomous Financial Analyst"
echo "=================================================="

# 1. Environment Check
if ! command -v python3 &> /dev/null; then
    echo "Error: python3 is not installed."
    exit 1
fi

# 2. Virtual Environment
if [ ! -d "venv" ]; then
    echo "[*] Creating virtual environment..."
    python3 -m venv venv
fi

echo "[*] Activating virtual environment..."
source venv/bin/activate

# 3. Dependencies
echo "[*] Checking dependencies..."
pip install --upgrade pip
# We use a lighter requirements set for the demo to speed it up if full reqs are huge
# But strictly we should use requirements.txt.
# For now, we install essential UI deps if missing, then the rest.
pip install Flask Flask-Cors flask-socketio requests pyyaml colorama > /dev/null
# If you want full AI capabilities, uncomment below:
# pip install -r requirements.txt

# 4. Launch
echo "=================================================="
echo "   Starting Unified Server..."
echo "   UI: http://localhost:5000"
echo "   API: http://localhost:5000/api"
echo "=================================================="

export PYTHONPATH=$PYTHONPATH:$(pwd)
python3 core/api/server.py
