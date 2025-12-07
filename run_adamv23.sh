#!/bin/bash

# run_adamv23.sh - ADAM v23.5 Environment Initializer & Runner

echo "=================================================="
echo "   ADAM v23.5 | AUTONOMOUS FINANCIAL ANALYST      "
echo "=================================================="
echo "Initializing environment..."

# 1. Check for Dependencies
if ! command -v python3 &> /dev/null; then
    echo "[ERROR] Python 3 is not installed."
    exit 1
fi

# 2. Setup Virtual Environment (Optional, if not present)
if [ ! -d "venv" ]; then
    echo "[INFO] Creating virtual environment..."
    python3 -m venv venv
fi
source venv/bin/activate 2>/dev/null || echo "[INFO] Running in current environment (no venv activated)"

# 3. Install Dependencies (Minimal check)
if [ -f "requirements.txt" ]; then
    echo "[INFO] Checking dependencies..."
    # pip install -r requirements.txt # Commented out to avoid auto-install delays in demo
fi

# 4. Deploy Dashboard for GitHub Pages
# Copies the dashboard to specific subfolders to allow deep linking or clean URLs
echo "[INFO] Deploying Web Portal..."
mkdir -p showcase/portal
cp showcase/dashboard.html showcase/portal/index.html
echo "[SUCCESS] Dashboard deployed to showcase/portal/index.html"

# 5. MCP Server Launch Helper
function run_server() {
    echo "[INFO] Starting MCP Server..."
    export PYTHONPATH=$PYTHONPATH:$(pwd)
    python3 core/vertical_risk_agent/tools/mcp_server/server.py
}

# 6. CLI Argument Parsing
if [ "$1" == "--server" ]; then
    run_server
    exit 0
fi

if [ "$1" == "--deploy" ]; then
    echo "[INFO] Deployment artifacts ready."
    exit 0
fi

echo ""
echo "Setup Complete."
echo "--------------------------------------------------"
echo "Usage:"
echo "  ./run_adamv23.sh --server   : Start the MCP Server"
echo "  ./run_adamv23.sh --deploy   : Prepare static files"
echo "  ./run_adamv23.sh            : Show this help"
echo ""
echo "Access the Web Portal at: showcase/dashboard.html"
echo "--------------------------------------------------"
