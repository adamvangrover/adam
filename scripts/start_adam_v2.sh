#!/bin/bash
set -e

echo "Starting Adam v2.0 Initialization..."

# Run interactive setup if .env is missing
if [ ! -f .env ]; then
    python3 scripts/setup_interactive.py
fi

# Ensure dependencies (simplified check)
if [ ! -d "venv" ] && [ -z "$VIRTUAL_ENV" ]; then
    echo "Warning: No virtual environment detected. Use 'make install' to set up."
fi

# Start the Core System
echo "Launching Adam Core..."
python3 scripts/run_adam.py
