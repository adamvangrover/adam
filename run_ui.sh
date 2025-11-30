#!/bin/bash
echo "Generating UI Data..."
python3 scripts/generate_ui_data.py

echo "Starting UI Backend Server..."
echo "Access the UI at http://localhost:5000"
python3 services/ui_backend.py
