#!/usr/bin/env bash
# prune_repo.sh
# Quickly cleans up obvious artifacts, temporary logs, screenshots, and cache folders
# from the root directory to reduce repository footprint for AI context.

set -e

echo "Starting repository cleanup..."

# Remove Python cache and build artifacts
echo "Removing __pycache__ directories and .egg-info..."
find . -type d -name "__pycache__" -exec rm -rf {} +
find . -type d -name "*.egg-info" -exec rm -rf {} +
find . -type d -name ".pytest_cache" -exec rm -rf {} +
find . -type d -name ".ruff_cache" -exec rm -rf {} +

# Remove root-level log files and raw outputs
echo "Removing root-level *.log and output *.txt files..."
find . -maxdepth 1 -type f -name "*.log" -delete
find . -maxdepth 1 -type f -name "*_output*.txt" -delete
find . -maxdepth 1 -type f -name "install_log_*.txt" -delete
find . -maxdepth 1 -type f -name "final_check_*.txt" -delete
find . -maxdepth 1 -type f -name "test_output*.txt" -delete

# Remove verification screenshots and images from the root and specific folders
echo "Removing root-level images and verification directories..."
find . -maxdepth 1 -type f -name "*.png" -delete
rm -rf verification_images/
rm -rf verification_screenshots/

echo "Cleanup complete. The repository size has been reduced."
