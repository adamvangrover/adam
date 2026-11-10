#!/usr/bin/env python3
"""
Executable entrypoint script for Antigravity CLI.
"""

from pathlib import Path
import sys

# Add repository root to PYTHONPATH
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.antigravity_cli import main

if __name__ == "__main__":
    sys.exit(main())
