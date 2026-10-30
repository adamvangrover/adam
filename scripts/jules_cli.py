#!/usr/bin/env python3
"""
Jules CLI Entrypoint Script.
CLI runner for autonomous agent execution, planning, REPL, and web IDE server.
"""

import sys
from pathlib import Path

# Ensure repo root is in python path
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from src.jules.cli import main

if __name__ == "__main__":
    sys.exit(main())
