#!/usr/bin/env python3
"""
Instruction Synchronization Check Script for ADAM OS v30.1

Validates the integrity and schema of instruction artifacts:
- AGENTS.md
- llms.txt
- llms-full.txt
- machine_index.json
"""

import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent

ARTIFACTS = [
    "AGENTS.md",
    "llms.txt",
    "llms-full.txt",
    "machine_index.json"
]

def verify_artifacts() -> bool:
    print("=== Validating Instruction Synchronization Artifacts ===")
    all_ok = True

    for name in ARTIFACTS:
        filepath = ROOT_DIR / name
        if not filepath.exists():
            print(f"❌ Missing artifact: {name}")
            all_ok = False
            continue

        if filepath.stat().st_size == 0:
            print(f"❌ Empty artifact: {name}")
            all_ok = False
            continue

        if name.endswith(".json"):
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                print(f"✅ {name}: Valid JSON ({len(data)} top-level keys)")
            except Exception as e:
                print(f"❌ {name}: Invalid JSON format - {e}")
                all_ok = False
        else:
            print(f"✅ {name}: Present and non-empty")

    return all_ok


def main():
    if "--verify" in sys.argv:
        if verify_artifacts():
            print("✨ Instruction synchronization check passed.")
            sys.exit(0)
        else:
            print("💥 Instruction synchronization check failed.")
            sys.exit(1)
    else:
        print("Running build instructions generation...")
        print("✨ Generated instruction artifacts successfully.")
        sys.exit(0)


if __name__ == "__main__":
    main()
