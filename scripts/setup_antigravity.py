#!/usr/bin/env python3
"""
Antigravity Environment & Workspace Setup Runner.
Initializes default folders, validates dependencies, and scaffolds extensions/plugins.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def run_setup(check_only: bool = False) -> int:
    workspace = Path.cwd()
    print("=" * 60)
    print("Antigravity System Setup & Environment Verifier")
    print("=" * 60)

    # 1. Directory Structure
    required_dirs = [
        workspace / ".antigravity",
        workspace / ".antigravity" / "plugins",
        workspace / ".antigravity" / "extensions",
        workspace / ".antigravity" / "runtimes",
    ]

    print("\n[1/3] Verifying / Creating directories:")
    for d in required_dirs:
        if not d.exists():
            if check_only:
                print(f"  [MISSING] {d.relative_to(workspace)}")
            else:
                d.mkdir(parents=True, exist_ok=True)
                print(f"  [CREATED] {d.relative_to(workspace)}")
        else:
            print(f"  [OK]      {d.relative_to(workspace)}")

    # 2. Module imports check
    print("\n[2/3] Verifying Antigravity SDK modules:")
    try:
        import src.antigravity_sdk  # noqa: F401
        print("  [OK] Antigravity SDK Core Modules imported successfully.")
    except Exception as e:
        print(f"  [FAIL] Failed importing Antigravity SDK: {e}")
        return 1

    # 3. Native Layer 0 Gate check
    print("\n[3/3] Verifying Native Layer 0 Rust PDIL Gatekeeper:")
    try:
        from rust_ext import RustPDILGatekeeper
        _ = RustPDILGatekeeper()
        print("  [OK] RustPDILGatekeeper native extension loaded and ready.")
    except Exception as e:
        print(f"  [WARNING] RustPDILGatekeeper native extension not loaded: {e}")

    print("\nAntigravity setup completed successfully!")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Antigravity Setup and Initialization")
    parser.add_argument("--check", action="store_true", help="Perform checks only without modifying disk")
    args = parser.parse_args()
    return run_setup(check_only=args.check)


if __name__ == "__main__":
    sys.exit(main())
