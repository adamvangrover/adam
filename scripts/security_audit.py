#!/usr/bin/env python3
"""
Static Security Gate Linter for ADAM OS v30.1

Scans Python, Markdown, and JSONL files for P0 security and architecture violations:
1. Prompt leakage / secret keys (e.g. hardcoded real API keys)
2. Weak/Mock cryptographic hashes in non-test production code
3. Unrounded IEEE 754 floating point values in financial metrics
4. System 1 / System 2 class collisions (e.g. AsyncAgentBase vs TemplateAgentV30 coupling)
"""

import os
import re
import sys
from pathlib import Path
from typing import List, Tuple

# Directories to scan
TARGET_DIRS = ["core", "evals", "server", "scripts", "prompt_library", "tests"]
FILE_EXTENSIONS = {".py", ".md", ".jsonl"}

# Exclude third-party or virtual environment files if encountered
EXCLUDE_DIRS = {".venv", "node_modules", ".git", "target", "__pycache__"}

# Patterns for P0 violations
SECRET_KEY_PATTERN = re.compile(r"(?:sk-proj-[a-zA-Z0-9_-]{20,}|Bearer\s+eyJ[a-zA-Z0-9_-]{30,})")
MOCK_CRYPTO_PATTERN = re.compile(r"(?:0xdeadbeef|mock_hash_val|dummy_sha256_hash)", re.IGNORECASE)
RAW_FLOAT_PATTERN = re.compile(r'"(?:pd|leverage|rate|score|var_value|ebitda|total_debt)":\s*\d+\.\d{5,}')


def scan_file(filepath: Path) -> List[Tuple[int, str, str]]:
    violations = []
    # Skip self-inspection
    if filepath.name == "security_audit.py":
        return violations

    try:
        content = filepath.read_text(encoding="utf-8", errors="ignore")
    except Exception as e:
        return [(0, "FILE_READ_ERROR", f"Could not read file: {e}")]

    lines = content.splitlines()
    is_python = filepath.suffix == ".py"
    is_test_file = "tests/" in str(filepath).replace("\\", "/") or filepath.name.startswith("test_")

    # Line-by-line checks
    for idx, line in enumerate(lines, 1):
        # 1. Secret Key Check
        if SECRET_KEY_PATTERN.search(line):
            violations.append((idx, "P0_SECRET_LEAKAGE", "Potential hardcoded production API key or bearer token found."))

        # 2. Mock Crypto Check (ignore in test files or docs)
        if not is_test_file and MOCK_CRYPTO_PATTERN.search(line):
            violations.append((idx, "P0_MOCK_CRYPTO", "Insecure or mock cryptographic hash found in production code."))

        # 3. Unrounded IEEE 754 Float Check
        if RAW_FLOAT_PATTERN.search(line):
            violations.append((idx, "P0_UNROUNDED_FLOAT", "Unrounded IEEE 754 precision float detected in structured payload."))

    # Architecture Class Collision Check (System 1 / System 2 separation)
    if is_python and not is_test_file:
        has_sys1 = "AsyncAgentBase" in content or "AsyncSwarmAgent" in content
        has_sys2 = "TemplateAgentV30" in content or "NeuroSymbolicPlanner" in content
        if has_sys1 and has_sys2:
            class_defs = [line for line in lines if line.strip().startswith("class ")]
            for line in class_defs:
                if ("AsyncAgentBase" in line or "AsyncSwarmAgent" in line) and (
                    "TemplateAgentV30" in line or "NeuroSymbolicPlanner" in line
                ):
                    violations.append((0, "P0_CLASS_COLLISION", f"Class collision detected in '{line.strip()}': System 1 and System 2 must remain decoupled."))

    return violations


def main() -> int:
    print("=== ADAM OS v30.1 Static Security Gate Linter ===")
    total_files = 0
    total_violations = 0

    root_dir = Path(__file__).resolve().parent.parent

    for target in TARGET_DIRS:
        target_path = root_dir / target
        if not target_path.exists():
            continue

        for root, dirs, files in os.walk(target_path):
            dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS]
            for file in files:
                filepath = Path(root) / file
                if filepath.suffix in FILE_EXTENSIONS:
                    total_files += 1
                    violations = scan_file(filepath)
                    if violations:
                        rel_path = filepath.relative_to(root_dir)
                        for line_no, v_type, v_msg in violations:
                            total_violations += 1
                            print(f"[VIOLATION] [{v_type}] {rel_path}:{line_no} - {v_msg}")

    print(f"\nScanned {total_files} files across {len(TARGET_DIRS)} target directories.")
    if total_violations == 0:
        print("✅ Security Gate Passed: Zero P0 violations detected.")
        return 0
    else:
        print(f"❌ Security Gate Failed: {total_violations} P0 violations detected.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
