import json
import os
import re
from pathlib import Path

# Config
REPO_ROOT = Path(__file__).resolve().parent.parent
SOURCE_FILE = REPO_ROOT / "showcase/js/mock_data.js"
TARGET_DIR = REPO_ROOT / "showcase/data/modular"

def main():
    print(f"[*] Reading {SOURCE_FILE}...")
    try:
        content = SOURCE_FILE.read_text(encoding='utf-8')
    except FileNotFoundError:
        print(f"[!] Error: {SOURCE_FILE} not found.")
        return

    # Strip JS assignment: window.MOCK_DATA = {...};
    # Regex to capture the JSON object
    match = re.search(r'window\.MOCK_DATA\s*=\s*({.*});', content, re.DOTALL)

    if not match:
        # Fallback for simple strip
        print("[!] Regex failed, trying simple strip...")
        content = content.replace("window.MOCK_DATA = ", "").strip()
        if content.endswith(";"):
            content = content[:-1]
    else:
        content = match.group(1)

    try:
        data = json.loads(content)
        print("[+] JSON parsed successfully.")
    except json.JSONDecodeError as e:
        print(f"[!] JSON Decode Error: {e}")
        # Save debug file
        with open("debug_mock_data_strip.json", "w") as f:
            f.write(content)
        return

    # Ensure target directory exists
    TARGET_DIR.mkdir(parents=True, exist_ok=True)

    # Split and Save
    mappings = {
        "reports": "reports.json",
        "credit_memos": "credit_memos.json",
        "files": "files.json",
        "stats": "system_status.json",
        "financial_data": "market_data.json"
    }

    for key, filename in mappings.items():
        if key in data:
            target_path = TARGET_DIR / filename
            print(f"[*] Extracting '{key}' to {target_path}...")
            with open(target_path, "w") as f:
                json.dump(data[key], f, indent=2)
        else:
            print(f"[!] Warning: Key '{key}' not found in source data.")

    # Create a manifest
    manifest = {
        "version": "1.0",
        "modules": list(mappings.values())
    }
    with open(TARGET_DIR / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print("[+] Modular data generation complete.")

if __name__ == "__main__":
    main()
