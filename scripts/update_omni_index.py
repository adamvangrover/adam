#!/usr/bin/env python3
import os
import re
import json

NEW_ENTRY = {
    "u": "adam_standalone_distribution.html",
    "t": "ADAM Repository — Standalone Interactive Distribution",
    "d": "ADAM Repository Standalone Interactive Distribution - Adaptive Moment Estimation & Loss Landscape Arena"
}

def update_html_files():
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    html_files = [f for f in os.listdir(root_dir) if f.endswith(".html")]

    pattern = re.compile(r"const omniIndex = (\[.*?\]);", re.DOTALL)

    updated_count = 0

    for filename in html_files:
        filepath = os.path.join(root_dir, filename)
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()

        match = pattern.search(content)
        if not match:
            continue

        raw_json = match.group(1)
        try:
            items = json.loads(raw_json)
        except json.JSONDecodeError:
            print(f"Failed to parse JSON in {filename}")
            continue

        # Check if entry already exists
        exists = any(item.get("u") == NEW_ENTRY["u"] for item in items)
        if not exists:
            items.append(NEW_ENTRY)
            new_json_str = json.dumps(items)
            replacement = f"const omniIndex = {new_json_str};"
            new_content = pattern.sub(lambda m: replacement, content)
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(new_content)
            updated_count += 1
            print(f"Updated omniIndex in {filename}")
        else:
            print(f"Entry already exists in {filename}")

    print(f"Finished updating omniIndex in {updated_count} HTML files.")

if __name__ == "__main__":
    update_html_files()
