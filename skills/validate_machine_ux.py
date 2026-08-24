import argparse
import json
import os
import sys

try:
    from bs4 import BeautifulSoup
except ImportError:
    print("Error: beautifulsoup4 is not installed. Please install it to run this skill (e.g. pip install beautifulsoup4).")
    sys.exit(1)

def validate_machine_ux(filepath: str) -> dict:
    """
    Validates an HTML file against the Adam OS machine-readable UX standards.
    Checks for semantic tags, data attributes, JSON-LD, and specific meta tags.
    """
    if not os.path.exists(filepath):
        return {"status": "error", "message": f"File not found: {filepath}"}

    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    soup = BeautifulSoup(content, 'html.parser')
    report = {
        "file": filepath,
        "status": "PASS",
        "checks": []
    }

    # 1. Check for data-* attributes
    has_data_attrs = any(tag for tag in soup.find_all() if any(attr.startswith('data-') for attr in tag.attrs))
    report["checks"].append({
        "check": "Data Binding Hooks (data-*)",
        "passed": has_data_attrs,
        "detail": "Found data-* attributes" if has_data_attrs else "Missing data-* attributes on any element."
    })

    # 2. Check for embedded application/json scripts (State parameters)
    json_scripts = soup.find_all('script', type='application/json')
    report["checks"].append({
        "check": "Embedded JSON State",
        "passed": len(json_scripts) > 0,
        "detail": f"Found {len(json_scripts)} <script type='application/json'> block(s)." if json_scripts else "Missing <script type='application/json'> state block."
    })

    # 3. Check for specific metadata tags (Adam architecture role)
    meta_tags = soup.find_all('meta')
    has_adam_role = any(meta.get('name') == 'adam-role' for meta in meta_tags)
    report["checks"].append({
        "check": "Adam Role Metadata",
        "passed": has_adam_role,
        "detail": "Found <meta name='adam-role'>." if has_adam_role else "Missing <meta name='adam-role'> tag defining the page's role."
    })

    # 4. Check for JSON-LD tags
    json_ld_scripts = soup.find_all('script', type='application/ld+json')
    report["checks"].append({
        "check": "JSON-LD Tags",
        "passed": len(json_ld_scripts) > 0,
        "detail": f"Found {len(json_ld_scripts)} JSON-LD script(s)." if json_ld_scripts else "Missing machine-readable JSON-LD tags."
    })

    # Evaluate overall status
    if not all(check["passed"] for check in report["checks"]):
        report["status"] = "FAIL"

    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate HTML for Adam Machine-Readable UX standards.")
    parser.add_argument("filepath", type=str, help="Path to the HTML file to validate")
    parser.add_argument("--format", choices=['text', 'json'], default='text', help="Output format")

    args = parser.parse_args()

    result = validate_machine_ux(args.filepath)

    if args.format == 'json':
        print(json.dumps(result, indent=2))
    else:
        if result.get("status") == "error":
            print(f"Error: {result.get('message')}")
            sys.exit(1)

        print(f"--- UX Audit Report for: {result['file']} ---")
        print(f"Overall Status: {result['status']}")
        print("-" * 40)
        for check in result.get("checks", []):
            status_str = "[PASS]" if check['passed'] else "[FAIL]"
            print(f"{status_str} {check['check']}")
            if not check['passed']:
                print(f"      -> {check['detail']}")

        if result['status'] == "FAIL":
            sys.exit(1)
        else:
            sys.exit(0)
