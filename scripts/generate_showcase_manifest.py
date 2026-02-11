import os
import json
import re
from datetime import datetime

SHOWCASE_DIR = "showcase"
OUTPUT_FILE = os.path.join(SHOWCASE_DIR, "data", "report_manifest.json")

def parse_date(filename, content):
    # Try filename first: YYYY-MM-DD
    match = re.search(r'(\d{4}-\d{2}-\d{2})', filename)
    if match:
        return match.group(1)

    # Try content meta tag
    # <meta name="report-date" content="2025-03-15">
    match = re.search(r'<meta\s+name=["\']report-date["\']\s+content=["\'](.*?)["\']', content, re.IGNORECASE)
    if match:
        return match.group(1)

    # Try specific content patterns
    # <span>2025-03-15</span>
    match = re.search(r'<span>(\d{4}-\d{2}-\d{2})</span>', content)
    if match:
        return match.group(1)

    return "Unknown"

def determine_type(filename, title, content):
    filename_lower = filename.lower()
    title_lower = title.lower()

    if "briefing" in filename_lower or "briefing" in title_lower:
        return "DAILY_BRIEFING"
    if "pulse" in filename_lower or "pulse" in title_lower:
        return "MARKET_PULSE"
    if "house_view" in filename_lower or "house view" in title_lower:
        return "HOUSE_VIEW"
    if "market_mayhem" in filename_lower or "market mayhem" in title_lower:
        return "MARKET_MAYHEM"
    if "report" in filename_lower:
        return "REPORT"

    return "OTHER"

def main():
    manifest = []

    if not os.path.exists(SHOWCASE_DIR):
        print(f"Directory {SHOWCASE_DIR} not found.")
        return

    # Ensure output dir exists
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

    for filename in os.listdir(SHOWCASE_DIR):
        if not filename.endswith(".html"):
            continue

        filepath = os.path.join(SHOWCASE_DIR, filename)

        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()

            # Extract Title
            title_match = re.search(r'<title>(.*?)</title>', content, re.IGNORECASE | re.DOTALL)
            title = title_match.group(1).strip() if title_match else filename

            # Remove "ADAM v..." prefix from title for cleaner display
            clean_title = re.sub(r'ADAM v.*? :: ', '', title).strip()

            report_date = parse_date(filename, content)
            report_type = determine_type(filename, clean_title, content)

            manifest.append({
                "id": filename.replace(".html", ""),
                "filename": filename,
                "path": filename, # Relative to showcase/
                "title": clean_title,
                "date": report_date,
                "type": report_type
            })

        except Exception as e:
            print(f"Error parsing {filename}: {e}")

    # Sort by date descending
    manifest.sort(key=lambda x: x['date'], reverse=True)

    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2)

    print(f"Manifest generated with {len(manifest)} entries at {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
