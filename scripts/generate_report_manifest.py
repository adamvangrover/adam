import os
import json
import re
from datetime import datetime

# Try importing BeautifulSoup, fallback to simple regex if missing
try:
    from bs4 import BeautifulSoup
    HAS_BS4 = True
except ImportError:
    HAS_BS4 = False

SHOWCASE_DIR = "showcase"
OUTPUT_FILE = "showcase/data/report_manifest.json"

# Regex patterns for date extraction from FILENAME
# Order matters: more specific patterns first
DATE_PATTERNS = [
    (r"(\d{4})_(\d{2})_(\d{2})", "%Y-%m-%d"),       # YYYY_MM_DD -> YYYY-MM-DD
    (r"(\d{4})-(\d{2})-(\d{2})", "%Y-%m-%d"),       # YYYY-MM-DD -> YYYY-MM-DD
    (r"(\d{4})(\d{2})(\d{2})", "%Y-%m-%d"),         # YYYYMMDD -> YYYY-MM-DD
    (r"(\d{2})(\d{2})(\d{4})", "%m-%d-%Y"),         # MMDDYYYY -> YYYY-MM-DD (requires reordering)
]

def extract_date_from_filename(filename):
    for pattern, fmt_type in DATE_PATTERNS:
        match = re.search(pattern, filename)
        if match:
            try:
                groups = match.groups()
                if fmt_type == "%Y-%m-%d":
                    return f"{groups[0]}-{groups[1]}-{groups[2]}"
                elif fmt_type == "%m-%d-%Y":
                    # MMDDYYYY -> YYYY-MM-DD
                    return f"{groups[2]}-{groups[0]}-{groups[1]}"
            except Exception:
                continue
    return "Unknown"

def extract_title(filepath):
    title = "Untitled Report"
    try:
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()

            if HAS_BS4:
                soup = BeautifulSoup(content, "html.parser")
                if soup.title and soup.title.string:
                    title = soup.title.string.strip()
            else:
                # Fallback regex for title
                match = re.search(r"<title>(.*?)</title>", content, re.IGNORECASE | re.DOTALL)
                if match:
                    title = match.group(1).strip()
    except Exception as e:
        print(f"Error reading {filepath}: {e}")

    # If title is still generic or empty, use filename
    if not title or title == "Untitled Report":
        title = os.path.basename(filepath).replace(".html", "").replace("_", " ")

    return title

def categorize(filename):
    lower_name = filename.lower()
    if "daily_briefing" in lower_name:
        return "DAILY_BRIEFING"
    elif "market_pulse" in lower_name:
        return "MARKET_PULSE"
    elif "market_mayhem" in lower_name or "mm" in lower_name[:2]: # heuristic for MM prefix
        return "MARKET_MAYHEM"
    elif "house_view" in lower_name:
        return "HOUSE_VIEW"
    elif "deep_dive" in lower_name:
        return "DEEP_DIVE"
    elif "industry_report" in lower_name or "company_report" in lower_name:
        return "REPORT"
    else:
        return "OTHER"

def main():
    manifest = []

    if not os.path.exists(SHOWCASE_DIR):
        print(f"Directory {SHOWCASE_DIR} not found.")
        return

    # Walk through showcase directory
    for root, dirs, files in os.walk(SHOWCASE_DIR):
        for file in files:
            if not file.endswith(".html"):
                continue

            # Skip library files, archives, and templates
            if "library" in file or "archive" in file or "index" in file or "template" in file:
                continue

            filepath = os.path.join(root, file)
            # relative_path relative to showcase/ root for href
            # If the file is in showcase/apps/, href should be apps/file.html
            # But wait, showcase/ is the root for the web server context usually?
            # Assuming report_manifest.json is loaded by pages IN showcase/,
            # links should be relative to showcase/.

            relative_path = os.path.relpath(filepath, SHOWCASE_DIR)

            date_str = extract_date_from_filename(file)
            category = categorize(file)
            title = extract_title(filepath)

            manifest.append({
                "id": file.replace(".html", ""),
                "filename": file,
                "path": relative_path,
                "title": title,
                "date": date_str,
                "type": category
            })

    # Sort: Dates descending, Unknowns last
    def sort_key(item):
        d = item["date"]
        if d == "Unknown":
            return "0000-00-00"
        return d

    manifest.sort(key=sort_key, reverse=True)

    # Ensure output directory exists
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=4)

    print(f"Generated manifest with {len(manifest)} entries at {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
