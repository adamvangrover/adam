import os
import re

ADAM_DAILY_DIR = "showcase/data/adam_daily"
INDEX_FILE = os.path.join(ADAM_DAILY_DIR, "index.html")

def find_entry_file(date_dir):
    dir_path = os.path.join(ADAM_DAILY_DIR, date_dir)
    if not os.path.isdir(dir_path):
        return "index.html"
    files = os.listdir(dir_path)
    # Preference: index.html > index_v*.html > market_mayhem_*.html > market_mayhem_*.md
    if "index.html" in files: return "index.html"
    for f in files:
        if f.startswith("index_v") and f.endswith(".html"): return f
    for f in files:
        if f.startswith("market_mayhem_") and f.endswith(".html"): return f
    for f in files:
        if f.startswith("market_mayhem_") and f.endswith(".md"): return f
    for f in files:
        if f.endswith(".html"): return f
    if files: return files[0]
    return "index.html"

def main():
    date_dirs = [d for d in os.listdir(ADAM_DAILY_DIR) if re.match(r'\d{4}-\d{2}-\d{2}', d) and os.path.isdir(os.path.join(ADAM_DAILY_DIR, d))]
    date_dirs.sort()

    dates2026 = [d for d in date_dirs if d.startswith("2026")]

    mapping = {
        'macro_tail_risks': 'macro_tail_risks.html',
        'industry_watchlists': 'industry_watchlists.html',
        'rates_outlook': 'rates_outlook.html',
        'portfolio_analysis': 'portfolio_analysis.html',
    }

    for d in date_dirs:
        mapping[d] = find_entry_file(d)

    with open(INDEX_FILE, "r") as f:
        content = f.read()

    # Replace dates2026 array
    dates_str = "', '".join(dates2026)
    content = re.sub(
        r'const dates2026 = \[.*?\];',
        f"const dates2026 = [\n            '{dates_str}'\n        ];",
        content,
        flags=re.DOTALL
    )

    # Replace mapping object
    mapping_str = "{\n"
    for k, v in mapping.items():
        mapping_str += f"                '{k}': '{v}',\n"
    mapping_str += "            }"

    content = re.sub(
        r'const mapping = \{.*?\};',
        f"const mapping = {mapping_str};",
        content,
        flags=re.DOTALL
    )

    with open(INDEX_FILE, "w") as f:
        f.write(content)

    print(f"Adam Daily index updated successfully. Found {len(dates2026)} dates for 2026.")

if __name__ == "__main__":
    main()
