import os
import json
from bs4 import BeautifulSoup
import re

def main():
    base_dir_restored = "core/libraries_and_archives/restored_newsletters"
    base_dir_newsletters = "core/libraries_and_archives/newsletters"
    dirs_to_scan = [base_dir_restored, base_dir_newsletters]
    output_file = "showcase/data/archive_data.json"
    archives = []

    for base_dir in dirs_to_scan:
        if not os.path.exists(base_dir):
            print(f"Directory {base_dir} does not exist. Skipping.")
            continue

        for filename in sorted(os.listdir(base_dir), reverse=True):
            if filename.endswith(".html") or filename.endswith(".md") or filename.endswith(".json"):
                filepath = os.path.join(base_dir, filename)
                with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
                    content = f.read()

            title = filename.replace(".html", "").replace(".md", "").replace("_", " ").title()
            date = "Unknown"

            # Extract date from filename if possible
            match = re.search(r'(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)_\d{1,2}_\d{4}', filename, re.IGNORECASE)
            if match:
                date = match.group(0).replace("_", " ").title()

            summary = "No summary available."
            if filename.endswith(".html"):
                soup = BeautifulSoup(content, 'html.parser')
                h1 = soup.find('h1')
                if h1:
                    title = h1.get_text(strip=True)
                p_tags = soup.find_all('p')
                for p in p_tags:
                    text = p.get_text(strip=True)
                    if len(text) > 50:
                        summary = text[:150] + "..."
                        break
            elif filename.endswith(".json"):
                try:
                    data = json.loads(content)
                    title = data.get('title', title)
                    summary = data.get('summary', data.get('executive_summary', "No summary available."))
                    date = data.get('date', date)
                except json.JSONDecodeError:
                    pass
            else:
                lines = content.split('\n')
                for line in lines:
                    if line.startswith('# '):
                        title = line[2:].strip()
                    elif len(line.strip()) > 50 and not line.startswith('#'):
                        summary = line.strip()[:150] + "..."
                        break

            archives.append({
                "title": title,
                "date": date,
                "type": "NEWSLETTER",
                "summary": summary,
                "url": f"../{filepath}"
            })

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(archives, f, indent=4)

    print(f"Compiled {len(archives)} archives to {output_file}")

if __name__ == "__main__":
    main()
