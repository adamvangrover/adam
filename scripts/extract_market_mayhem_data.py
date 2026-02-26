import os
import json
from bs4 import BeautifulSoup

def extract_data():
    input_file = 'showcase/market_mayhem_archive.html'
    output_file = 'showcase/data/market_mayhem_index.json'

    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found.")
        return

    with open(input_file, 'r', encoding='utf-8') as f:
        html_content = f.read()

    soup = BeautifulSoup(html_content, 'html.parser')
    items = soup.find_all('div', class_='archive-item')

    archives = []

    print(f"Found {len(items)} archive items.")

    for i, item in enumerate(items):
        try:
            # Basic Attributes
            year = item.get('data-year')
            type_ = item.get('data-type')

            # Inner Content
            date_span = item.find('span', class_='mono')
            date = date_span.text.strip() if date_span else ""

            h3 = item.find('h3')
            title = h3.text.strip() if h3 else ""

            p = item.find('p')
            description = p.text.strip() if p else ""

            a = item.find('a', class_='cyber-btn')
            link = a['href'] if a else ""

            # Metrics
            metrics_span = item.find_all('span', class_='mono')
            sentiment = 50
            conviction = 50
            sem = 0

            if len(metrics_span) > 1:
                metrics_text = metrics_span[1].text
                # Parse "SENT: 50 | CONV: 50 | SEM: 91"
                parts = metrics_text.split('|')
                for part in parts:
                    part = part.strip()
                    if part.startswith('SENT:'):
                        sentiment = int(part.replace('SENT:', '').strip())
                    elif part.startswith('CONV:'):
                        conviction = int(part.replace('CONV:', '').strip())
                    elif part.startswith('SEM:'):
                        sem = int(part.replace('SEM:', '').strip())

            archive_entry = {
                "id": f"mm-{i:04d}",
                "date": date,
                "title": title,
                "type": type_,
                "year": year,
                "sentiment": sentiment,
                "conviction": conviction,
                "sem": sem,
                "description": description,
                "link": link
            }
            archives.append(archive_entry)

        except Exception as e:
            print(f"Error parsing item {i}: {e}")
            continue

    data = {
        "archives": archives,
        "metadata": {
            "count": len(archives),
            "generated_at": "2025-10-27"
        }
    }

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)

    print(f"Successfully extracted {len(archives)} items to {output_file}")

if __name__ == "__main__":
    extract_data()
