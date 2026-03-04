import json
import os
import jinja2

DATA_FILE = "showcase/data/newsletter_data.json"
OUTPUT_DIR = "showcase"
TEMPLATE_FILE = "showcase/templates/market_mayhem_report.html"

def generate_html_files():
    if not os.path.exists(DATA_FILE):
        return

    with open(DATA_FILE, 'r') as f:
        data = json.load(f)

    # Use Jinja2 to correctly process the template and all its variables
    template_dir = os.path.dirname(TEMPLATE_FILE)
    template_name = os.path.basename(TEMPLATE_FILE)

    env = jinja2.Environment(loader=jinja2.FileSystemLoader(template_dir), autoescape=True)
    try:
        template = env.get_template(template_name)
    except Exception as e:
        print(f"Template loading error: {e}")
        return

    count = 0
    for item in data:
        if item.get("type") == "DAILY_BRIEFING":
            filename = item.get("filename")
            if filename:
                filepath = os.path.join(OUTPUT_DIR, filename)

                # Provide default values for all possible variables expected by the template
                # to prevent unrendered placeholders
                render_data = {
                    "tier_class": "standard",
                    "title": item.get("title", ""),
                    "summary": item.get("summary", ""),
                    "date": item.get("date", ""),
                    "full_body": item.get("full_body", ""),
                    "sentiment_score": item.get("sentiment_score", 50),
                    "conviction": item.get("conviction", 50),
                    "semantic_score": item.get("semantic_score", 50),
                    "prov_hash": item.get("provenance_hash", "00000000000000000000000000000000"),
                    "entities": item.get("entities", {"tickers": [], "sovereigns": [], "keywords": []})
                }

                html = template.render(**render_data)

                with open(filepath, 'w') as out:
                    out.write(html)
                count += 1

    print(f"Generated {count} HTML files for Daily Briefings with proper template rendering.")

if __name__ == "__main__":
    generate_html_files()
