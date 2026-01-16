import os
import re
import glob
import json
from datetime import datetime

# --- Configuration ---
OUTPUT_DIR = "showcase"
SOURCE_DIR = "core/libraries_and_archives/newsletters"
ARCHIVE_FILE = "showcase/market_mayhem_archive.html"

# --- Formatting Helpers ---

def simple_md_to_html(text):
    """Converts basic Markdown to HTML using Regex."""
    if not text: return ""

    # Headers
    text = re.sub(r'^# (.*?)$', r'<h1 class="title">\1</h1>', text, flags=re.MULTILINE)
    text = re.sub(r'^## (.*?)$', r'<h2>\1</h2>', text, flags=re.MULTILINE)
    text = re.sub(r'^### (.*?)$', r'<h3>\1</h3>', text, flags=re.MULTILINE)

    # Bold/Italic
    text = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', text)
    text = re.sub(r'\*(.*?)\*', r'<em>\1</em>', text)

    # Lists
    lines = text.split('\n')
    new_lines = []
    in_list = False

    for line in lines:
        if line.strip().startswith('* '):
            if not in_list:
                new_lines.append('<ul>')
                in_list = True
            content = line.strip()[2:]
            new_lines.append(f'<li>{content}</li>')
        else:
            if in_list:
                new_lines.append('</ul>')
                in_list = False
            new_lines.append(line)

    if in_list: new_lines.append('</ul>')
    text = '\n'.join(new_lines)

    # Paragraphs (double newline) - skip if line starts with < (html tag)
    paragraphs = text.split('\n\n')
    final_html = ""
    for p in paragraphs:
        p = p.strip()
        if not p: continue
        if p.startswith('<'):
            final_html += p + "\n"
        else:
            final_html += f"<p>{p}</p>\n"

    return final_html

def parse_date(date_str):
    """Attempts to parse various date formats into YYYY-MM-DD."""
    formats = [
        "%Y-%m-%d",
        "%B %d, %Y",       # March 15, 2025
        "%b %d, %Y",       # Oct 31, 2025
        "%d %B %Y",        # 15 March 2025
        "%Y%m%d",          # 20251210
        "%b_%d_%Y",        # nov_01_2025
    ]

    for fmt in formats:
        try:
            return datetime.strptime(date_str.strip(), fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue

    # Fallback: Try regex to extract date parts
    match = re.search(r'(\d{4})[-_]?(\d{2})[-_]?(\d{2})', date_str)
    if match:
        return f"{match.group(1)}-{match.group(2)}-{match.group(3)}"

    return date_str # Return original if failed (or handle error)

# --- Parsers ---

def parse_json_file(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Extract fields
        title = data.get('title', 'Untitled Report')

        # Try to find date in title or filename
        date_str = "2025-01-01" # Default

        # Heuristic: Extract date from filename if not in data
        filename = os.path.basename(filepath)
        date_match = re.search(r'(\d{4}_\d{2}_\d{2})', filename)
        if date_match:
            date_str = parse_date(date_match.group(1).replace('_', '-'))
        elif "date" in data: # sometimes in root
             date_str = parse_date(data["date"])

        # Construct Body
        body_html = ""
        sections = data.get('sections', [])
        summary = ""

        for section in sections:
            sec_title = section.get('title', '')
            if sec_title: body_html += f"<h2>{sec_title}</h2>"

            if 'content' in section:
                body_html += f"<p>{section['content']}</p>"
                if not summary and "Summary" in sec_title:
                    summary = section['content'][:200] + "..."

            if 'items' in section:
                body_html += "<ul>"
                for item in section['items']:
                    body_html += f"<li>{item}</li>"
                body_html += "</ul>"

            if 'ideas' in section:
                for idea in section['ideas']:
                    body_html += f"<h3>{idea.get('asset', 'Asset')}</h3>"
                    body_html += f"<p><strong>Rationale:</strong> {idea.get('rationale','')}</p>"

        if not summary: summary = f"Market Report from {date_str}"

        return {
            "title": title,
            "date": date_str,
            "summary": summary,
            "type": "NEWSLETTER", # Default for JSONs in this folder
            "full_body": body_html,
            "sentiment_score": 50, # Default
            "filename": filename.replace('.json', '.html').replace('.md', '.html'),
            "is_sourced": True
        }
    except Exception as e:
        print(f"Error parsing JSON {filepath}: {e}")
        return None

def parse_markdown_file(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()

        filename = os.path.basename(filepath)

        # Metadata Extraction
        title_match = re.search(r'^# (.*?)$', content, re.MULTILINE)
        title = title_match.group(1).strip() if title_match else filename.replace('.md', '')

        # Date Extraction
        date_match = re.search(r'\*\*Date:\*\* (.*?)$', content, re.MULTILINE)
        if not date_match:
             # Try regex in filename
             date_match = re.search(r'(\d{4})[-_]?(\d{2})[-_]?(\d{2})', filename)
             if date_match:
                 raw_date = f"{date_match.group(1)}-{date_match.group(2)}-{date_match.group(3)}"
             else:
                 # Try finding date in title
                 raw_date = "2025-01-01" # Fallback
        else:
            raw_date = date_match.group(1)

        date = parse_date(raw_date)

        # Type Heuristics
        if "Deep_Dive" in filename: type_ = "DEEP_DIVE"
        elif "Pulse" in filename or "Pulse" in title: type_ = "MARKET_PULSE"
        elif "Industry" in filename: type_ = "INDUSTRY_REPORT"
        elif "Weekly" in filename: type_ = "WEEKLY_RECAP"
        elif "Tech_Watch" in filename: type_ = "TECH_WATCH"
        elif "Agent" in filename or "Alignment" in title: type_ = "AGENT_NOTE"
        elif "Performance" in filename or "Audit" in title: type_ = "PERFORMANCE_REVIEW"
        else: type_ = "NEWSLETTER"

        # Summary Heuristics
        summary_match = re.search(r'(Executive Summary|Summary)\s*\n(.*?)(?=\n##)', content, re.DOTALL | re.IGNORECASE)
        if summary_match:
            summary_raw = summary_match.group(2).strip()
            summary = re.sub(r'[*#]', '', summary_raw)[:200] + "..."
        else:
            summary = "Market analysis and strategic insights."

        # Convert Body
        # Remove the title line from content before converting if it was extracted
        if title_match:
             content = content.replace(title_match.group(0), "", 1).strip()

        body_html = simple_md_to_html(content)

        return {
            "title": title,
            "date": date,
            "summary": summary,
            "type": type_,
            "full_body": body_html,
            "sentiment_score": 50,
            "filename": filename.replace('.md', '.html').replace('.json', '.html'),
            "is_sourced": True
        }

    except Exception as e:
        print(f"Error parsing MD {filepath}: {e}")
        return None

# --- Main Generation ---

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ADAM v23.5 :: {title}</title>
    <link rel="stylesheet" href="css/style.css">
    <link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Inter:wght@300;400;600;700&family=Playfair+Display:ital,wght@0,400;0,700;1,400&display=swap" rel="stylesheet">
    <script src="https://cdn.tailwindcss.com"></script>
    <style>
        :root {{
            --paper-bg: #fdfbf7;
            --ink-color: #1a1a1a;
            --accent-red: #cc0000;
            --cyber-black: #050b14;
            --cyber-blue: #00f3ff;
        }}
        body {{ margin: 0; background: var(--cyber-black); color: #e0e0e0; font-family: 'Inter', sans-serif; }}

        .newsletter-wrapper {{
            max-width: 1000px;
            margin: 40px auto;
            display: grid;
            grid-template-columns: 3fr 1fr;
            gap: 40px;
            padding: 20px;
        }}

        /* Responsive for mobile */
        @media (max-width: 768px) {{
            .newsletter-wrapper {{ grid-template-columns: 1fr; }}
        }}

        .paper-sheet {{
            background: var(--paper-bg);
            color: var(--ink-color);
            padding: 60px;
            font-family: 'Georgia', 'Times New Roman', serif;
            box-shadow: 0 0 50px rgba(0,0,0,0.5);
            position: relative;
        }}

        /* Typography */
        h1.title {{
            font-family: 'Playfair Display', serif;
            font-size: 3rem;
            border-bottom: 4px solid var(--ink-color);
            padding-bottom: 20px;
            margin-bottom: 30px;
            letter-spacing: -1px;
            line-height: 1.1;
        }}
        h2 {{
            font-family: 'Inter', sans-serif;
            font-weight: 700;
            font-size: 1.2rem;
            margin-top: 40px;
            margin-bottom: 15px;
            color: var(--accent-red);
            text-transform: uppercase;
            letter-spacing: 1px;
        }}
        p {{ line-height: 1.8; margin-bottom: 20px; font-size: 1.05rem; }}
        li {{ line-height: 1.6; margin-bottom: 10px; }}

        /* Sidebar */
        .cyber-sidebar {{
            font-family: 'JetBrains Mono', monospace;
        }}
        .sidebar-widget {{
            border: 1px solid #333;
            background: rgba(255,255,255,0.02);
            padding: 20px;
            margin-bottom: 20px;
        }}
        .sidebar-title {{
            color: var(--cyber-blue);
            font-size: 0.8rem;
            text-transform: uppercase;
            border-bottom: 1px solid #333;
            padding-bottom: 10px;
            margin-bottom: 15px;
        }}

        .cyber-btn {{
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.75rem;
            padding: 6px 12px;
            border: 1px solid #444;
            color: #e0e0e0;
            background: rgba(0,0,0,0.8);
            text-decoration: none;
            display: inline-block;
            margin-bottom: 20px;
        }}
        .cyber-btn:hover {{ border-color: var(--cyber-blue); color: var(--cyber-blue); }}
    </style>
</head>
<body>
    <div style="max-width: 1400px; margin: 0 auto; padding: 20px;">
        <a href="market_mayhem_archive.html" class="cyber-btn">&larr; BACK TO ARCHIVE</a>
    </div>

    <div class="newsletter-wrapper">
        <!-- Main Content (Paper Style) -->
        <div class="paper-sheet">
            <div style="display:flex; justify-content:space-between; font-family:'JetBrains Mono'; font-size:0.8rem; color:#666; margin-bottom:20px;">
                <span>{date}</span>
                <span>TYPE: {type}</span>
            </div>

            <h1 class="title">{title}</h1>

            <!-- Body Content -->
            {full_body}

            <div style="margin-top: 40px; border-top: 1px solid #ccc; padding-top: 20px; font-style: italic; color: #666;">
                End of Report.
            </div>
        </div>

        <!-- Sidebar (Cyber Style) -->
        <aside class="cyber-sidebar">
            <div class="sidebar-widget">
                <div class="sidebar-title">Metadata</div>
                <div style="font-size: 0.8rem; color: #aaa;">
                    <div>DATE: <span style="color:#fff">{date}</span></div>
                    <div>TYPE: <span style="color:#fff">{type}</span></div>
                    <div>CLASS: <span style="color:#fff">UNCLASSIFIED</span></div>
                </div>
            </div>

             <div class="sidebar-widget">
                <div class="sidebar-title">System Status</div>
                 <div style="display:flex; align-items:center; gap:10px;">
                    <div style="width:10px; height:10px; background:#00ff00; border-radius:50%; box-shadow: 0 0 5px #00ff00;"></div>
                    <span style="font-size:0.8rem; color:#00ff00;">ONLINE</span>
                </div>
            </div>
        </aside>
    </div>
</body>
</html>
"""

def generate_archive():
    print("Starting Archive Generation...")

    # 1. Gather all data sources
    all_items = []

    # Scan Source Directory
    print(f"Scanning {SOURCE_DIR}...")
    source_files = glob.glob(os.path.join(SOURCE_DIR, "*"))

    for filepath in source_files:
        item = None
        if filepath.endswith(".json"):
            item = parse_json_file(filepath)
        elif filepath.endswith(".md"):
            item = parse_markdown_file(filepath)

        if item:
            all_items.append(item)

    # 2. Add Legacy/Hardcoded items if missing (from original script logic)
    historical_items = [
        {
            "date": "2020-03-20",
            "title": "MARKET MAYHEM: THE GREAT SHUT-IN",
            "summary": "Lockdown. The global economy has come to a screeching halt.",
            "type": "HISTORICAL",
            "filename": "newsletter_market_mayhem_mar_2020.html",
            "full_body": "<p>The World Has Stopped...</p>", # simplified for this reconstruction
            "is_sourced": True
        },
        {
            "date": "2008-09-19",
            "title": "MARKET MAYHEM: THE LEHMAN MOMENT",
            "summary": "Existential Panic. A 158-year-old bank vanished.",
            "type": "HISTORICAL",
            "filename": "newsletter_market_mayhem_sep_2008.html",
             "full_body": "<p>The Week Wall Street Died...</p>",
             "is_sourced": True
        },
        {
            "date": "1987-10-23",
            "title": "MARKET MAYHEM: BLACK MONDAY AFTERMATH",
            "summary": "Shell-Shocked. Dow Jones fell 22.6% in a single day.",
            "type": "HISTORICAL",
            "filename": "newsletter_market_mayhem_oct_1987.html",
             "full_body": "<p>The Crash...</p>",
             "is_sourced": True
        }
    ]

    # Check if we already have these dates, if not, add them
    existing_dates = set(i['date'] for i in all_items)
    for h in historical_items:
        if h['date'] not in existing_dates:
            all_items.append(h)

    # 3. Scan for Orphan HTML Files in Showcase
    # This ensures that if we have HTML files that were generated previously or manually added,
    # but lack a source file in the newsletters directory, they are still indexed.
    print("Scanning for orphan HTML files in showcase/...")
    existing_filenames = set(item['filename'] for item in all_items)
    showcase_files = glob.glob(os.path.join(OUTPUT_DIR, "newsletter_*.html")) + \
                     glob.glob(os.path.join(OUTPUT_DIR, "MM*.html")) + \
                     glob.glob(os.path.join(OUTPUT_DIR, "*_Market_Mayhem.html")) + \
                     glob.glob(os.path.join(OUTPUT_DIR, "Deep_Dive_*.html")) + \
                     glob.glob(os.path.join(OUTPUT_DIR, "Tech_Watch_*.html")) + \
                     glob.glob(os.path.join(OUTPUT_DIR, "Industry_Report_*.html")) + \
                     glob.glob(os.path.join(OUTPUT_DIR, "House_View_*.html")) + \
                     glob.glob(os.path.join(OUTPUT_DIR, "Equity_Research_*.html")) + \
                     glob.glob(os.path.join(OUTPUT_DIR, "Weekly_Recap_*.html")) + \
                     glob.glob(os.path.join(OUTPUT_DIR, "Market_Mayhem_*.html")) + \
                     glob.glob(os.path.join(OUTPUT_DIR, "market_*.html"))

    TITLE_RE = re.compile(r'<h1 class="title">(.*?)</h1>', re.IGNORECASE)
    DATE_RE = re.compile(r'<span>([\d-]+)</span>', re.IGNORECASE)
    SUMMARY_RE = re.compile(r'font-style: italic.*?>(.*?)</p>', re.IGNORECASE | re.DOTALL)
    TYPE_RE = re.compile(r'TYPE: (.*?)</span>', re.IGNORECASE)

    for filepath in showcase_files:
        filename = os.path.basename(filepath)
        if filename == "market_mayhem_archive.html": continue
        if filename in existing_filenames: continue

        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()

            title_m = TITLE_RE.search(content)
            date_m = DATE_RE.search(content)
            summary_m = SUMMARY_RE.search(content)
            type_m = TYPE_RE.search(content)

            if title_m and date_m:
                all_items.append({
                    "date": date_m.group(1).strip(),
                    "title": title_m.group(1).strip(),
                    "summary": summary_m.group(1).strip() if summary_m else "No summary available.",
                    "filename": filename,
                    "type": type_m.group(1).strip() if type_m else "NEWSLETTER",
                    "sentiment_score": 50, # Default
                    "is_sourced": False # Don't regenerate file
                })
        except Exception as e:
            # print(f"Skipping {filename}: {e}")
            pass

    # 4. Generate HTML Files (Only for sourced items)
    print(f"Processing {len(all_items)} total reports...")
    for item in all_items:
        if not item.get("is_sourced", False): continue

        out_path = os.path.join(OUTPUT_DIR, item['filename'])

        content = HTML_TEMPLATE.format(
            title=item["title"],
            date=item["date"],
            summary=item["summary"],
            type=item["type"],
            full_body=item["full_body"]
        )

        with open(out_path, "w", encoding='utf-8') as f:
            f.write(content)

    # 5. Generate Archive Index
    all_items.sort(key=lambda x: x["date"], reverse=True)

    grouped = {}
    historical = []

    for item in all_items:
        try:
            year = item["date"].split("-")[0]
            if int(year) < 2020:
                historical.append(item)
            else:
                if year not in grouped: grouped[year] = []
                grouped[year].append(item)
        except: pass

    # Build Archive HTML
    list_html = ""

    # Filter Bar
    list_html += """
    <div style="margin-bottom: 30px; display: flex; gap: 10px;">
        <input type="text" id="searchInput" placeholder="Search archive..."
            style="background: #111; border: 1px solid #333; color: white; padding: 10px; flex-grow: 1; font-family: 'JetBrains Mono';"
            onkeyup="filterArchive()">
        <select id="yearFilter" onchange="filterArchive()" style="background: #111; border: 1px solid #333; color: white; padding: 10px; font-family: 'JetBrains Mono';">
            <option value="ALL">ALL YEARS</option>
            """
    for y in sorted(grouped.keys(), reverse=True):
        list_html += f'<option value="{y}">{y}</option>'

    list_html += """
            <option value="HISTORICAL">HISTORICAL</option>
        </select>
    </div>
    <div id="archiveGrid">
    """

    for year in sorted(grouped.keys(), reverse=True):
        list_html += f'<div class="year-header" data-year="{year}">{year} ARCHIVE</div>\n'
        for item in grouped[year]:
            type_color = "#333"
            if item['type'] == 'MARKET_PULSE': type_color = "#00f3ff"
            elif item['type'] == 'DEEP_DIVE': type_color = "#cc0000"
            elif item['type'] == 'NEWSLETTER': type_color = "#ffff00"
            elif item['type'] == 'AGENT_NOTE': type_color = "#00ff00" # Matrix Green
            elif item['type'] == 'PERFORMANCE_REVIEW': type_color = "#60a5fa" # Slate Blue

            list_html += f"""
            <div class="archive-item" data-title="{item['title'].lower()}" data-year="{year}">
                <div style="width: 5px; background: {type_color}; margin-right: 15px;"></div>
                <div style="flex-grow: 1;">
                    <div style="display:flex; align-items:center; gap:10px;">
                        <span class="item-date">{item["date"]}</span>
                        <span class="type-badge" style="background:{type_color}; color:#000;">{item.get("type", "REPORT")}</span>
                    </div>
                    <h3 class="item-title">{item["title"]}</h3>
                    <div class="item-summary">{item["summary"]}</div>
                </div>
                <a href="{item["filename"]}" class="read-btn">DECRYPT &rarr;</a>
            </div>
            """

    if historical:
        list_html += f'<div class="year-header" data-year="HISTORICAL">HISTORICAL ARCHIVE</div>\n'
        for item in historical:
            list_html += f"""
            <div class="archive-item" data-title="{item['title'].lower()}" data-year="HISTORICAL">
                <div style="width: 5px; background: #666; margin-right: 15px;"></div>
                <div style="flex-grow: 1;">
                    <div style="display:flex; align-items:center; gap:10px;">
                        <span class="item-date">{item["date"]}</span>
                        <span class="type-badge historical">HISTORICAL</span>
                    </div>
                    <h3 class="item-title">{item["title"]}</h3>
                    <div class="item-summary">{item["summary"]}</div>
                </div>
                <a href="{item["filename"]}" class="read-btn">DECRYPT &rarr;</a>
            </div>
            """

    list_html += "</div>" # End Grid

    page_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ADAM v23.5 :: MARKET MAYHEM ARCHIVE</title>
    <link rel="stylesheet" href="css/style.css">
    <link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Inter:wght@300;400;600;700&display=swap" rel="stylesheet">
    <style>
        :root {{
            --primary-color: #00f3ff;
            --accent-color: #cc0000;
            --bg-color: #050b14;
            --text-primary: #e0e0e0;
        }}
        body {{ margin: 0; background: var(--bg-color); color: var(--text-primary); font-family: 'Inter', sans-serif; overflow-x: hidden; }}
        .mono {{ font-family: 'JetBrains Mono', monospace; }}

        .cyber-header {{
            height: 60px; display: flex; align-items: center; justify-content: space-between;
            padding: 0 20px; border-bottom: 1px solid var(--accent-color);
            background: rgba(5, 11, 20, 0.95); position: sticky; top: 0; z-index: 100;
        }}
        .scan-line {{ position: fixed; top: 0; left: 0; width: 100%; height: 2px; background: rgba(204, 0, 0, 0.1); animation: scan 3s linear infinite; pointer-events: none; z-index: 999; }}
        @keyframes scan {{ 0% {{ top: 0; }} 100% {{ top: 100%; }} }}

        .cyber-btn {{
            font-family: 'JetBrains Mono', monospace; font-size: 0.75rem; padding: 6px 12px;
            border: 1px solid #444; color: var(--text-primary); background: rgba(0,0,0,0.3);
            text-decoration: none; display: inline-block;
        }}
        .cyber-btn:hover {{ border-color: var(--primary-color); color: var(--primary-color); }}

        .archive-list {{ max-width: 1000px; margin: 40px auto; padding: 0 20px; }}

        .year-header {{
            font-family: 'JetBrains Mono'; font-size: 2rem; color: #333; font-weight: bold;
            border-bottom: 2px solid #222; margin-top: 40px; margin-bottom: 20px;
        }}

        .archive-item {{
            border: 1px solid #333; background: rgba(255, 255, 255, 0.03); padding: 20px;
            display: flex; align-items: stretch; gap: 20px;
            transition: all 0.2s ease;
            margin-bottom: 15px;
        }}
        .archive-item:hover {{ border-color: var(--accent-color); background: rgba(204, 0, 0, 0.05); transform: translateX(5px); }}

        .item-date {{ font-family: 'JetBrains Mono'; font-size: 0.8rem; color: var(--accent-color); }}
        .item-title {{ font-size: 1.2rem; font-weight: 700; color: #fff; margin: 5px 0; font-family: 'Inter'; }}
        .item-summary {{ color: #888; font-size: 0.85rem; max-width: 600px; }}

        .read-btn {{
            padding: 8px 16px; border: 1px solid var(--accent-color); color: var(--accent-color);
            font-family: 'JetBrains Mono'; text-transform: uppercase; font-size: 0.75rem;
            background: rgba(0,0,0,0.5); text-decoration: none; white-space: nowrap; align-self: center;
        }}
        .read-btn:hover {{ background: var(--accent-color); color: #000; }}

        .type-badge {{ font-size: 0.6rem; padding: 2px 6px; border-radius: 2px; font-weight: bold; font-family: 'JetBrains Mono'; background: #333; color: white; }}
        .historical {{ background: #666; }}
    </style>
    <script>
        function filterArchive() {{
            const input = document.getElementById('searchInput').value.toLowerCase();
            const yearFilter = document.getElementById('yearFilter').value;
            const items = document.querySelectorAll('.archive-item');
            const headers = document.querySelectorAll('.year-header');

            items.forEach(item => {{
                const title = item.getAttribute('data-title');
                const year = item.getAttribute('data-year');

                let matchesSearch = title.includes(input);
                let matchesYear = (yearFilter === 'ALL') || (year === yearFilter) || (yearFilter === 'HISTORICAL' && year === 'HISTORICAL');

                if (matchesSearch && matchesYear) {{
                    item.style.display = 'flex';
                }} else {{
                    item.style.display = 'none';
                }}
            }});
        }}
    </script>
</head>
<body>
    <div class="scan-line"></div>
    <header class="cyber-header">
        <div style="display: flex; align-items: center; gap: 20px;">
            <h1 class="mono" style="margin: 0; font-size: 1.5rem; color: var(--accent-color); letter-spacing: 2px;">MARKET MAYHEM</h1>
            <div class="mono" style="font-size: 0.8rem; color: #666; border-left: 1px solid #333; padding-left: 10px;">DEEP ARCHIVE</div>
        </div>
        <nav><a href="index.html" class="cyber-btn">&larr; MISSION CONTROL</a></nav>
    </header>
    <main>
        <div class="archive-list">
            {list_html}
        </div>
        <div style="text-align: center; margin: 60px 0; color: #444; font-family: 'JetBrains Mono'; font-size: 0.7rem;">
            END OF TRANSMISSION
        </div>
    </main>
</body>
</html>
    """

    with open(ARCHIVE_FILE, "w", encoding='utf-8') as f:
        f.write(page_html)
    print(f"Archive Index Updated: {ARCHIVE_FILE}")

if __name__ == "__main__":
    generate_archive()
