import os
import re
import glob
import json
import yaml
import hashlib
from datetime import datetime
from collections import Counter

# --- Configuration ---
OUTPUT_DIR = "showcase"
SOURCE_DIRS = [
    "core/libraries_and_archives/newsletters",
    "core/libraries_and_archives/reports"
]
ARCHIVE_FILE = "showcase/market_mayhem_archive.html"

# --- Analysis Helpers ---

def calculate_provenance(content):
    """Generates a SHA-256 hash of the content."""
    return hashlib.sha256(content.encode('utf-8')).hexdigest()[:16]

def analyze_sentiment(text):
    """
    Simple keyword-based sentiment analysis.
    Returns a score between 0 (Extreme Fear) and 100 (Extreme Greed).
    """
    if not text: return 50
    text = text.lower()

    # Keyword Lists
    bullish = ['growth', 'bull', 'rally', 'surge', 'optimism', 'recovery', 'boom', 'record', 'high', 'profit', 'gain', 'support', 'breakout', 'demand', 'strong', 'resilient', 'up', 'buy', 'long']
    bearish = ['crash', 'bear', 'recession', 'collapse', 'crisis', 'panic', 'fear', 'sell', 'down', 'loss', 'debt', 'inflation', 'risk', 'correction', 'drop', 'weak', 'slowdown', 'short', 'volatility']

    score = 50
    for word in re.findall(r'\w+', text):
        if word in bullish: score += 2
        if word in bearish: score -= 2

    return max(0, min(100, score))

def extract_entities(text):
    """Extracts potential Tickers ($AAPL) and Key Concepts."""
    if not text: return []

    # Tickers: $XYZ or (XYZ) usually uppercase
    tickers = re.findall(r'\$([A-Z]{2,5})', text)
    # Also look for (AAPL) patterns common in financial text
    tickers += re.findall(r'\(([A-Z]{2,5})\)', text)

    # Exclude common false positives
    exclude = {'USD', 'YTD', 'YoY', 'QoQ', 'CEO', 'CFO', 'ETF', 'GDP', 'CPI', 'FED', 'SEC', 'FOMC'}
    tickers = [t for t in tickers if t not in exclude]

    return list(set(tickers))

def identify_agent(text, metadata):
    """Identifies the creating agent based on keywords or metadata."""
    if metadata.get('agent'): return metadata['agent']

    text_lower = text.lower()
    if "valuation" in text_lower and "dcf" in text_lower: return "FinancialModelingAgent"
    if "macro" in text_lower or "gdp" in text_lower: return "MacroEconomicAgent"
    if "chart" in text_lower or "technical" in text_lower: return "TechnicalAnalystAgent"
    if "sentiment" in text_lower: return "SentimentAnalysisAgent"
    if "risk" in text_lower or "compliance" in text_lower: return "RiskManagementAgent"
    if "crypto" in text_lower or "blockchain" in text_lower: return "CryptoForensicsAgent"

    return "Adam_Core_v23"

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
        if line.strip().startswith('* ') or line.strip().startswith('- '):
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
    if not date_str: return "2025-01-01"

    if isinstance(date_str, (datetime,)):
        return date_str.strftime("%Y-%m-%d")
    date_str = str(date_str)

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

    match = re.search(r'(\d{4})[-_]?(\d{2})[-_]?(\d{2})', date_str)
    if match:
        return f"{match.group(1)}-{match.group(2)}-{match.group(3)}"

    match_us = re.search(r'(\d{2})[-_]?(\d{2})[-_]?(\d{4})', date_str)
    if match_us:
        return f"{match_us.group(3)}-{match_us.group(1)}-{match_us.group(2)}"

    return date_str

# --- Parsers ---

def format_dict_as_html(data_dict, level=2):
    """Recursively formats a dictionary as HTML."""
    html = ""
    for key, value in data_dict.items():
        title = key.replace('_', ' ').title()
        html += f"<h{level}>{title}</h{level}>"

        if isinstance(value, list):
            html += "<ul>"
            for item in value:
                if isinstance(item, dict):
                    html += "<li>"
                    parts = []
                    for k, v in item.items():
                        parts.append(f"<strong>{k}:</strong> {v}")
                    html += ", ".join(parts) + "</li>"
                else:
                    html += f"<li>{item}</li>"
            html += "</ul>"
        elif isinstance(value, dict):
            html += format_dict_as_html(value, level + 1)
        else:
            html += f"<p>{value}</p>"
    return html

def parse_json_file(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
            # Remove comments for JSON parsing if any (basic JS style)
            clean_content = re.sub(r'//.*', '', content)
            try:
                data = json.loads(clean_content)
            except:
                f.seek(0)
                data = yaml.safe_load(f)

        if not data: return None

        # Extract Title
        title = data.get('title') or data.get('topic') or data.get('report_title')
        if not title:
            company = data.get('company')
            title = f"{company} Report" if company else 'Untitled Report'

        # Extract Date
        date_str = "2025-01-01"
        filename = os.path.basename(filepath)
        date_match = re.search(r'(\d{4}_\d{2}_\d{2})', filename)

        if "date" in data:
             date_str = parse_date(data["date"])
        elif "report_date" in data:
             date_str = parse_date(data["report_date"])
        elif date_match:
            date_str = parse_date(date_match.group(1).replace('_', '-'))

        # Construct Body
        body_html = ""
        summary = data.get('summary', "")

        # Logic for different schemas
        if "sections" in data:
            for section in data['sections']:
                if isinstance(section, dict):
                    sec_title = section.get('title', '')
                    if sec_title: body_html += f"<h2>{sec_title}</h2>"
                    if 'content' in section:
                        body_html += simple_md_to_html(section['content'])
                        if not summary: summary = section['content'][:200] + "..."
                    if 'items' in section:
                        body_html += "<ul>" + "".join([f"<li>{i}</li>" for i in section['items']]) + "</ul>"

        elif "analysis" in data:
            if summary: body_html += f"<h2>Executive Summary</h2><p>{summary}</p>"
            body_html += format_dict_as_html(data['analysis'])
            for key in ['trading_levels', 'financial_performance', 'valuation']:
                if key in data:
                     body_html += f"<h2>{key.replace('_', ' ').title()}</h2>"
                     body_html += format_dict_as_html(data[key], 3) if isinstance(data[key], dict) else f"<p>{data[key]}</p>"

        # Determine Type
        type_ = "NEWSLETTER"
        lower_title = title.lower()
        if "deep dive" in lower_title: type_ = "DEEP_DIVE"
        elif "industry" in lower_title: type_ = "INDUSTRY_REPORT"
        elif "company" in lower_title: type_ = "COMPANY_REPORT"
        elif "thematic" in lower_title: type_ = "THEMATIC_REPORT"

        # Analysis
        raw_text = json.dumps(data)
        sentiment = analyze_sentiment(raw_text)
        entities = extract_entities(raw_text)
        provenance = calculate_provenance(raw_text)
        agent = identify_agent(raw_text, data)

        if "file_name" in data:
            out_filename = data["file_name"].replace('.json', '.html')
        else:
            out_filename = filename.replace('.json', '.html').replace('.md', '.html')

        return {
            "title": title,
            "date": date_str,
            "summary": summary,
            "type": type_,
            "full_body": body_html,
            "sentiment_score": sentiment,
            "entities": entities,
            "provenance": provenance,
            "agent": agent,
            "filename": out_filename,
            "is_sourced": True
        }
    except Exception as e:
        print(f"Error processing {filepath}: {e}")
        return None

def parse_markdown_file(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()

        filename = os.path.basename(filepath)

        # Metadata
        title_match = re.search(r'^# (.*?)$', content, re.MULTILINE)
        title = title_match.group(1).strip() if title_match else filename.replace('.md', '')

        # Date
        date_match = re.search(r'\*\*Date:\*\* (.*?)$', content, re.MULTILINE)
        if date_match:
            date = parse_date(date_match.group(1))
        else:
            date_match_fn = re.search(r'(\d{4})[-_]?(\d{2})[-_]?(\d{2})', filename)
            date = f"{date_match_fn.group(1)}-{date_match_fn.group(2)}-{date_match_fn.group(3)}" if date_match_fn else "2025-01-01"

        # Type
        type_ = "NEWSLETTER"
        if "Deep_Dive" in filename: type_ = "DEEP_DIVE"
        elif "Pulse" in filename: type_ = "MARKET_PULSE"
        elif "Industry" in filename: type_ = "INDUSTRY_REPORT"

        # Summary
        summary_match = re.search(r'(Executive Summary|Summary)\s*\n(.*?)(?=\n##)', content, re.DOTALL | re.IGNORECASE)
        summary = summary_match.group(2).strip()[:300] + "..." if summary_match else "Market analysis."

        # Analysis
        sentiment = analyze_sentiment(content)
        entities = extract_entities(content)
        provenance = calculate_provenance(content)
        agent = identify_agent(content, {})

        # Convert Body
        if title_match: content = content.replace(title_match.group(0), "", 1).strip()
        body_html = simple_md_to_html(content)

        return {
            "title": title,
            "date": date,
            "summary": summary,
            "type": type_,
            "full_body": body_html,
            "sentiment_score": sentiment,
            "entities": entities,
            "provenance": provenance,
            "agent": agent,
            "filename": filename.replace('.md', '.html'),
            "is_sourced": True
        }

    except Exception as e:
        print(f"Error parsing MD {filepath}: {e}")
        return None

# --- Templates ---

REPORT_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ADAM v23.5 :: {title}</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Inter:wght@300;400;600;700&family=Playfair+Display:ital,wght@0,400;0,700;1,400&display=swap" rel="stylesheet">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    <style>
        :root {{
            --paper-bg: #fdfbf7;
            --ink-color: #1a1a1a;
            --cyber-black: #050b14;
            --cyber-blue: #00f3ff;
            --cyber-purple: #a855f7;
        }}
        body {{ background: var(--cyber-black); color: #e0e0e0; font-family: 'Inter', sans-serif; }}

        .paper-sheet {{
            background: var(--paper-bg); color: var(--ink-color);
            padding: 60px; font-family: 'Georgia', serif;
            box-shadow: 0 0 50px rgba(0,0,0,0.5);
        }}

        h1.title {{
            font-family: 'Playfair Display', serif; font-size: 3rem;
            border-bottom: 4px solid var(--ink-color); padding-bottom: 20px; margin-bottom: 30px;
        }}
        h2 {{
            font-family: 'Inter', sans-serif; font-weight: 700; font-size: 1.2rem;
            margin-top: 40px; color: #cc0000; text-transform: uppercase;
        }}

        /* Cyber-HUD Sidebar */
        .cyber-sidebar {{ font-family: 'JetBrains Mono', monospace; }}
        .hud-widget {{
            border: 1px solid rgba(0, 243, 255, 0.2);
            background: rgba(5, 11, 20, 0.8);
            backdrop-filter: blur(10px);
            padding: 20px; margin-bottom: 20px;
            position: relative; overflow: hidden;
        }}
        .hud-widget::before {{
            content: ''; position: absolute; top: 0; left: 0; width: 100%; height: 2px;
            background: linear-gradient(90deg, transparent, var(--cyber-blue), transparent);
        }}
        .hud-title {{ color: var(--cyber-blue); font-size: 0.75rem; text-transform: uppercase; margin-bottom: 10px; }}
        .hud-value {{ color: white; font-size: 0.9rem; }}
        .hud-label {{ color: #64748b; font-size: 0.7rem; }}
    </style>
</head>
<body>
    <div class="fixed top-0 w-full z-50">
        <!-- Global Nav Injection Point -->
    </div>

    <div class="max-w-[1600px] mx-auto p-6 pt-20 grid grid-cols-1 xl:grid-cols-4 gap-8">

        <!-- Left: Navigation/Context -->
        <div class="hidden xl:block col-span-1">
            <a href="market_mayhem_archive.html" class="inline-flex items-center gap-2 text-cyan-400 hover:text-white transition-colors mb-8 font-mono text-sm">
                <i class="fas fa-arrow-left"></i> RETURN TO ARCHIVE
            </a>

            <div class="hud-widget">
                <div class="hud-title">Contributing Agent</div>
                <div class="flex items-center gap-3 mb-2">
                    <div class="w-10 h-10 rounded bg-cyan-900/30 flex items-center justify-center text-cyan-400">
                        <i class="fas fa-robot text-lg"></i>
                    </div>
                    <div>
                        <div class="hud-value font-bold">{agent}</div>
                        <div class="hud-label">v23.5.0 Active</div>
                    </div>
                </div>
                <a href="agents.html" class="text-xs text-cyan-500 hover:text-white block mt-2 text-right">VIEW PROFILE >></a>
            </div>

            <div class="hud-widget">
                <div class="hud-title">Provenance</div>
                <div class="font-mono text-xs text-slate-400 break-all">
                    ID: {provenance}<br>
                    TS: {date}
                </div>
                <div class="mt-3 flex items-center gap-2 text-xs text-green-400">
                    <i class="fas fa-check-circle"></i> VERIFIED
                </div>
            </div>
        </div>

        <!-- Center: The Report -->
        <div class="col-span-1 xl:col-span-2">
            <div class="paper-sheet rounded-sm">
                <div class="flex justify-between font-mono text-xs text-gray-500 mb-8">
                    <span>{date}</span>
                    <span>{type}</span>
                </div>

                <h1 class="title">{title}</h1>

                <div class="prose prose-slate max-w-none">
                    {full_body}
                </div>

                <div class="mt-12 pt-8 border-t border-gray-300 italic text-gray-500 text-sm font-serif text-center">
                    Generated by Adam v23.5 Autonomous Financial System
                </div>
            </div>
        </div>

        <!-- Right: Analysis HUD -->
        <div class="col-span-1">
            <div class="hud-widget">
                <div class="hud-title">Sentiment Analysis</div>
                <div class="flex items-end gap-2 mb-2">
                    <div class="text-3xl font-bold {sent_color}">{sentiment_score}</div>
                    <div class="text-sm text-slate-400 mb-1">/ 100</div>
                </div>
                <div class="h-2 bg-slate-800 rounded-full overflow-hidden">
                    <div class="h-full {sent_bg}" style="width: {sentiment_score}%"></div>
                </div>
                <div class="mt-2 text-xs text-center {sent_color} font-bold">{sent_label}</div>
            </div>

            <div class="hud-widget">
                <div class="hud-title">Detected Entities</div>
                <div class="flex flex-wrap gap-2">
                    {entity_badges}
                </div>
            </div>

            <div class="hud-widget">
                <div class="hud-title">Contextual Links</div>
                <ul class="space-y-2 text-sm text-slate-400">
                    <li class="hover:text-cyan-400 cursor-pointer"><i class="fas fa-link mr-2 text-xs"></i>Sector Performance</li>
                    <li class="hover:text-cyan-400 cursor-pointer"><i class="fas fa-link mr-2 text-xs"></i>Related 13F Filings</li>
                    <li class="hover:text-cyan-400 cursor-pointer"><i class="fas fa-link mr-2 text-xs"></i>Historical Correlation</li>
                </ul>
            </div>
        </div>
    </div>

    <!-- Adam Global Nav -->
    <script src="js/nav.js"></script>
</body>
</html>
"""

ARCHIVE_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ADAM v23.5 | Market Mayhem Archive</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&display=swap" rel="stylesheet">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    <style>
        body {{ background-color: #050b14; color: #e2e8f0; font-family: 'JetBrains Mono', monospace; }}
        .glass-panel {{
            background: rgba(13, 20, 31, 0.7); backdrop-filter: blur(10px);
            border: 1px solid rgba(0, 243, 255, 0.1); box-shadow: 0 0 15px rgba(0, 243, 255, 0.05);
        }}
        .metric-value {{ text-shadow: 0 0 10px rgba(0, 243, 255, 0.5); }}
    </style>
</head>
<body class="min-h-screen">

    <div class="max-w-[1600px] mx-auto p-6 md:pl-72 transition-all duration-300">

        <!-- Header -->
        <header class="flex justify-between items-end mb-8 pb-4 border-b border-slate-800">
            <div>
                <h1 class="text-4xl font-bold text-white tracking-tighter mb-1">MARKET MAYHEM</h1>
                <div class="flex items-center gap-3">
                    <span class="px-2 py-0.5 bg-red-900/30 border border-red-800 text-red-400 text-xs rounded">CLASSIFIED</span>
                    <span class="text-slate-500 text-xs">DEEP ARCHIVE PROTOCOL v2.1</span>
                </div>
            </div>
            <div class="text-right">
                <div class="text-xs text-slate-500 mb-1">TOTAL RECORDS</div>
                <div class="text-2xl font-bold text-cyan-400 metric-value">{total_records}</div>
            </div>
        </header>

        <!-- Dashboard Widgets -->
        <div class="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-8">

            <!-- Sentiment Chart -->
            <div class="glass-panel p-4 rounded-lg col-span-2">
                <h3 class="text-sm text-cyan-400 mb-4 font-bold flex items-center gap-2">
                    <i class="fas fa-chart-line"></i> SENTIMENT TIMELINE
                </h3>
                <div class="h-64">
                    <canvas id="sentimentChart"></canvas>
                </div>
            </div>

            <!-- Top Entities -->
            <div class="glass-panel p-4 rounded-lg">
                <h3 class="text-sm text-cyan-400 mb-4 font-bold flex items-center gap-2">
                    <i class="fas fa-tags"></i> ENTITY HEATMAP
                </h3>
                <div class="flex flex-wrap gap-2 content-start h-64 overflow-y-auto custom-scrollbar">
                    {entity_cloud}
                </div>
            </div>
        </div>

        <!-- Filter Bar -->
        <div class="flex gap-4 mb-6 sticky top-4 z-30 glass-panel p-2 rounded">
            <div class="relative flex-grow">
                <i class="fas fa-search absolute left-3 top-3 text-slate-500 text-xs"></i>
                <input type="text" id="searchInput" placeholder="Search report titles, entities, summaries..."
                    class="w-full bg-slate-900/50 border border-slate-700 rounded px-10 py-2 text-sm focus:border-cyan-500 focus:outline-none transition-colors text-white"
                    onkeyup="filterGrid()">
            </div>
            <select id="yearFilter" onchange="filterGrid()" class="bg-slate-900/50 border border-slate-700 rounded px-4 py-2 text-sm text-slate-300 focus:border-cyan-500 focus:outline-none">
                <option value="ALL">All Years</option>
                {year_options}
            </select>
        </div>

        <!-- Report Grid -->
        <div class="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-4" id="reportGrid">
            {grid_items}
        </div>

        <footer class="mt-20 text-center text-slate-600 text-xs border-t border-slate-800 pt-8">
            ADAM FINANCIAL SYSTEM // PROVENANCE VERIFIED
        </footer>
    </div>

    <!-- Data Injection -->
    <script>
        const SENTIMENT_DATA = {chart_json};

        // Initialize Chart
        const ctx = document.getElementById('sentimentChart').getContext('2d');
        new Chart(ctx, {{
            type: 'line',
            data: {{
                labels: SENTIMENT_DATA.labels,
                datasets: [{{
                    label: 'Market Sentiment',
                    data: SENTIMENT_DATA.data,
                    borderColor: '#00f3ff',
                    backgroundColor: 'rgba(0, 243, 255, 0.1)',
                    borderWidth: 2,
                    tension: 0.4,
                    pointRadius: 2,
                    fill: true
                }}]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                plugins: {{ legend: {{ display: false }} }},
                scales: {{
                    y: {{ min: 0, max: 100, grid: {{ color: '#1e293b' }} }},
                    x: {{ grid: {{ display: false }} }}
                }}
            }}
        }});

        function filterGrid() {{
            const query = document.getElementById('searchInput').value.toLowerCase();
            const year = document.getElementById('yearFilter').value;
            const items = document.querySelectorAll('.report-card');

            items.forEach(item => {{
                const text = item.innerText.toLowerCase();
                const itemYear = item.getAttribute('data-year');

                const matchesSearch = text.includes(query);
                const matchesYear = year === 'ALL' || itemYear === year;

                if (matchesSearch && matchesYear) {{
                    item.style.display = 'block';
                }} else {{
                    item.style.display = 'none';
                }}
            }});
        }}
    </script>
    <script src="js/nav.js"></script>
</body>
</html>
"""

# --- Main Generation ---

def generate_archive():
    print("Starting Archive Generation...")

    # 1. Gather all data
    items = []

    # Scan source directories
    for source_dir in SOURCE_DIRS:
        if not os.path.exists(source_dir): continue
        files = glob.glob(os.path.join(source_dir, "*"))
        for f in files:
            item = None
            if f.endswith(".json"): item = parse_json_file(f)
            elif f.endswith(".md"): item = parse_markdown_file(f)
            if item: items.append(item)

    # 2. Process Items
    items.sort(key=lambda x: x['date'], reverse=True)

    # Aggregation for Dashboard
    all_entities = []
    sentiment_history = []

    grid_html = ""
    years = set()

    for item in items:
        # Generate Individual Page
        out_path = os.path.join(OUTPUT_DIR, item['filename'])

        # Prepare HUD Data
        sent_score = item['sentiment_score']
        if sent_score > 60:
            sent_color = "text-green-400"
            sent_bg = "bg-green-500"
            sent_label = "BULLISH"
        elif sent_score < 40:
            sent_color = "text-red-400"
            sent_bg = "bg-red-500"
            sent_label = "BEARISH"
        else:
            sent_color = "text-yellow-400"
            sent_bg = "bg-yellow-500"
            sent_label = "NEUTRAL"

        entity_badges = ""
        for ent in item['entities'][:8]: # Limit to top 8
            entity_badges += f'<span class="px-2 py-1 bg-slate-800 text-cyan-400 rounded text-xs border border-slate-700">{ent}</span>'

        all_entities.extend(item['entities'])
        sentiment_history.append({"date": item['date'], "score": sent_score})
        years.add(item['date'].split('-')[0])

        with open(out_path, "w", encoding='utf-8') as f:
            f.write(REPORT_TEMPLATE.format(
                title=item['title'],
                date=item['date'],
                type=item['type'],
                full_body=item['full_body'],
                agent=item['agent'],
                provenance=item['provenance'],
                sentiment_score=sent_score,
                sent_color=sent_color,
                sent_bg=sent_bg,
                sent_label=sent_label,
                entity_badges=entity_badges
            ))

        # Generate Grid Card HTML
        year = item['date'].split('-')[0]
        grid_html += f"""
        <div class="report-card glass-panel p-5 rounded hover:border-cyan-500/50 transition-colors group cursor-pointer" data-year="{year}" onclick="window.location.href='{item['filename']}'">
            <div class="flex justify-between items-start mb-3">
                <span class="text-xs font-mono text-slate-500">{item['date']}</span>
                <span class="text-[10px] px-2 py-0.5 rounded bg-slate-800 text-slate-300 border border-slate-700">{item['type']}</span>
            </div>
            <h4 class="text-white font-bold text-lg mb-2 group-hover:text-cyan-400 transition-colors leading-tight">{item['title']}</h4>
            <p class="text-sm text-slate-400 line-clamp-3 mb-4">{item['summary']}</p>
            <div class="flex gap-2 flex-wrap">
                {entity_badges}
            </div>
        </div>
        """

    # 3. Generate Dashboard
    # Entity Cloud
    entity_counts = Counter(all_entities).most_common(20)
    entity_cloud_html = ""
    for ent, count in entity_counts:
        size = "text-xs"
        if count > 5: size = "text-lg"
        elif count > 2: size = "text-sm"

        entity_cloud_html += f'<span class="{size} px-2 py-1 bg-slate-800/50 text-cyan-300 rounded border border-slate-700/50">{ent} <span class="text-slate-500 text-[10px]">{count}</span></span> '

    # Chart Data (Reverse to chronological)
    sentiment_history.sort(key=lambda x: x['date'])
    chart_data = {
        "labels": [x['date'] for x in sentiment_history],
        "data": [x['score'] for x in sentiment_history]
    }

    # Year Options
    year_options = ""
    for y in sorted(list(years), reverse=True):
        year_options += f'<option value="{y}">{y}</option>'

    # Write Archive Index
    with open(ARCHIVE_FILE, "w", encoding='utf-8') as f:
        f.write(ARCHIVE_TEMPLATE.format(
            total_records=len(items),
            grid_items=grid_html,
            entity_cloud=entity_cloud_html,
            year_options=year_options,
            chart_json=json.dumps(chart_data)
        ))

    print(f"Archive generation complete. {len(items)} items processed.")

if __name__ == "__main__":
    generate_archive()
