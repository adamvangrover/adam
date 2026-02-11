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
SHOWCASE_DIR = "showcase"
ARCHIVE_FILE = "showcase/market_mayhem_archive.html"

# --- Analysis Helpers ---

POSITIVE_WORDS = {"growth", "profit", "boom", "bullish", "strong", "gain", "rally", "opportunity", "breakthrough", "resilient", "optimism", "surge", "upward", "recovery"}
NEGATIVE_WORDS = {"loss", "crash", "recession", "bearish", "weak", "risk", "inflation", "crisis", "downturn", "collapse", "uncertainty", "volatile", "fear", "panic", "decline"}

def analyze_sentiment(text):
    """Calculates a sentiment score (0-100) based on keyword density."""
    if not text: return 50
    text_lower = text.lower()
    words = re.findall(r'\w+', text_lower)
    if not words: return 50

    pos_count = sum(1 for w in words if w in POSITIVE_WORDS)
    neg_count = sum(1 for w in words if w in NEGATIVE_WORDS)

    total_matches = pos_count + neg_count
    if total_matches == 0: return 50

    # Scale: 0 (All Neg) to 100 (All Pos)
    # Base is 50.
    score = (pos_count / total_matches) * 100
    return int(score)

def extract_entities(text):
    """Extracts Tickers ($AAPL) and Agents (@AgentName) and potential Sovereigns."""
    tickers = sorted(list(set(re.findall(r'\$([A-Z]{2,5})', text))))
    agents = sorted(list(set(re.findall(r'@([A-Za-z0-9_]+)', text))))

    # Simple heuristic for Sovereigns/Countries
    sovereigns = []
    if "United States" in text or "US" in text: sovereigns.append("US")
    if "China" in text: sovereigns.append("CN")
    if "Europe" in text or "EU" in text: sovereigns.append("EU")

    return {
        "tickers": tickers,
        "agents": agents,
        "sovereigns": sovereigns
    }

def calculate_provenance(text):
    """Generates a SHA-256 hash of the content."""
    return hashlib.sha256(text.encode('utf-8')).hexdigest()

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
        "%m%d%Y",          # 09192025
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
            try:
                data = yaml.safe_load(f)
            except yaml.YAMLError as ye:
                f.seek(0)
                content = f.read()
                content = re.sub(r'//.*', '', content)
                try:
                    data = json.loads(content)
                except json.JSONDecodeError:
                     return None

        if not data: return None

        # Extract Title
        title = data.get('title') or data.get('topic') or data.get('report_title')
        if not title:
            company = data.get('company')
            if company:
                title = f"{company} Report"
            else:
                title = 'Untitled Report'

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
        else:
             date_match_2 = re.search(r'(\d{4})[-_]?(\d{2})[-_]?(\d{2})', filename)
             if date_match_2:
                 date_str = f"{date_match_2.group(1)}-{date_match_2.group(2)}-{date_match_2.group(3)}"

        # Construct Body
        body_html = ""
        summary = data.get('summary', "")

        # Financial Metrics for Visualization (if available)
        metrics_json = "{}"
        if "financial_performance" in data and "metrics" in data["financial_performance"]:
             metrics_json = json.dumps(data["financial_performance"]["metrics"])
        elif "metrics" in data:
             metrics_json = json.dumps(data["metrics"])

        # Schema 1: 'sections' list
        sections = data.get('sections', [])
        if sections:
            for section in sections:
                if isinstance(section, dict):
                    sec_title = section.get('title', '')
                    if sec_title: body_html += f"<h2>{sec_title}</h2>"

                    if 'content' in section:
                        body_html += simple_md_to_html(section['content'])
                        if not summary and ("Summary" in sec_title or not summary):
                            summary = section['content'][:200] + "..."

                    if 'items' in section:
                        body_html += "<ul>"
                        for item in section['items']:
                            body_html += f"<li>{item}</li>"
                        body_html += "</ul>"

                    if 'metrics' in section:
                         # Capture metrics for this section for the body, but also potential chart
                         body_html += format_dict_as_html(section['metrics'], 3)


        # Schema 2: 'analysis' dict
        elif "analysis" in data:
            if summary: body_html += f"<h2>Executive Summary</h2><p>{summary}</p>"
            body_html += format_dict_as_html(data['analysis'])

            for key in ['trading_levels', 'financial_performance', 'valuation']:
                if key in data and key not in data['analysis']:
                     body_html += f"<h2>{key.replace('_', ' ').title()}</h2>"
                     if isinstance(data[key], dict):
                         body_html += format_dict_as_html(data[key], 3)
                     else:
                         body_html += f"<p>{data[key]}</p>"

        # Schema 3: Generic
        elif summary:
             body_html += f"<p>{summary}</p>"
             for k, v in data.items():
                 if k not in ['title', 'date', 'summary', 'file_name', 'company', 'analyst'] and isinstance(v, str):
                     body_html += f"<h3>{k.replace('_', ' ').title()}</h3><p>{v}</p>"

        if not summary: summary = f"Market Report from {date_str}"

        # Determine Type
        type_ = "NEWSLETTER"
        lower_title = title.lower()
        if "deep dive" in lower_title: type_ = "DEEP_DIVE"
        elif "industry" in lower_title: type_ = "INDUSTRY_REPORT"
        elif "company" in lower_title or "report" in lower_title: type_ = "COMPANY_REPORT"
        elif "thematic" in lower_title: type_ = "THEMATIC_REPORT"
        elif "outlook" in lower_title or "review" in lower_title: type_ = "MARKET_OUTLOOK"

        # Determine filename
        if "file_name" in data:
            out_filename = data["file_name"].replace('.json', '.html')
        else:
            out_filename = filename.replace('.json', '.html').replace('.md', '.html')

        # Analysis
        full_text = f"{title} {summary} {body_html}"
        sentiment = analyze_sentiment(full_text)
        entities = extract_entities(full_text)
        prov_hash = calculate_provenance(full_text)

        return {
            "title": title,
            "date": date_str,
            "summary": summary,
            "type": type_,
            "full_body": body_html,
            "sentiment_score": sentiment,
            "entities": entities,
            "provenance_hash": prov_hash,
            "filename": out_filename,
            "is_sourced": True,
            "metrics_json": metrics_json
        }
    except Exception as e:
        print(f"Error processing {filepath}: {e}")
        return None

def parse_txt_file(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()

        filename = os.path.basename(filepath)

        title_match = re.search(r'COMPREHENSIVE.*?:\s*(.*?)$', content, re.MULTILINE)
        title = title_match.group(1).strip() if title_match else filename.replace('.txt', '').replace('_', ' ')

        date_match = re.search(r'DATE:\s*(.*?)$', content, re.MULTILINE)
        date_str = parse_date(date_match.group(1)) if date_match else "2025-01-01"

        type_ = "REPORT"
        if "Deep Dive" in content or "DEEP DIVE" in content: type_ = "DEEP_DIVE"

        body_html = f"<div style='white-space: pre-wrap; font-family: inherit;'>{content}</div>"

        summary = "Comprehensive report."
        summary_match = re.search(r'EXECUTIVE SUMMARY.*?\n\n(.*?)\n', content, re.DOTALL)
        if summary_match:
             summary = summary_match.group(1)[:300] + "..."

        # Analysis
        sentiment = analyze_sentiment(content)
        entities = extract_entities(content)
        prov_hash = calculate_provenance(content)

        return {
            "title": title,
            "date": date_str,
            "summary": summary,
            "type": type_,
            "full_body": body_html,
            "sentiment_score": sentiment,
            "entities": entities,
            "provenance_hash": prov_hash,
            "filename": filename.replace('.txt', '.html'),
            "is_sourced": True,
            "metrics_json": "{}"
        }
    except Exception as e:
        print(f"Error parsing TXT {filepath}: {e}")
        return None

def parse_markdown_file(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()

        filename = os.path.basename(filepath)

        title_match = re.search(r'^# (.*?)$', content, re.MULTILINE)
        title = title_match.group(1).strip() if title_match else filename.replace('.md', '')

        date_match = re.search(r'\*\*Date:\*\* (.*?)$', content, re.MULTILINE)
        if not date_match:
             date_match = re.search(r'(\d{4})[-_]?(\d{2})[-_]?(\d{2})', filename)
             if date_match:
                 raw_date = f"{date_match.group(1)}-{date_match.group(2)}-{date_match.group(3)}"
             else:
                 raw_date = "2025-01-01"
        else:
            raw_date = date_match.group(1)

        date = parse_date(raw_date)

        if "Deep_Dive" in filename: type_ = "DEEP_DIVE"
        elif "Pulse" in filename or "Pulse" in title: type_ = "MARKET_PULSE"
        elif "Industry" in filename: type_ = "INDUSTRY_REPORT"
        elif "Weekly" in filename: type_ = "WEEKLY_RECAP"
        elif "Tech_Watch" in filename: type_ = "TECH_WATCH"
        elif "Agent" in filename or "Alignment" in title: type_ = "AGENT_NOTE"
        elif "Performance" in filename or "Audit" in title: type_ = "PERFORMANCE_REVIEW"
        else: type_ = "NEWSLETTER"

        summary_match = re.search(r'(Executive Summary|Summary)\s*\n(.*?)(?=\n##)', content, re.DOTALL | re.IGNORECASE)
        if summary_match:
            summary_raw = summary_match.group(2).strip()
            summary = re.sub(r'[*#]', '', summary_raw)[:200] + "..."
        else:
            summary = "Market analysis and strategic insights."

        if title_match:
             content = content.replace(title_match.group(0), "", 1).strip()

        body_html = simple_md_to_html(content)

        # Analysis
        sentiment = analyze_sentiment(content)
        entities = extract_entities(content)
        prov_hash = calculate_provenance(content)

        return {
            "title": title,
            "date": date,
            "summary": summary,
            "type": type_,
            "full_body": body_html,
            "sentiment_score": sentiment,
            "entities": entities,
            "provenance_hash": prov_hash,
            "filename": filename.replace('.md', '.html').replace('.json', '.html'),
            "is_sourced": True,
            "metrics_json": "{}"
        }

    except Exception as e:
        print(f"Error parsing MD {filepath}: {e}")
        return None

def parse_html_file(filepath):
    """Parses metadata from existing HTML files in showcase."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()

        filename = os.path.basename(filepath)

        # Skip indices and dashboards
        if filename in ["index.html", "market_mayhem_archive.html", "daily_briefings_library.html",
                        "market_pulse_library.html", "house_view_library.html", "portfolio_dashboard.html",
                        "data.html", "reports.html", "agents.html", "chat.html"]:
            return None

        # Extract Title
        title_match = re.search(r'<title>(?:ADAM v23\.5 :: )?(.*?)</title>', content, re.IGNORECASE)
        title = title_match.group(1).strip() if title_match else filename.replace('.html', '').replace('_', ' ')

        # Determine Date (Filename Priority -> Content Regex)
        date = "2025-01-01" # Default

        # 1. MMXXXXYYYY format (e.g. MM09192025)
        mm_match = re.search(r'MM(\d{2})(\d{2})(\d{4})', filename)
        if mm_match:
            date = f"{mm_match.group(3)}-{mm_match.group(1)}-{mm_match.group(2)}"
        else:
            # 2. Standard patterns
            date_match = re.search(r'(\d{4})[-_]?(\d{2})[-_]?(\d{2})', filename)
            if date_match:
                date = f"{date_match.group(1)}-{date_match.group(2)}-{date_match.group(3)}"
            else:
                # 3. Content search
                content_date = re.search(r'(\w+ \d{1,2}, \d{4})', content)
                if content_date:
                    date = parse_date(content_date.group(1))

        # Determine Type
        type_ = "NEWSLETTER"
        lower_title = title.lower()
        if "deep dive" in lower_title or "deep_dive" in filename: type_ = "DEEP_DIVE"
        elif "industry" in lower_title: type_ = "INDUSTRY_REPORT"
        elif "company" in lower_title: type_ = "COMPANY_REPORT"
        elif "pulse" in lower_title: type_ = "MARKET_PULSE"
        elif "glitch" in lower_title: type_ = "CYBER_GLITCH"
        elif "outlook" in lower_title or "strategy" in lower_title: type_ = "STRATEGY"
        elif "snc" in lower_title: type_ = "GUIDE"

        # Sentiment Analysis
        sentiment = analyze_sentiment(content)
        entities = extract_entities(content)
        prov_hash = calculate_provenance(content)

        return {
            "title": title,
            "date": date,
            "summary": "Report content.",
            "type": type_,
            "full_body": "", # No need to re-render body for existing HTML
            "sentiment_score": sentiment,
            "entities": entities,
            "provenance_hash": prov_hash,
            "filename": filename,
            "is_sourced": False, # Do not overwrite existing HTML
            "metrics_json": "{}"
        }

    except Exception as e:
        print(f"Error parsing HTML {filepath}: {e}")
        return None

# --- Static Data ---

NEWSLETTER_DATA = [
    {
        "date": "2026-01-12",
        "title": "GLOBAL MACRO-STRATEGIC OUTLOOK 2026: THE REFLATIONARY AGENTIC BOOM",
        "summary": "As markets open on January 12, 2026, the global financial system has entered a new regime: the Reflationary Agentic Boom.",
        "type": "ANNUAL_STRATEGY",
        "filename": "newsletter_market_mayhem_jan_2026.html",
        "is_sourced": True,
        "full_body": """
        <h3>1. Executive Intelligence Summary: The Architecture of the New Regime</h3>
        <p>As markets open on January 12, 2026, the global financial system has decisively exited the post-pandemic transitional phase and entered a new, distinct market regime: the <strong>Reflationary Agentic Boom</strong>. This paradigm is defined by a paradoxical but potent combination of accelerating economic growth in the United States, sticky inflation floors driven by geopolitical fragmentation and tariffs, and a technological productivity shock moving from generative experimentation to "agentic" execution.</p>
        <p>The prevailing narrative of late 2024 and 2025—that the Federal Reserve's tightening cycle would inevitably induce a recession—has been falsified by the data. Instead, the US economy is tracking toward a robust 2.5% to 2.6% real GDP growth rate for 2026. This resilience is not merely a cyclical rebound but a structural shift powered by three pillars: the fiscal impulse of anticipated tax cuts, the capital expenditure (Capex) super-cycle associated with "Sovereign AI," and the integration of digital assets into the institutional balance sheet via new accounting standards.</p>

        <h3>2. Macroeconomic Dynamics: The "No Landing" Reality</h3>
        <p>The defining macroeconomic surprise of 2026 is the persistent exceptionalism of the United States economy. While consensus forecasts in 2024 predicted a hard landing, the actual trajectory has been one of re-acceleration. Current data indicates that the US is poised to outperform all other major developed economies.</p>
        <p><strong>The Inflation Paradox:</strong> While growth is robust, the inflation battle has morphed rather than ended. We are currently witnessing a clash between two opposing secular forces: the inflationary pressure of geopolitical fragmentation and tariffs, versus the deflationary pressure of technological automation.</p>

        <h3>3. The Sovereign AI Paradigm</h3>
        <p>The most significant thematic evolution in 2026 is the shift from corporate AI adoption to "Sovereign AI." This concept posits that artificial intelligence infrastructure—data centers, foundation models, and the energy grids that power them—is not merely a commercial asset but a critical component of national security. Nations are building "AI Factories" to secure their digital borders.</p>
        <p><strong>Implication:</strong> The "Sovereign AI Capex Floor" provides a baseline of demand for hardware (NVDA) and operating systems (PLTR) that is inelastic to interest rates.</p>

        <h3>4. The Agentic Technology Revolution</h3>
        <p>We are witnessing the transition from "Generative AI"—tools that create content—to "Agentic AI"—systems that execute work. This "Binary Big Bang" is reshaping labor markets and corporate efficiency. By automating cognitive labor, companies are reducing their operating leverage and suppressing wage inflation in white-collar sectors. This "Tech Deflation" is the counterweight to "Tariff Inflation".</p>

        <h3>5. The New Crypto-Financial Architecture</h3>
        <p>The year 2026 marks the definitive integration of cryptocurrency into the global financial architecture. The implementation of FASB ASU 2023-08 has unlocked corporate treasuries, allowing Bitcoin to be held at fair value. Combined with rumors of sovereign wealth accumulation (Qatar, Peru), this has established a valuation floor for Bitcoin above $100,000.</p>

        <h3>8. Strategic Asset Allocation: The "Barbell" Strategy</h3>
        <p>Given the confluence of "No Landing" growth, sticky inflation, and the Agentic Boom, a "Barbell" asset allocation strategy is optimal:</p>
        <ul>
            <li><strong>Leg 1 (Agentic Growth):</strong> Overweight AI Infrastructure (NVDA) and Sovereign Operating Systems (PLTR).</li>
            <li><strong>Leg 2 (Sovereignty Hedge):</strong> Overweight Bitcoin (BTC) and Industrial Commodities (Copper).</li>
            <li><strong>Underweight:</strong> Regional Banks and Consumer Discretionary firms reliant on cheap debt.</li>
        </ul>
        """
    },
     {
        "date": "2008-09-19",
        "title": "MARKET MAYHEM: THE LEHMAN MOMENT",
        "summary": "\"Existential Panic\". There are decades where nothing happens; and there are weeks where decades happen. This was one of those weeks. A 158-year-old bank vanished, the world's largest insurer was nationalized, and the money market broke the buck.",
        "type": "HISTORICAL",
        "filename": "newsletter_market_mayhem_sep_2008.html",
        "is_sourced": True,
        "full_body": """
        <p><strong>The Week Wall Street Died.</strong> On Monday, September 15th, Lehman Brothers filed for the largest bankruptcy in U.S. history ($600B+ assets). The government let them fail, hoping to reduce moral hazard. The result was global panic.</p>
        <p>By Tuesday, AIG—the insurer of the world's financial system via CDS—was on the brink. The Fed stepped in with an $85B revolving credit facility, effectively nationalizing the company.</p>
        <p><strong>The Real Panic:</strong> The Reserve Primary Fund, a money market fund considered 'as good as cash', broke the buck (NAV fell to $0.97) due to Lehman exposure. This triggered a $140B run on money market funds, freezing the commercial paper market. The gears of capitalism have ground to a halt.</p>
        """
    },
    {
        "date": "1987-10-23",
        "title": "MARKET MAYHEM: BLACK MONDAY AFTERMATH",
        "summary": "\"Shell-Shocked\". On October 19th, the Dow Jones Industrial Average fell 22.6% in a single day. 508 points. It was the largest one-day percentage drop in history.",
        "type": "HISTORICAL",
        "filename": "newsletter_market_mayhem_oct_1987.html",
        "is_sourced": True,
        "full_body": """
        <p><strong>The Crash.</strong> Monday, October 19th, will live in infamy. The Dow Jones Industrial Average collapsed 508 points, losing 22.6% of its value in a single session. Volume on the NYSE reached an unprecedented 604 million shares, leaving the ticker tape hours behind.</p>
        <p><strong>The Culprit:</strong> Program trading. 'Portfolio Insurance' strategies, designed to sell futures as the market falls to hedge portfolios, kicked in simultaneously. This selling pressure crushed the futures market, which dragged down the spot market in a vicious spiral.</p>
        <p><strong>The Aftermath:</strong> Alan Greenspan's Fed has issued a statement: 'The Federal Reserve, consistent with its responsibilities as the Nation's central bank, affirmed today its readiness to serve as a source of liquidity to support the economic and financial system.' The bleeding has stopped, but the scar remains.</p>
        """
    },
     {
        "date": "2020-03-20",
        "title": "MARKET MAYHEM: THE GREAT SHUT-IN",
        "summary": "\"Lockdown\". The global economy has come to a screeching halt. With \"15 Days to Slow the Spread\" in effect, markets are pricing in a depression-level GDP contraction.",
        "type": "HISTORICAL",
        "filename": "newsletter_market_mayhem_mar_2020.html",
        "is_sourced": True,
        "full_body": """
        <p><strong>The World Has Stopped.</strong> In an unprecedented event, the global economy has entered a medically-induced coma. The S&P 500 has crashed 34% from its February highs, the fastest bear market in history.</p>
        <p>Volatility is off the charts. The VIX closed at 82.69 on March 16th, surpassing the 2008 peak. Credit spreads have blown out, and liquidity in the Treasury market—usually the deepest in the world—has evaporated.</p>
        <p><strong>Oil Shock:</strong> Demand destruction is so severe that WTI crude futures are trading at imminent risk of turning negative due to storage capacity constraints. (Update: They did, hitting -$37.63 in April).</p>
        <p><strong>Central Bank Response:</strong> The Fed has unleashed 'Unlimited QE', buying corporate bonds for the first time in history. The mantra is 'Don't Fight the Fed', but the economic data is catastrophic.</p>
        """
    }
]

# --- Report Template (Enhanced with Cyber HUD) ---

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ADAM v23.5 :: {title}</title>
    <link rel="stylesheet" href="css/style.css">
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
    <link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Inter:wght@300;400;600;700&family=Playfair+Display:ital,wght@0,400;0,700;1,400&display=swap" rel="stylesheet">
    <script src="https://cdn.tailwindcss.com"></script>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <script src="js/nav.js" defer></script>
    <style>
        :root {{
            --paper-bg: #fdfbf7;
            --ink-color: #1a1a1a;
            --accent-red: #cc0000;
            --cyber-black: #050b14;
            --cyber-blue: #00f3ff;
            --cyber-green: #00ff00;
        }}
        body {{ margin: 0; background: var(--cyber-black); color: #e0e0e0; font-family: 'Inter', sans-serif; }}

        .newsletter-wrapper {{
            max-width: 1200px;
            margin: 40px auto;
            display: grid;
            grid-template-columns: 3fr 1fr;
            gap: 40px;
            padding: 20px;
        }}
        @media (max-width: 768px) {{ .newsletter-wrapper {{ grid-template-columns: 1fr; }} }}

        .paper-sheet {{
            background: var(--paper-bg);
            color: var(--ink-color);
            padding: 60px;
            font-family: 'Georgia', 'Times New Roman', serif;
            box-shadow: 0 0 50px rgba(0,0,0,0.5);
            position: relative;
        }}

        h1.title {{
            font-family: 'Playfair Display', serif; font-size: 3rem; border-bottom: 4px solid var(--ink-color);
            padding-bottom: 20px; margin-bottom: 30px; letter-spacing: -1px; line-height: 1.1;
        }}
        h2 {{
            font-family: 'Inter', sans-serif; font-weight: 700; font-size: 1.2rem; margin-top: 40px;
            margin-bottom: 15px; color: var(--accent-red); text-transform: uppercase; letter-spacing: 1px;
        }}
        p {{ line-height: 1.8; margin-bottom: 20px; font-size: 1.05rem; }}
        li {{ line-height: 1.6; margin-bottom: 10px; }}

        /* Cyber HUD Sidebar */
        .cyber-sidebar {{ font-family: 'JetBrains Mono', monospace; }}
        .sidebar-widget {{
            border: 1px solid #333; background: rgba(5, 11, 20, 0.8); padding: 15px; margin-bottom: 20px;
            backdrop-filter: blur(5px);
        }}
        .sidebar-title {{
            color: var(--cyber-blue); font-size: 0.75rem; text-transform: uppercase; border-bottom: 1px solid #333;
            padding-bottom: 8px; margin-bottom: 12px; letter-spacing: 1px;
        }}
        .stat-row {{ display: flex; justify-content: space-between; margin-bottom: 8px; font-size: 0.8rem; color: #aaa; }}
        .stat-val {{ color: #fff; font-weight: bold; }}

        .sentiment-bar {{ height: 4px; background: #333; margin-top: 5px; border-radius: 2px; overflow: hidden; }}
        .sentiment-fill {{ height: 100%; background: var(--cyber-blue); }}

        .tag-cloud {{ display: flex; flex-wrap: wrap; gap: 5px; }}
        .tag {{ background: #222; color: #ccc; padding: 2px 6px; font-size: 0.7rem; border-radius: 3px; border: 1px solid #444; }}
        .tag.ticker {{ color: var(--cyber-green); border-color: var(--cyber-green); }}
        .tag.agent {{ color: var(--cyber-blue); border-color: var(--cyber-blue); }}

        .cyber-btn {{
            font-family: 'JetBrains Mono', monospace; font-size: 0.75rem; padding: 8px 16px; border: 1px solid #444;
            color: #e0e0e0; background: rgba(0,0,0,0.8); text-decoration: none; display: inline-block;
        }}
        .cyber-btn:hover {{ border-color: var(--cyber-blue); color: var(--cyber-blue); }}
    </style>
</head>
<body>
    <div style="max-width: 1400px; margin: 0 auto; padding: 20px; display:flex; justify-content:space-between; align-items:center;">
        <a href="market_mayhem_archive.html" class="cyber-btn">&larr; RETURN TO ARCHIVE</a>
        <div style="font-family:'JetBrains Mono'; font-size:0.7rem; color:#666;">SECURE CONNECTION ESTABLISHED</div>
    </div>

    <div class="newsletter-wrapper">
        <div class="paper-sheet">
            <div style="display:flex; justify-content:space-between; font-family:'JetBrains Mono'; font-size:0.8rem; color:#666; margin-bottom:20px;">
                <span>{date}</span>
                <span>ID: {prov_short}</span>
            </div>

            <h1 class="title">{title}</h1>

            <!-- Dynamic Chart Injection Area if metrics exist -->
            <div id="financialChartContainer" style="display:none; margin-bottom: 30px;">
                <canvas id="financialChart"></canvas>
            </div>

            {full_body}

            <div style="margin-top: 40px; border-top: 1px solid #ccc; padding-top: 20px; font-style: italic; color: #666;">
                End of Transmission.
            </div>
        </div>

        <aside class="cyber-sidebar">
            <div class="sidebar-widget">
                <div class="sidebar-title">Intelligence Metadata</div>
                <div class="stat-row"><span>DATE</span> <span class="stat-val">{date}</span></div>
                <div class="stat-row"><span>TYPE</span> <span class="stat-val">{type}</span></div>
                <div class="stat-row"><span>CLASS</span> <span class="stat-val">UNCLASSIFIED</span></div>
                <div class="stat-row"><span>PROVENANCE</span> <span class="stat-val" title="{prov_hash}">{prov_short}</span></div>
            </div>

            <div class="sidebar-widget">
                <div class="sidebar-title">Sentiment Analysis</div>
                <div class="stat-row"><span>SCORE</span> <span class="stat-val">{sentiment_score}/100</span></div>
                <div class="sentiment-bar">
                    <div class="sentiment-fill" style="width: {sentiment_score}%;"></div>
                </div>
            </div>

            <div class="sidebar-widget">
                <div class="sidebar-title">Detected Entities</div>
                <div class="tag-cloud">
                    {entity_html}
                </div>
            </div>

             <div class="sidebar-widget">
                <div class="sidebar-title">Contributing Agents</div>
                 <div style="display:flex; flex-direction:column; gap:5px;">
                    {agent_html}
                </div>
            </div>
        </aside>
    </div>

    <script>
        // Chart Injection Logic
        const metricsData = {metrics_json};

        if (metricsData && Object.keys(metricsData).length > 0) {{
            const ctx = document.getElementById('financialChart');
            const container = document.getElementById('financialChartContainer');

            // Basic parsing of metrics structure (assuming flat or Year->Val)
            let labels = [];
            let values = [];
            let labelName = "Metric";

            // Try to find a timeseries
            for (const [key, val] of Object.entries(metricsData)) {{
                if (typeof val === 'object') {{
                    // Likely year -> value
                    labels = Object.keys(val);
                    values = Object.values(val).map(v => parseFloat(v.replace(/[^0-9.-]/g, '')));
                    labelName = key;
                    break;
                }}
            }}

            if (labels.length > 0) {{
                container.style.display = 'block';
                new Chart(ctx, {{
                    type: 'bar',
                    data: {{
                        labels: labels,
                        datasets: [{{
                            label: labelName,
                            data: values,
                            backgroundColor: 'rgba(204, 0, 0, 0.5)',
                            borderColor: '#cc0000',
                            borderWidth: 1
                        }}]
                    }},
                    options: {{
                        responsive: true,
                        plugins: {{ legend: {{ display: true }} }}
                    }}
                }});
            }}
        }}
    </script>
</body>
</html>
"""

# --- Main Generation ---

def get_all_data():
    """Gather all newsletter and report data."""
    all_items = []

    # 1. Add Static Data (ensure processing)
    for item in NEWSLETTER_DATA:
        # Create a copy to avoid modifying the original global list repeatedly if called multiple times
        item = item.copy()
        full_text = f"{item['title']} {item['summary']} {item.get('full_body', '')}"
        item['sentiment_score'] = analyze_sentiment(full_text)
        item['entities'] = extract_entities(full_text)
        item['provenance_hash'] = calculate_provenance(full_text)
        item['metrics_json'] = "{}"
        all_items.append(item)

    # 2. Scan Directories (Source)
    for source_dir in SOURCE_DIRS:
        if not os.path.exists(source_dir): continue
        files = glob.glob(os.path.join(source_dir, "*"))
        for filepath in files:
            item = None
            if filepath.endswith(".json"): item = parse_json_file(filepath)
            elif filepath.endswith(".md"): item = parse_markdown_file(filepath)
            elif filepath.endswith(".txt"): item = parse_txt_file(filepath)

            if item: all_items.append(item)

    # 3. Scan Showcase for existing HTML Artifacts (Data Vault, Glitches, Newsletters)
    if os.path.exists(SHOWCASE_DIR):
        files = glob.glob(os.path.join(SHOWCASE_DIR, "*.html"))
        for filepath in files:
            item = parse_html_file(filepath)
            if item: all_items.append(item)

    # Deduplicate based on title/date
    unique_items = {f"{i['date']}_{i['title']}": i for i in all_items}.values()
    sorted_items = sorted(unique_items, key=lambda x: x['date'], reverse=True)
    return sorted_items

def generate_archive():
    print("Starting Enhanced Archive Generation...")

    sorted_items = get_all_data()

    # 3. Generate HTML Reports (Only for sourced items)
    print(f"Generating report pages for sourced items...")
    count = 0
    for item in sorted_items:
        if not item.get("is_sourced", False): continue
        count += 1

        # Build Entity HTML
        entity_html = ""
        for t in item['entities']['tickers']: entity_html += f'<span class="tag ticker">${t}</span>'
        for s in item['entities']['sovereigns']: entity_html += f'<span class="tag">{s}</span>'

        # Build Agent HTML
        agent_html = ""
        if item['entities']['agents']:
            for a in item['entities']['agents']:
                agent_html += f'<div style="font-size:0.75rem; color:#ccc;"><i class="fas fa-robot"></i> {a}</div>'
        else:
            agent_html = '<div style="font-size:0.75rem; color:#666;">System Core</div>'

        out_path = os.path.join(OUTPUT_DIR, item['filename'])

        content = HTML_TEMPLATE.format(
            title=item["title"],
            date=item["date"],
            summary=item["summary"],
            type=item["type"],
            full_body=item.get("full_body", ""),
            sentiment_score=item.get("sentiment_score", 50),
            prov_hash=item.get("provenance_hash", "UNKNOWN"),
            prov_short=item.get("provenance_hash", "UNKNOWN")[:8],
            entity_html=entity_html,
            agent_html=agent_html,
            metrics_json=item.get("metrics_json", "{}")
        )

        with open(out_path, "w", encoding='utf-8') as f:
            f.write(content)
    print(f"Generated {count} sourced pages.")

    # 4. Generate Main Archive Page (Cyber Dashboard)

    # Prepare data for Chart.js
    dates = [i['date'] for i in sorted_items][::-1]
    sentiments = [i['sentiment_score'] for i in sorted_items][::-1]

    # Top Entities
    all_tickers = []
    for i in sorted_items: all_tickers.extend(i['entities']['tickers'])
    ticker_counts = Counter(all_tickers).most_common(10)

    # Grouping
    grouped = {}
    historical = []
    for item in sorted_items:
        year = item["date"].split("-")[0]
        if int(year) < 2021: historical.append(item)
        else:
            if year not in grouped: grouped[year] = []
            grouped[year].append(item)

    tags_html = "".join([f'<span class="tag-cloud-item" onclick="setSearch(\'${t[0]}\')">${t[0]}</span>' for t in ticker_counts])

    archive_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ADAM v23.5 :: MARKET MAYHEM ARCHIVE</title>
    <link rel="stylesheet" href="css/style.css">
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
    <link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Inter:wght@300;400;600;700&display=swap" rel="stylesheet">
    <script src="https://cdn.tailwindcss.com"></script>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <script src="js/nav.js" defer></script>
    <style>
        :root {{
            --primary-color: #00f3ff;
            --accent-color: #cc0000;
            --bg-color: #050b14;
            --panel-bg: rgba(15, 23, 42, 0.6);
            --text-primary: #e0e0e0;
        }}
        body {{ margin: 0; background: var(--bg-color); color: var(--text-primary); font-family: 'Inter', sans-serif; overflow-x: hidden; }}
        .mono {{ font-family: 'JetBrains Mono', monospace; }}

        .cyber-header {{
            height: 60px; display: flex; align-items: center; justify-content: space-between;
            padding: 0 20px; border-bottom: 1px solid #333;
            background: rgba(5, 11, 20, 0.95); position: sticky; top: 0; z-index: 100;
        }}

        .dashboard-grid {{
            display: grid;
            grid-template-columns: 250px 1fr;
            min-height: calc(100vh - 60px);
        }}

        .filters-panel {{
            background: var(--panel-bg);
            border-right: 1px solid #333;
            padding: 20px;
        }}

        .content-panel {{
            padding: 30px;
            overflow-y: auto;
        }}

        .chart-container {{
            background: rgba(0,0,0,0.3);
            border: 1px solid #333;
            border-radius: 8px;
            padding: 20px;
            margin-bottom: 30px;
            height: 300px;
        }}

        .archive-item {{
            border: 1px solid #333; background: rgba(255, 255, 255, 0.03); padding: 20px;
            display: flex; gap: 20px; transition: all 0.2s ease; margin-bottom: 15px;
            border-left: 3px solid #666;
        }}
        .archive-item:hover {{ border-color: var(--primary-color); background: rgba(0, 243, 255, 0.05); transform: translateX(5px); }}

        .type-badge {{ font-size: 0.6rem; padding: 2px 6px; border-radius: 2px; font-weight: bold; font-family: 'JetBrains Mono'; background: #333; color: white; margin-left: 10px; }}

        .filter-group {{ margin-bottom: 20px; }}
        .filter-label {{ font-size: 0.7rem; color: #888; text-transform: uppercase; margin-bottom: 8px; display: block; }}
        .filter-select {{ width: 100%; background: #111; color: #fff; border: 1px solid #444; padding: 8px; font-family: 'JetBrains Mono'; border-radius: 4px; }}

        .tag-cloud-item {{
            display: inline-block; font-size: 0.75rem; color: #aaa; margin: 2px; padding: 2px 6px;
            border: 1px solid #333; border-radius: 4px; cursor: pointer;
        }}
        .tag-cloud-item:hover {{ border-color: var(--primary-color); color: var(--primary-color); }}

        /* Types Colors */
        .type-NEWSLETTER {{ border-left-color: #ffff00; }}
        .type-DEEP_DIVE {{ border-left-color: #cc0000; }}
        .type-MARKET_PULSE {{ border-left-color: #00f3ff; }}
        .type-HISTORICAL {{ border-left-color: #666; }}
    </style>
</head>
<body>
    <header class="cyber-header">
        <div style="display: flex; align-items: center; gap: 20px;">
            <i class="fas fa-archive text-cyan-400"></i>
            <h1 class="mono" style="margin: 0; font-size: 1.2rem; letter-spacing: 1px;">MARKET MAYHEM ARCHIVE</h1>
        </div>
        <div class="mono" style="font-size: 0.8rem; color: #666;">v23.5.0</div>
    </header>

    <div class="dashboard-grid">
        <aside class="filters-panel">
            <div class="filter-group">
                <label class="filter-label">Special Collections</label>
                <div style="display: flex; flex-direction: column; gap: 5px;">
                    <a href="daily_briefings_library.html" class="cyber-btn" style="text-align:center;">DAILY BRIEFINGS</a>
                    <a href="market_pulse_library.html" class="cyber-btn" style="text-align:center;">MARKET PULSE</a>
                    <a href="house_view_library.html" class="cyber-btn" style="text-align:center;">HOUSE VIEW</a>
                    <a href="portfolio_dashboard.html" class="cyber-btn" style="text-align:center; border-color: #33ff00; color: #33ff00;">PORTFOLIO DASHBOARD</a>
                    <a href="data.html" class="cyber-btn" style="text-align:center; border-color: #888; color: #aaa;">DATA VAULT</a>
                    <a href="macro_glitch_monitor.html" class="cyber-btn" style="text-align:center; border-color: #ff00ff; color: #ff00ff;">GLITCH MONITOR</a>
                </div>
            </div>

            <div class="filter-group">
                <label class="filter-label">Search Intelligence</label>
                <input type="text" id="searchInput" placeholder="Keywords..." class="filter-select" onkeyup="applyFilters()">
            </div>

            <div class="filter-group">
                <label class="filter-label">Fiscal Year</label>
                <select id="yearFilter" class="filter-select" onchange="applyFilters()">
                    <option value="ALL">ALL YEARS</option>
                    {"".join([f'<option value="{y}">{y}</option>' for y in sorted(grouped.keys(), reverse=True)])}
                    <option value="HISTORICAL">HISTORICAL</option>
                </select>
            </div>

            <div class="filter-group">
                <label class="filter-label">Report Type</label>
                <select id="typeFilter" class="filter-select" onchange="applyFilters()">
                    <option value="ALL">ALL TYPES</option>
                    <option value="NEWSLETTER">Newsletter</option>
                    <option value="DEEP_DIVE">Deep Dive</option>
                    <option value="COMPANY_REPORT">Company Report</option>
                    <option value="MARKET_PULSE">Market Pulse</option>
                </select>
            </div>

            <div class="filter-group">
                <label class="filter-label">Top Entities</label>
                <div>
                    {tags_html}
                </div>
            </div>
        </aside>

        <main class="content-panel">
            <div class="chart-container">
                <canvas id="sentimentChart"></canvas>
            </div>

            <div id="archiveGrid">
    """

    # Inject Items
    for year in sorted(grouped.keys(), reverse=True):
        for item in grouped[year]:
            archive_html += f"""
            <div class="archive-item type-{item['type']}" data-year="{year}" data-type="{item['type']}" data-title="{item['title'].lower()} {item['summary'].lower()}">
                <div style="flex-grow: 1;">
                    <div style="display:flex; align-items:center; gap:10px; margin-bottom:5px;">
                        <span class="mono" style="font-size:0.7rem; color:#888;">{item['date']}</span>
                        <span class="type-badge">{item['type']}</span>
                        <span class="mono" style="font-size:0.7rem; color:#444;">SENTIMENT: {item['sentiment_score']}</span>
                    </div>
                    <h3 style="margin:0 0 5px 0; font-size:1.1rem;">{item['title']}</h3>
                    <p style="margin:0; font-size:0.85rem; color:#aaa;">{item['summary']}</p>
                </div>
                <a href="{item['filename']}" class="cyber-btn" style="align-self:center;">ACCESS</a>
            </div>
            """

    # Historical
    for item in historical:
         archive_html += f"""
            <div class="archive-item type-HISTORICAL" data-year="HISTORICAL" data-type="HISTORICAL" data-title="{item['title'].lower()}">
                <div style="flex-grow: 1;">
                     <div style="display:flex; align-items:center; gap:10px; margin-bottom:5px;">
                        <span class="mono" style="font-size:0.7rem; color:#888;">{item['date']}</span>
                        <span class="type-badge">HISTORICAL</span>
                    </div>
                    <h3 style="margin:0 0 5px 0; font-size:1.1rem;">{item['title']}</h3>
                    <p style="margin:0; font-size:0.85rem; color:#aaa;">{item['summary']}</p>
                </div>
                 <a href="{item['filename']}" class="cyber-btn" style="align-self:center;">ACCESS</a>
            </div>
            """

    archive_html += """
            </div>
        </main>
    </div>

    <script>
        // Chart Initialization
        const ctx = document.getElementById('sentimentChart').getContext('2d');
        const chartData = {
            labels: """ + json.dumps(dates) + """,
            datasets: [{
                label: 'Market Sentiment Index',
                data: """ + json.dumps(sentiments) + """,
                borderColor: '#00f3ff',
                backgroundColor: 'rgba(0, 243, 255, 0.1)',
                tension: 0.4,
                fill: true
            }]
        };

        new Chart(ctx, {
            type: 'line',
            data: chartData,
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { labels: { color: '#ccc', font: { family: 'JetBrains Mono' } } },
                    title: { display: true, text: 'SENTIMENT TIMELINE', color: '#888', font: { family: 'JetBrains Mono' } }
                },
                scales: {
                    y: { grid: { color: '#333' }, ticks: { color: '#666' } },
                    x: { grid: { color: '#333' }, ticks: { display: false } }
                }
            }
        });

        // Filter Logic
        function applyFilters() {
            const search = document.getElementById('searchInput').value.toLowerCase();
            const year = document.getElementById('yearFilter').value;
            const type = document.getElementById('typeFilter').value;

            const items = document.querySelectorAll('.archive-item');

            items.forEach(item => {
                const itemYear = item.dataset.year;
                const itemType = item.dataset.type;
                const itemText = item.dataset.title;

                let matchSearch = itemText.includes(search);
                let matchYear = year === 'ALL' || itemYear === year;
                let matchType = type === 'ALL' || itemType === type;

                item.style.display = (matchSearch && matchYear && matchType) ? 'flex' : 'none';
            });
        }

        function setSearch(term) {
            document.getElementById('searchInput').value = term;
            applyFilters();
        }
    </script>
</body>
</html>
    """

    with open(ARCHIVE_FILE, "w", encoding='utf-8') as f:
        f.write(archive_html)
    print(f"Archive Updated: {ARCHIVE_FILE}")

if __name__ == "__main__":
    generate_archive()
