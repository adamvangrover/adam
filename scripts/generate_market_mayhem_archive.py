import os
import re
import glob
import json
import yaml
import hashlib
from datetime import datetime
from collections import Counter
from bs4 import BeautifulSoup

# --- Configuration ---
OUTPUT_DIR = "showcase"
SOURCE_DIRS = [
    "core/libraries_and_archives/newsletters",
    "core/libraries_and_archives/reports",
    "core/libraries_and_archives/restored_newsletters",
    "core/libraries_and_archives/generated_content"
]
SHOWCASE_DIR = "showcase"
ARCHIVE_FILE = "showcase/market_mayhem_archive_v24.html" # Updated to v24 to match previous file name usage
TEMPLATE_FILE = "showcase/templates/market_mayhem_report.html"
DATA_FILE = "showcase/data/newsletter_data.json"

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
    """Extracts Tickers ($AAPL), Agents (@AgentName), Sovereigns, and Keywords."""
    tickers = sorted(list(set(re.findall(r'\$([A-Z]{2,5})', text))))
    agents = sorted(list(set(re.findall(r'@([A-Za-z0-9_]+)', text))))

    # Enhanced Sovereign Detection
    sovereigns = []
    if "United States" in text or "US" in text or "Fed" in text: sovereigns.append("US_FED")
    if "China" in text or "PBOC" in text: sovereigns.append("CN_PBOC")
    if "Europe" in text or "ECB" in text: sovereigns.append("EU_ECB")
    if "Japan" in text or "BOJ" in text: sovereigns.append("JP_BOJ")

    # Keywords (Fallback Tags)
    keywords = []
    common_terms = ["Inflation", "Recession", "AI", "Crypto", "Bitcoin", "Ethereum", "Gold", "Oil", "Tech", "Energy", "Banks", "Housing", "Retail", "Supply Chain", "Labor", "Yields", "Bonds", "Equities", "Volatility", "Growth", "Value", "Quality", "Momentum", "Dividend", "IPO", "SPAC", "NFT", "DeFi", "Web3", "Metaverse", "Cloud", "SaaS", "Cybersecurity", "Semiconductors", "EV", "Green Energy", "ESG"]

    for term in common_terms:
        if term.lower() in text.lower():
            keywords.append(term)

    return {
        "tickers": tickers,
        "agents": agents,
        "sovereigns": sorted(list(set(sovereigns))),
        "keywords": sorted(list(set(keywords)))[:5]  # Limit to 5 keywords
    }

def calculate_conviction(text):
    """Calculates conviction score based on modal verbs."""
    if not text: return 50
    text_lower = text.lower()

    high_conviction = ["must", "will", "always", "never", "definitely", "crucial", "essential", "undeniable", "guaranteed"]
    low_conviction = ["might", "could", "maybe", "possibly", "perhaps", "unlikely", "potential", "uncertain"]

    words = re.findall(r'\w+', text_lower)
    high_count = sum(1 for w in words if w in high_conviction)
    low_count = sum(1 for w in words if w in low_conviction)

    total = high_count + low_count
    if total == 0: return 50

    # Scale: 0-100. More high conviction = higher score.
    score = (high_count / total) * 100
    return int(score)

def calculate_semantic_score(text):
    """Calculates semantic density (unique words / total words)."""
    if not text: return 0
    words = re.findall(r'\w+', text.lower())
    if not words: return 0

    unique_words = set(words)
    density = (len(unique_words) / len(words)) * 100
    return int(density)

def calculate_probability(text):
    """Calculates a probability score for the scenario/event described."""
    conviction = calculate_conviction(text)

    # Check for keywords
    text_lower = text.lower()
    if "high probability" in text_lower or "certainty" in text_lower or "inevitable" in text_lower:
        return min(conviction + 20, 99)
    if "low probability" in text_lower or "unlikely" in text_lower:
        return max(conviction - 20, 1)

    return conviction

def generate_critique(item):
    """Generates a structured critique if missing."""
    if item.get("critique") and item["critique"] != "No automated critique available.":
        return item["critique"]

    # Mock Critique Generation
    agents = ["Overseer", "Risk_Engine", "Sovereign_AI", "Market_Maker"]
    # Pick agent based on title hash to be deterministic
    h = int(hashlib.md5(item["title"].encode()).hexdigest(), 16)
    agent = agents[h % len(agents)]

    verdicts = ["VALIDATED", "REVIEW_REQUIRED", "HIGH_CONFIDENCE", "SPECULATIVE"]
    verdict = verdicts[h % len(verdicts)]

    critique = f"Agent {agent} reviewed this intelligence. Verdict: {verdict}. "
    critique += f"Sentiment alignment: {item.get('sentiment_score')}/100. "
    critique += "Cross-reference with knowledge graph completed."

    return critique

def calculate_provenance(text):
    """Generates a SHA-256 hash of the content."""
    return hashlib.sha256(text.encode('utf-8')).hexdigest()

def determine_tier(type_, quality):
    """Determines the visual tier based on type and quality."""
    # Force low tier for specific types regardless of quality score (Feed/Chat style)
    if type_ in ["MARKET_PULSE", "DAILY_BRIEFING", "CYBER_GLITCH", "TECH_WATCH", "AGENT_NOTE"]:
        return "tier-low"

    if type_ in ["DEEP_DIVE", "SPECIAL_EDITION", "STRATEGY", "ANNUAL_STRATEGY", "HISTORICAL", "COUNTERFACTUAL"]:
        return "tier-high"

    # Fallback to quality for standard newsletters
    if quality >= 90:
        return "tier-high"

    if type_ in ["NEWSLETTER", "WEEKLY_RECAP", "INDUSTRY_REPORT", "COMPANY_REPORT"]:
        return "tier-medium"

    return "tier-medium"

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
    date_str = str(date_str).strip()

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
            return datetime.strptime(date_str, fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue

    match_us = re.search(r'^(\d{2})[-_]?(\d{2})[-_]?(\d{4})$', date_str)
    if match_us:
        return f"{match_us.group(3)}-{match_us.group(1)}-{match_us.group(2)}"

    match_iso = re.search(r'^(\d{4})[-_]?(\d{2})[-_]?(\d{2})$', date_str)
    if match_iso:
         return f"{match_iso.group(1)}-{match_iso.group(2)}-{match_iso.group(3)}"

    match = re.search(r'(\d{4})[-_]?(\d{2})[-_]?(\d{2})', date_str)
    if match:
        return f"{match.group(1)}-{match.group(2)}-{match.group(3)}"

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

        title = data.get('title') or data.get('topic') or data.get('report_title')
        if not title:
            company = data.get('company') or data.get('company_name')
            if company:
                title = f"{company} Report"
            else:
                title = 'Untitled Report'

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
             date_match_2 = re.search(r'((?:19|20)\d{2})[-_]?(\d{2})[-_]?(\d{2})', filename)
             if date_match_2:
                 date_str = f"{date_match_2.group(1)}-{date_match_2.group(2)}-{date_match_2.group(3)}"
             else:
                 match_us = re.search(r'(\d{2})(\d{2})(\d{4})', filename)
                 if match_us:
                     date_str = f"{match_us.group(3)}-{match_us.group(1)}-{match_us.group(2)}"

        body_html = ""
        summary = data.get('summary', "")

        metrics_json = "{}"
        if "financial_performance" in data and "metrics" in data["financial_performance"]:
             metrics_json = json.dumps(data["financial_performance"]["metrics"])
        elif "metrics" in data:
             metrics_json = json.dumps(data["metrics"])

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
                         metrics_data = section['metrics']
                         if isinstance(metrics_data, dict):
                             body_html += format_dict_as_html(metrics_data, 3)
                         elif isinstance(metrics_data, list):
                             body_html += "<ul>"
                             for m in metrics_data:
                                 body_html += f"<li>{m}</li>"
                             body_html += "</ul>"

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

        elif summary:
             body_html += f"<p>{summary}</p>"
             for k, v in data.items():
                 if k not in ['title', 'date', 'summary', 'file_name', 'company', 'analyst'] and isinstance(v, str):
                     body_html += f"<h3>{k.replace('_', ' ').title()}</h3><p>{v}</p>"

        if not summary: summary = f"Market Report from {date_str}"

        type_ = "NEWSLETTER"
        lower_title = title.lower()
        if "deep dive" in lower_title: type_ = "DEEP_DIVE"
        elif "industry" in lower_title: type_ = "INDUSTRY_REPORT"
        elif "daily briefing" in lower_title: type_ = "DAILY_BRIEFING"
        elif "market pulse" in lower_title: type_ = "MARKET_PULSE"
        elif "tech watch" in lower_title: type_ = "TECH_WATCH"
        elif "company" in lower_title or "report" in lower_title: type_ = "COMPANY_REPORT"
        elif "thematic" in lower_title: type_ = "THEMATIC_REPORT"
        elif "outlook" in lower_title or "review" in lower_title: type_ = "MARKET_OUTLOOK"

        if "file_name" in data:
            out_filename = data["file_name"].replace('.json', '.html')
        else:
            out_filename = filename.replace('.json', '.html').replace('.md', '.html')

        full_text = f"{title} {summary} {body_html}"
        sentiment = analyze_sentiment(full_text)
        entities = extract_entities(full_text)
        prov_hash = calculate_provenance(full_text)

        conviction = calculate_conviction(full_text)
        semantic_score = calculate_semantic_score(full_text)
        probability = calculate_probability(full_text)

        item = {
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
            "metrics_json": metrics_json,
            "source_priority": 2,
            "conviction": conviction,
            "semantic_score": semantic_score,
            "probability": probability
        }

        item["critique"] = generate_critique(item)
        return item
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

        sentiment = analyze_sentiment(content)
        entities = extract_entities(content)
        prov_hash = calculate_provenance(content)

        conviction = calculate_conviction(content)
        semantic_score = calculate_semantic_score(content)
        probability = calculate_probability(content)

        item = {
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
            "metrics_json": "{}",
            "source_priority": 2,
            "conviction": conviction,
            "semantic_score": semantic_score,
            "probability": probability
        }
        item["critique"] = generate_critique(item)
        return item
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
             date_in_title = re.search(r'(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}', title, re.IGNORECASE)
             if date_in_title:
                 raw_date = date_in_title.group(0)
             else:
                 date_match = re.search(r'((?:19|20)\d{2})[-_]?(\d{2})[-_]?(\d{2})', filename)
                 if date_match:
                     raw_date = f"{date_match.group(1)}-{date_match.group(2)}-{date_match.group(3)}"
                 else:
                     match_us = re.search(r'(\d{2})(\d{2})(\d{4})', filename)
                     if match_us:
                         raw_date = f"{match_us.group(3)}-{match_us.group(1)}-{match_us.group(2)}"
                     else:
                         raw_date = "2025-01-01"
        else:
            raw_date = date_match.group(1)

        date = parse_date(raw_date)

        if "Deep_Dive" in filename: type_ = "DEEP_DIVE"
        elif "Pulse" in filename or "Pulse" in title: type_ = "MARKET_PULSE"
        elif "Daily" in filename or "Briefing" in title: type_ = "DAILY_BRIEFING"
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

        conviction_match = re.search(r'\*\*Conviction:\*\* .*?(\d+)/100', content)
        conviction_score = int(conviction_match.group(1)) if conviction_match else 50

        critique_match = re.search(r'\*\*Critique:\*\* (.*?)\n', content)
        critique_text = critique_match.group(1) if critique_match else "No automated critique available."

        quality_match = re.search(r'\*\*Quality Score:\*\* (\d+)/100', content)
        quality_score = int(quality_match.group(1)) if quality_match else 100

        if title_match:
             content = content.replace(title_match.group(0), "", 1).strip()

        body_html = simple_md_to_html(content)

        sentiment = analyze_sentiment(content)
        entities = extract_entities(content)
        prov_hash = calculate_provenance(content)

        semantic_score = calculate_semantic_score(content)
        probability = calculate_probability(content)

        if conviction_score == 50:
             conviction_score = calculate_conviction(content)

        item = {
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
            "metrics_json": "{}",
            "source_priority": 2,
            "conviction": conviction_score,
            "critique": critique_text,
            "quality": quality_score,
            "semantic_score": semantic_score,
            "probability": probability
        }

        if critique_text == "No automated critique available.":
             item["critique"] = generate_critique(item)

        return item

    except Exception as e:
        print(f"Error parsing MD {filepath}: {e}")
        return None

def parse_html_file(filepath):
    """Parses existing HTML files (legacy or generated) to extract metadata."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
            f.seek(0)
            soup = BeautifulSoup(f, 'html.parser')

        filename = os.path.basename(filepath)

        if filename in ["index.html", "market_mayhem_archive.html", "market_mayhem_archive_v24.html", "daily_briefings_library.html",
                        "market_pulse_library.html", "house_view_library.html", "portfolio_dashboard.html",
                        "data.html", "reports.html", "agents.html", "chat.html", "market_mayhem_rebuild.html", "market_mayhem_conviction.html"]:
            return None

        container = soup.find('div', class_='newsletter-container')
        if container:
            title_tag = soup.find('h1', class_='title')
            title = title_tag.get_text(strip=True) if title_tag else "Untitled Newsletter"

            date_str = "2025-01-01"
            meta_div = soup.find('div', class_='meta-header')
            if meta_div:
                for div in meta_div.find_all('div'):
                    text = div.get_text(strip=True)
                    if text.startswith("Date:"):
                        date_str = parse_date(text.replace("Date:", "").strip())
                        break
            
            summary = "Market analysis."
            summary_p = soup.find('p', style=lambda s: s and 'italic' in s)
            if summary_p:
                summary = summary_p.get_text(strip=True).replace("Executive Summary:", "").strip()

            type_badge = soup.find('div', class_='cyber-badge-overlay')
            type_ = "NEWSLETTER"
            if type_badge:
                badge_text = type_badge.get_text(strip=True)
                if "WEEKLY" in badge_text: type_ = "WEEKLY_RECAP"
                elif "MONTHLY" in badge_text: type_ = "NEWSLETTER"
                elif "FLASH" in badge_text: type_ = "MARKET_PULSE"

            for match in container.find_all(['h1', 'div'], class_=['title', 'meta-header', 'cyber-badge-overlay', 'back-link']):
                match.decompose()
            disclaimers = container.find_all('div', style=lambda s: s and 'border-top' in s)
            for disc in disclaimers: disc.decompose()
            
            body_html = "".join([str(x) for x in container.contents])
            
            full_text = soup.get_text()
            sentiment = analyze_sentiment(full_text)
            entities = extract_entities(full_text)
            prov_hash = calculate_provenance(full_text)

            conviction = calculate_conviction(full_text)
            semantic_score = calculate_semantic_score(full_text)
            probability = calculate_probability(full_text)

            item = {
                "title": title,
                "date": date_str,
                "summary": summary,
                "type": type_,
                "full_body": body_html,
                "sentiment_score": sentiment,
                "entities": entities,
                "provenance_hash": prov_hash,
                "filename": filename,
                "is_sourced": True,
                "metrics_json": "{}",
                "source_priority": 1,
                "conviction": conviction,
                "semantic_score": semantic_score,
                "probability": probability
            }
            item["critique"] = generate_critique(item)
            return item

        title_match = re.search(r'<title>(?:ADAM v23\.5 :: )?(.*?)</title>', content, re.IGNORECASE)
        title = title_match.group(1).strip() if title_match else filename.replace('.html', '').replace('_', ' ')

        date = "2025-01-01" 
        
        mm_match = re.search(r'MM(\d{2})(\d{2})(\d{4})', filename)
        if mm_match:
            date = f"{mm_match.group(3)}-{mm_match.group(1)}-{mm_match.group(2)}"
        else:
            date_match = re.search(r'(\d{4})[-_]?(\d{2})[-_]?(\d{2})', filename)
            if date_match:
                date = f"{date_match.group(1)}-{date_match.group(2)}-{date_match.group(3)}"
            else:
                content_date = re.search(r'(\w+ \d{1,2}, \d{4})', content)
                if content_date:
                    date = parse_date(content_date.group(1))

        summary = "Report content."
        meta_desc = soup.find('meta', attrs={'name': 'description'})
        if meta_desc and meta_desc.get('content'):
            summary = meta_desc['content']

        type_ = "NEWSLETTER"
        lower_title = title.lower()
        if "deep dive" in lower_title or "deep_dive" in filename: type_ = "DEEP_DIVE"
        elif "industry" in lower_title: type_ = "INDUSTRY_REPORT"
        elif "company" in lower_title: type_ = "COMPANY_REPORT"
        elif "pulse" in lower_title: type_ = "MARKET_PULSE"
        elif "glitch" in lower_title: type_ = "CYBER_GLITCH"
        elif "outlook" in lower_title or "strategy" in lower_title: type_ = "STRATEGY"
        elif "snc" in lower_title: type_ = "GUIDE"
        elif "counterfactual" in lower_title: type_ = "COUNTERFACTUAL"
        elif "audit" in lower_title: type_ = "SYSTEM_AUDIT"

        sentiment = analyze_sentiment(content)
        entities = extract_entities(content)
        prov_hash = calculate_provenance(content)

        conviction = calculate_conviction(content)
        semantic_score = calculate_semantic_score(content)
        probability = calculate_probability(content)

        item = {
            "title": title,
            "date": date,
            "summary": summary,
            "type": type_,
            "full_body": "",
            "sentiment_score": sentiment,
            "entities": entities,
            "provenance_hash": prov_hash,
            "filename": filename,
            "is_sourced": False,
            "metrics_json": "{}",
            "conviction": conviction,
            "semantic_score": semantic_score,
            "probability": probability
        }
        item["critique"] = generate_critique(item)
        return item

    except Exception as e:
        print(f"Error parsing HTML {filepath}: {e}")
        return None

# --- Main Generation ---

def load_template():
    """Loads the HTML template from file."""
    if not os.path.exists(TEMPLATE_FILE):
        raise FileNotFoundError(f"Template file not found: {TEMPLATE_FILE}")
    with open(TEMPLATE_FILE, 'r', encoding='utf-8') as f:
        return f.read()

def load_static_data():
    """Loads static newsletter data from JSON."""
    if not os.path.exists(DATA_FILE):
        return []
    with open(DATA_FILE, 'r', encoding='utf-8') as f:
        return json.load(f)

def get_all_data():
    """Gather all newsletter and report data."""
    all_items = []

    # 1. Add Static Data
    static_data = load_static_data()
    for item in static_data:
        item = item.copy()
        full_text = f"{item['title']} {item['summary']} {item.get('full_body', '')}"
        item['sentiment_score'] = analyze_sentiment(full_text)
        item['entities'] = extract_entities(full_text)
        item['provenance_hash'] = calculate_provenance(full_text)
        item['metrics_json'] = "{}"

        item['conviction'] = calculate_conviction(full_text)
        item['semantic_score'] = calculate_semantic_score(full_text)
        item['probability'] = calculate_probability(full_text)
        item['critique'] = generate_critique(item)

        all_items.append(item)

    # 2. Scan Directories
    for source_dir in SOURCE_DIRS:
        if not os.path.exists(source_dir): continue
        files = glob.glob(os.path.join(source_dir, "*"))
        for filepath in files:
            item = None
            if filepath.endswith(".json"): item = parse_json_file(filepath)
            elif filepath.endswith(".md"): item = parse_markdown_file(filepath)
            elif filepath.endswith(".txt"): item = parse_txt_file(filepath)
            elif filepath.endswith(".html"): item = parse_html_file(filepath)

            if item: all_items.append(item)

    # 3. Scan Showcase
    if os.path.exists(SHOWCASE_DIR):
        files = glob.glob(os.path.join(SHOWCASE_DIR, "*.html"))
        for filepath in files:
            item = parse_html_file(filepath)
            if item: all_items.append(item)

    # Deduplicate
    grouped_by_date = {}
    for item in all_items:
        date = item['date']
        if date not in grouped_by_date:
            grouped_by_date[date] = []
        grouped_by_date[date].append(item)

    final_items = []
    for date, items in grouped_by_date.items():
        for i in items:
            if 'source_priority' not in i: i['source_priority'] = 0 if not i.get('is_sourced', False) else 1

        items.sort(key=lambda x: x.get('source_priority', 0), reverse=True)

        unique_types = {}
        for item in items:
            t = item['type']
            key = f"{t}_{item['title']}"
            
            if key not in unique_types:
                unique_types[key] = item
            else:
                if item.get('source_priority', 0) > unique_types[key].get('source_priority', 0):
                    unique_types[key] = item

        for item in unique_types.values():
            final_items.append(item)

    sorted_items = sorted(final_items, key=lambda x: x['date'], reverse=True)
    return sorted_items

def generate_archive():
    print("Starting Enhanced Archive Generation...")

    try:
        HTML_TEMPLATE = load_template()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    sorted_items = get_all_data()

    # Dump JSON Index
    json_path = os.path.join("showcase/data", "market_mayhem_index.json")
    os.makedirs(os.path.dirname(json_path), exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(sorted_items, f, default=str, indent=2)
    print(f"Generated Index JSON: {json_path}")

    # Generate HTML Reports
    print(f"Generating report pages for sourced items...")
    count = 0
    for item in sorted_items:
        if not item.get("is_sourced", False) or not item.get("full_body"): 
            continue
        
        if item.get("source_priority", 0) < 2:
            continue

        count += 1

        # Entity HTML
        entity_html = ""
        for t in item['entities']['tickers']: entity_html += f'<a href="market_mayhem_archive_v24.html?search={t}" class="tag ticker" style="text-decoration:none;">${t}</a>'
        for s in item['entities']['sovereigns']: entity_html += f'<a href="market_mayhem_archive_v24.html?search={s}" class="tag" style="text-decoration:none;">{s}</a>'
        for k in item['entities'].get('keywords', []): entity_html += f'<a href="market_mayhem_archive_v24.html?search={k}" class="tag" style="text-decoration:none; border-color:#666;">{k}</a>'

        if not entity_html:
            entity_html = f'<span class="tag" style="border-color:#444;">{item["type"]}</span>'

        # Agent HTML (with Avatars)
        agent_html = ""
        if item['entities']['agents']:
            for a in item['entities']['agents']:
                avatar_class = "core"
                if "Market" in a: avatar_class = "market"
                if "Risk" in a: avatar_class = "risk"
                agent_html += f'<div style="display:flex; align-items:center;"><div class="agent-avatar {avatar_class}">{a[:2]}</div><div style="font-size:0.75rem; color:#ccc;">{a}</div></div>'
        else:
            agent_html = '<div style="font-size:0.75rem; color:#666;">System Core</div>'

        out_path = os.path.join(OUTPUT_DIR, item['filename'])

        conviction = item.get("conviction", 50)
        conviction_color = "#00ff00" if conviction > 70 else ("#ffff00" if conviction > 40 else "#ff0000")

        # Determine Tier
        tier_class = determine_tier(item['type'], item.get('quality', 100))

        # UI Toggles based on Tier
        toolbar_display = "flex" if tier_class == "tier-high" else "none"
        header_ui_display = "flex" if tier_class == "tier-high" else "none"
        metadata_panel_display = "grid" if tier_class == "tier-high" else "none"
        nav_display = "none" if tier_class == "tier-high" else "flex" # High tier uses toolbar for nav

        # Extract reviewer agent from critique for metadata panel
        critique = item.get("critique", "Agent System reviewed this.")
        reviewer_match = re.search(r'Agent (\w+)', critique)
        reviewer_agent = reviewer_match.group(1) if reviewer_match else "SYSTEM"

        # Serialize entire item as raw source
        raw_source = json.dumps(item, default=str, indent=2)

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
            metrics_json=item.get("metrics_json", "{}"),
            conviction=conviction,
            conviction_color=conviction_color,
            critique=critique,
            quality=item.get("quality", 100),
            semantic_score=item.get("semantic_score", 0),
            probability=item.get("probability", 50),
            tier_class=tier_class,
            toolbar_display=toolbar_display,
            header_ui_display=header_ui_display,
            metadata_panel_display=metadata_panel_display,
            nav_display=nav_display,
            reviewer_agent=reviewer_agent,
            raw_source=raw_source
        )

        with open(out_path, "w", encoding='utf-8') as f:
            f.write(content)
    print(f"Generated {count} sourced pages.")

    # Generate Main Archive Page
    chart_items = [i for i in sorted_items if i['date'] and i['date'] != "2025-01-01"]
    chart_items.sort(key=lambda x: x['date'])
    
    dates = [i['date'] for i in chart_items]
    sentiments = [i['sentiment_score'] for i in chart_items]
    convictions = [i.get('conviction', 50) for i in chart_items]

    window_size = 5
    moving_averages = []
    for i in range(len(sentiments)):
        if i < window_size - 1:
             partial = sentiments[:i+1]
             moving_averages.append(sum(partial)/len(partial))
        else:
             window = sentiments[i-window_size+1:i+1]
             moving_averages.append(sum(window)/window_size)

    if not sentiments:
        market_regime = "UNKNOWN"
        regime_color = "#888"
    else:
        latest_avg = moving_averages[-1]
        if latest_avg > 60:
            market_regime = "BULLISH AGENTIC"
            regime_color = "#00ff00"
        elif latest_avg < 40:
            market_regime = "BEARISH CRASH"
            regime_color = "#ff0000"
        else:
            market_regime = "VOLATILE TRANSITION"
            regime_color = "#ffff00"

    active_agents_count = len(set([a for i in sorted_items for a in i['entities']['agents']]))

    all_tickers = []
    for i in sorted_items: all_tickers.extend(i['entities']['tickers'])
    ticker_counts = Counter(all_tickers).most_common(10)

    grouped = {}
    historical = []
    for item in sorted_items:
        year = item["date"].split("-")[0]
        if int(year) < 2021: historical.append(item)
        else:
            if year not in grouped: grouped[year] = []
            grouped[year].append(item)

    tags_html = "".join([f'<span class="tag-cloud-item" onclick="setSearch(\'{t[0]}\')">{t[0]}</span>' for t in ticker_counts])

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

        /* System Dashboard */
        .system-dashboard {{
            display: grid; grid-template-columns: repeat(3, 1fr); gap: 20px;
            margin-bottom: 30px; background: rgba(0,0,0,0.4); border: 1px solid #333; padding: 20px; border-radius: 8px;
        }}
        .metric {{ display: flex; flex-direction: column; align-items: center; border-right: 1px solid #333; }}
        .metric:last-child {{ border-right: none; }}
        .metric .label {{ font-size: 0.7rem; color: #888; margin-bottom: 5px; letter-spacing: 1px; }}
        .metric .value {{ font-size: 1.2rem; font-weight: bold; color: #fff; font-family: 'JetBrains Mono'; }}

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
        <div class="mono" style="font-size: 0.8rem; color: #666;">v24.0.2</div>
    </header>

    <div class="dashboard-grid">
        <aside class="filters-panel">
            <div class="filter-group">
                <label class="filter-label">Special Collections</label>
                <div style="display: flex; flex-direction: column; gap: 5px;">
                    <a href="market_mayhem_rebuild.html" class="cyber-btn" style="text-align:center; border-color: #00f3ff; color: #00f3ff;">SYSTEM 2 HUB</a>
                    <a href="market_mayhem_conviction.html" class="cyber-btn" style="text-align:center;">CONVICTION PAPER</a>
                    <a href="portfolio_dashboard.html" class="cyber-btn" style="text-align:center; border-color: #33ff00; color: #33ff00;">PORTFOLIO DASHBOARD</a>
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
            <div class="system-dashboard">
                <div class="metric">
                    <span class="label">MARKET REGIME</span>
                    <span class="value" style="color: {regime_color}">{market_regime}</span>
                </div>
                <div class="metric">
                    <span class="label">SYSTEM HEALTH</span>
                    <span class="value">OPTIMAL</span>
                </div>
                <div class="metric">
                    <span class="label">ACTIVE AGENTS</span>
                    <span class="value">{active_agents_count}</span>
                </div>
            </div>

            <div class="chart-container">
                <canvas id="sentimentChart"></canvas>
            </div>

            <div id="archiveGrid">
    """

    for year in sorted(grouped.keys(), reverse=True):
        for item in grouped[year]:
            safe_title = item['title'].replace('"', "'").lower()
            safe_summary = item['summary'].replace('"', "'").lower()
            
            archive_html += f"""
            <div class="archive-item type-{item['type']}" data-year="{year}" data-type="{item['type']}" data-title="{safe_title} {safe_summary}">
                <div style="flex-grow: 1;">
                    <div style="display:flex; align-items:center; gap:10px; margin-bottom:5px;">
                        <span class="mono" style="font-size:0.7rem; color:#888;">{item['date']}</span>
                        <span class="type-badge">{item['type']}</span>
                        <span class="mono" style="font-size:0.7rem; color:#444;">SENT: {item['sentiment_score']} | CONV: {item.get('conviction', 50)} | SEM: {item.get('semantic_score', 0)}</span>
                    </div>
                    <h3 style="margin:0 0 5px 0; font-size:1.1rem;">{item['title']}</h3>
                    <p style="margin:0; font-size:0.85rem; color:#aaa;">{item['summary']}</p>
                </div>
                <a href="{item['filename']}" class="cyber-btn" style="align-self:center;">ACCESS</a>
            </div>
            """

    for item in historical:
         safe_title = item['title'].replace('"', "'").lower()
         safe_summary = item['summary'].replace('"', "'").lower()
         archive_html += f"""
            <div class="archive-item type-HISTORICAL" data-year="HISTORICAL" data-type="HISTORICAL" data-title="{safe_title} {safe_summary}">
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
        const ctx = document.getElementById('sentimentChart').getContext('2d');
        const chartData = {
            labels: """ + json.dumps(dates) + """,
            datasets: [
                {
                    label: 'Sentiment Score',
                    data: """ + json.dumps(sentiments) + """,
                    borderColor: '#00f3ff',
                    backgroundColor: 'rgba(0, 243, 255, 0.1)',
                    tension: 0.4,
                    fill: true
                },
                {
                    label: 'Conviction Score',
                    data: """ + json.dumps(convictions) + """,
                    borderColor: '#ff00ff',
                    backgroundColor: 'rgba(255, 0, 255, 0.1)',
                    borderDash: [5, 5],
                    tension: 0.4,
                    hidden: false
                },
                {
                    label: 'Moving Average (5p)',
                    data: """ + json.dumps(moving_averages) + """,
                    borderColor: '#ffffff',
                    borderWidth: 2,
                    pointRadius: 0,
                    tension: 0.4,
                    fill: false
                }
            ]
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

        // Initialize from URL params
        window.addEventListener('DOMContentLoaded', () => {
            const params = new URLSearchParams(window.location.search);
            const search = params.get('search');
            if (search) {
                setSearch(search);
            }
        });
    </script>
</body>
</html>
    """

    with open(ARCHIVE_FILE, "w", encoding='utf-8') as f:
        f.write(archive_html)
    print(f"Archive Updated: {ARCHIVE_FILE}")

if __name__ == "__main__":
    generate_archive()
