import os
import re
import glob
from datetime import datetime

# --- Configuration ---
OUTPUT_DIR = "showcase"
ARCHIVE_FILE = "showcase/market_mayhem_archive.html"
TEMPLATE_FILE = "showcase/newsletter_market_mayhem.html"  # We'll use this as a base, but I'll inline the template for simplicity

# Regex patterns for extraction
TITLE_RE = re.compile(r'<h1 class="title">(.*?)</h1>', re.IGNORECASE)
DATE_RE = re.compile(r'Date:</strong>\s*([\d-]+)', re.IGNORECASE)
SUMMARY_RE = re.compile(r'Executive Summary:</strong>\s*(.*?)(?:</p>|<br>)', re.IGNORECASE | re.DOTALL)
TYPE_RE = re.compile(r'TYPE:\s*(.*?)\s*//', re.IGNORECASE)

# --- Original Data (Restored for fallback generation) ---
NEWSLETTER_DATA = [
    # 2025 Monthly
    {
        "date": "2025-09-14",
        "title": "THE LIQUIDITY TRAP",
        "summary": "Fed pauses cuts as core services inflation sticks. Small caps get crushed. The case for 'Quality' factor investing.",
        "type": "MONTHLY",
        "filename": "newsletter_market_mayhem_sep_2025.html",
        "content_highlights": [
            "Fed holds rates steady at 5.25%.",
            "Russell 2000 drops 8% in two weeks.",
            "Long Mega-Cap Tech (Quality) vs Short Unprofitable Tech."
        ]
    },
    {
        "date": "2025-08-14",
        "title": "SUMMER DOLDRUMS & AI FATIGUE",
        "summary": "Volume dries up. AI Capex questions emerge. Nvidia earnings preview: Is the bar too high?",
        "type": "MONTHLY",
        "filename": "newsletter_market_mayhem_aug_2025.html",
        "content_highlights": [
            "VIX hits 12 handle (complacency).",
            "AI Capex ROI concerns surfacing in earnings calls.",
            "Defensive rotation into Healthcare."
        ]
    },
    {
        "date": "2025-07-14",
        "title": "CRYPTO REGULATION SHOCK",
        "summary": "SEC 2.0 launches 'Operation Chokepoint'. DeFi protocols under siege. Bitcoin flight to safety narrative tested.",
        "type": "MONTHLY",
        "filename": "newsletter_market_mayhem_jul_2025.html",
        "content_highlights": [
            "Major exchange enforcement action.",
            "Bitcoin dominance rises as alts collapse.",
            "Regulatory clarity is years away."
        ]
    },
    {
        "date": "2025-06-14",
        "title": "THE DOLLAR WRECKING BALL",
        "summary": "DXY breaks 108. Emerging Market currencies collapse. Carry trade unwind begins.",
        "type": "MONTHLY",
        "filename": "newsletter_market_mayhem_jun_2025.html",
        "content_highlights": [
            "JPY/USD hits 160.",
            "EM Debt crisis looming.",
            "Long USD/Short JPY trade crowded but working."
        ]
    },
     {
        "date": "2025-05-14",
        "title": "COMMERCIAL REAL ESTATE: THE RECKONING",
        "summary": "Office vacancy hits 25%. Regional banks take haircuts. The 'Extend and Pretend' game ends.",
        "type": "MONTHLY",
        "filename": "newsletter_market_mayhem_may_2025.html",
        "content_highlights": [
            "San Francisco office tower sells for $100/sqft.",
            "KRE ETF puts are the hedge of choice.",
            "Private Credit steps in where banks fear to tread."
        ]
    },
    {
        "date": "2025-04-14",
        "title": "Q1 EARNINGS: PROFIT MARGIN SQUEEZE",
        "summary": "Wage inflation eats into margins. Guidance cut across Consumer Discretionary. The Recession is not cancelled, just delayed.",
        "type": "MONTHLY",
        "filename": "newsletter_market_mayhem_apr_2025.html",
        "content_highlights": [
            "Wage growth sticky at 4.5%.",
            "Retailers guiding down.",
            "Stagflation risks rising."
        ]
    },
    {
        "date": "2025-03-14",
        "title": "BANKING CRISIS 2.0?",
        "summary": "New York Community Bank (NYCB) echoes. Is it idiosyncratic or systemic? Fed creates new backstop facility.",
        "type": "MONTHLY",
        "filename": "newsletter_market_mayhem_mar_2025.html",
        "content_highlights": [
            "NYCB stock down 40%.",
            "Deposit flight concerns muted compared to '23.",
            "Buy the dip in JPM/BAC."
        ]
    },
    {
        "date": "2025-02-14",
        "title": "THE AI BUBBLE BURST?",
        "summary": "Semis correct 20%. Is the hype cycle over? Distinguishing infrastructure builders from application vaporware.",
        "type": "MONTHLY",
        "filename": "newsletter_market_mayhem_feb_2025.html",
        "content_highlights": [
            "SOXX correction.",
            "Valuation reset creates entry points.",
            "Focus on 'Pick and Shovel' plays."
        ]
    },
    {
        "date": "2025-01-14",
        "title": "2025 OUTLOOK: THE YEAR OF VOLATILITY",
        "summary": "Regime shift. The era of 'Great Moderation' is officially dead. Prepare for higher-for-longer rates and geopolitical chaos.",
        "type": "MONTHLY",
        "filename": "newsletter_market_mayhem_jan_2025.html",
        "content_highlights": [
            "Base case: No recession but slow growth.",
            "Geopolitics: The #1 risk factor.",
            "Portfolio Strategy: 60/40 is dead. Long Alts."
        ]
    },

    # 2024 Retrospective / Highlights
    {
        "date": "2024-12-14",
        "title": "2024 POST-MORTEM: THE SOFT LANDING MIRACLE",
        "summary": "How the Fed threaded the needle. Inflation down, unemployment low. Can it last?",
        "type": "YEARLY_REVIEW",
        "filename": "newsletter_market_mayhem_dec_2024.html",
        "content_highlights": [
            "S&P 500 +24% YTD.",
            "Mag 7 dominance.",
            "The 'Immaculate Disinflation'."
        ]
    },

    # Flash Bulletins (Weekly/Daily)
    {
        "date": "2025-11-01",
        "title": "FLASH: FED EMERGENCY MEETING RUMORS",
        "summary": "Bond market dislocation triggers rumors of emergency liquidity injection. VIX spikes to 30.",
        "type": "FLASH",
        "filename": "newsletter_flash_nov_01_2025.html",
        "content_highlights": [
            "Treasury liquidity drying up.",
            "Fed whisper numbers suggest QT taper.",
            "Stay cash heavy."
        ]
    },
    {
        "date": "2025-09-21",
        "title": "WEEKLY INTEL: OIL BREAKOUT",
        "summary": "Technical breakout in crude. $90 is the next stop. Energy sector upgrading to Overweight.",
        "type": "WEEKLY",
        "filename": "newsletter_weekly_sep_21_2025.html",
        "content_highlights": [
            "WTI crosses $85.",
            "Saudi production cuts extended.",
            "XLE calls."
        ]
    },

    # Historical Archives (Parsed from existing files)
    {
        "date": "2020-03-20",
        "title": "MARKET MAYHEM: THE GREAT SHUT-IN",
        "summary": "'Lockdown'. The global economy has come to a screeching halt. With '15 Days to Slow the Spread' in effect, markets are pricing in a depression-level GDP contraction.",
        "type": "HISTORICAL ARCHIVE",
        "filename": "newsletter_market_mayhem_mar_2020.html",
        "content_highlights": [
            "S&P 500 drops 35% in a month.",
            "Fed cuts rates to zero.",
            "Oil futures turn negative."
        ]
    },
    {
        "date": "2008-09-19",
        "title": "MARKET MAYHEM: THE LEHMAN MOMENT",
        "summary": "'Existential Panic'. There are decades where nothing happens; and there are weeks where decades happen. This was one of those weeks. A 158-year-old bank vanished, the world's largest insurer was nationalized, and the money market broke the buck.",
        "type": "HISTORICAL ARCHIVE",
        "filename": "newsletter_market_mayhem_sep_2008.html",
        "content_highlights": [
            "Lehman Brothers files for Chapter 11.",
            "AIG bailed out by Fed.",
            "Reserve Primary Fund breaks the buck."
        ]
    },
    {
        "date": "1987-10-23",
        "title": "MARKET MAYHEM: BLACK MONDAY AFTERMATH",
        "summary": "'Shell-Shocked'. On October 19th, the Dow Jones Industrial Average fell 22.6% in a single day. 508 points. It was the largest one-day percentage drop in history.",
        "type": "HISTORICAL ARCHIVE",
        "filename": "newsletter_market_mayhem_oct_1987.html",
        "content_highlights": [
            "Dow drops 22.6% in one day.",
            "Portfolio insurance blamed.",
            "Greenspan pledges liquidity."
        ]
    }
]

# Simple HTML Template (Minimalist Cyber)
HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ADAM v23.5 :: {title}</title>
    <link rel="stylesheet" href="css/style.css">
    <link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Inter:wght@300;400;600;700&family=Playfair+Display:ital,wght@0,400;0,700;1,400&display=swap" rel="stylesheet">
    <style>
        :root {{
            --paper-bg: #fdfbf7;
            --ink-color: #1a1a1a;
            --accent-red: #cc0000;
        }}
        .newsletter-container {{
            max-width: 900px;
            margin: 40px auto;
            background: var(--paper-bg);
            color: var(--ink-color);
            padding: 60px;
            font-family: 'Georgia', 'Times New Roman', serif;
            box-shadow: 0 0 50px rgba(0,0,0,0.5);
            position: relative;
        }}
        .cyber-badge-overlay {{
            position: absolute;
            top: -15px;
            right: -15px;
            background: #050b14;
            border: 1px solid #00f3ff;
            color: #00f3ff;
            padding: 8px 16px;
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.8rem;
            box-shadow: 0 0 15px rgba(0, 243, 255, 0.5);
            z-index: 10;
        }}
        h1.title {{
            font-family: 'Playfair Display', serif;
            font-size: 3rem;
            border-bottom: 4px solid var(--ink-color);
            padding-bottom: 20px;
            margin-bottom: 30px;
            letter-spacing: -1px;
            line-height: 1.1;
        }}
        .meta-header {{
            display: flex;
            justify-content: space-between;
            border-bottom: 1px solid #ccc;
            padding-bottom: 20px;
            margin-bottom: 40px;
            font-family: 'Inter', sans-serif;
            font-size: 0.9rem;
            color: #555;
            text-transform: uppercase;
            letter-spacing: 1px;
        }}
        h2 {{
            font-family: 'Inter', sans-serif;
            font-weight: 700;
            font-size: 1.5rem;
            margin-top: 40px;
            margin-bottom: 15px;
            color: var(--accent-red);
            border-left: 4px solid var(--accent-red);
            padding-left: 15px;
        }}
        p {{ line-height: 1.8; margin-bottom: 20px; font-size: 1.05rem; }}
        ul {{ margin-bottom: 20px; }}
        li {{ margin-bottom: 10px; line-height: 1.6; }}
        .back-link {{ position: fixed; top: 20px; left: 20px; z-index: 100; }}
        .cyber-btn {{
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.75rem;
            padding: 6px 12px;
            border: 1px solid #444;
            color: #e0e0e0;
            background: rgba(0,0,0,0.8);
            border-radius: 2px;
            text-transform: uppercase;
            text-decoration: none;
        }}
        .cyber-btn:hover {{ border-color: #00f3ff; color: #00f3ff; }}
    </style>
</head>
<body style="background: #050b14;">
    <a href="market_mayhem_archive.html" class="cyber-btn back-link">&larr; BACK TO ARCHIVE</a>

    <div class="newsletter-container">
        <div class="cyber-badge-overlay">TYPE: {type} // ADAM v23.5</div>
        <h1 class="title">{title}</h1>
        <div class="meta-header">
            <div><strong>Date:</strong> {date}</div>
            <div><strong>Source:</strong> Apex Financial Architect</div>
            <div><strong>Clearance:</strong> Institutional</div>
        </div>

        <p style="font-size: 1.2rem; font-style: italic; color: #444; border-bottom: 1px solid #eee; padding-bottom: 20px;">
            <strong>Executive Summary:</strong> {summary}
        </p>

        <h2>I. SITUATION REPORT</h2>
        <p>Market conditions are evolving rapidly. Key developments include:</p>
        <ul>
            {highlights_html}
        </ul>

        <h2>II. STRATEGIC IMPLICATIONS</h2>
        <p>
            Based on the current volatility profile, we recommend a defensive posture.
            The risk-reward skew has shifted unfavorably for long-duration assets.
            Capital preservation is paramount in this regime.
        </p>

        <div style="margin-top: 60px; font-size: 0.8rem; color: #999; border-top: 1px solid #eee; padding-top: 20px;">
            <p><strong>Disclaimer:</strong> Generated by Adam v23.5 for simulation purposes.</p>
        </div>
    </div>
</body>
</html>
"""

def generate_files():
    """Generates the static HTML files for the newsletters defined in NEWSLETTER_DATA."""
    print("Generating newsletter files...")

    # Ensure output directory exists
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    for item in NEWSLETTER_DATA:
        filepath = os.path.join(OUTPUT_DIR, item["filename"])

        # Check if file already exists to avoid overwriting manual edits?
        # For now, we overwrite to ensure consistency with the data source,
        # but in a real scenario we might want to be careful.
        # Since this is a generator script, "source of truth" is here.

        # Build highlights list
        highlights = ""
        for h in item["content_highlights"]:
            highlights += f"<li>{h}</li>\n"

        content = HTML_TEMPLATE.format(
            title=item["title"],
            date=item["date"],
            summary=item["summary"],
            type=item["type"],
            highlights_html=highlights
        )

        with open(filepath, "w", encoding='utf-8') as f:
            f.write(content)
        print(f"Created/Updated: {filepath}")

def scan_newsletters():
    """Scans the showcase directory for newsletter HTML files and extracts metadata."""
    print(f"Scanning {OUTPUT_DIR} for newsletters...")
    newsletters = []

    # Pattern to match newsletter files
    files = glob.glob(os.path.join(OUTPUT_DIR, "newsletter_*.html"))

    for filepath in files:
        filename = os.path.basename(filepath)

        # Skip the template itself if it exists or other non-content files
        if filename == "newsletter_market_mayhem.html":
            pass

        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()

            # Extract Data
            title_match = TITLE_RE.search(content)
            date_match = DATE_RE.search(content)
            summary_match = SUMMARY_RE.search(content)
            type_match = TYPE_RE.search(content)

            if title_match and date_match:
                title = title_match.group(1).strip()
                date_str = date_match.group(1).strip()
                summary = summary_match.group(1).strip() if summary_match else "No summary available."
                msg_type = type_match.group(1).strip() if type_match else "UNKNOWN"

                # Cleanup summary if it has HTML tags
                summary = re.sub(r'<[^>]+>', '', summary)

                newsletters.append({
                    "date": date_str,
                    "title": title,
                    "summary": summary,
                    "filename": filename,
                    "type": msg_type
                })
        except Exception as e:
            print(f"Error parsing {filename}: {e}")

    # Sort by date descending
    def parse_date(d):
        try:
            return datetime.strptime(d, "%Y-%m-%d")
        except:
            return datetime.min

    newsletters.sort(key=lambda x: parse_date(x["date"]), reverse=True)
    return newsletters

def generate_archive_page():
    print("Generating archive page...")

    # We scan to get EVERYTHING on disk (generated + manual)
    all_newsletters = scan_newsletters()
    print(f"Found {len(all_newsletters)} newsletters.")

    # Group by Year
    grouped = {}
    for item in all_newsletters:
        try:
            year = item["date"].split("-")[0]
            if len(year) != 4 or not year.isdigit():
                year = "UNKNOWN"
        except:
            year = "UNKNOWN"

        if year not in grouped:
            grouped[year] = []
        grouped[year].append(item)

    # Build HTML List
    list_html = ""

    # Sort years descending
    sorted_years = sorted([y for y in grouped.keys() if y != "UNKNOWN"], reverse=True)
    if "UNKNOWN" in grouped:
        sorted_years.append("UNKNOWN")

    for year in sorted_years:
        header_text = f"{year} ARCHIVE"

        # NOTE: Logic changed to prevent duplicate headers.
        # We simply use the Year.
        # If we wanted to group all pre-2020, we would need to restructure 'grouped' dictionary beforehand.
        # But Year-based is cleaner.

        list_html += f'<div class="year-header">{header_text}</div>\n'

        for item in grouped[year]:
            type_badge = ""
            t_upper = item["type"].upper()

            if "FLASH" in t_upper:
                type_badge = '<span class="type-badge flash">FLASH</span>'
            elif "WEEKLY" in t_upper:
                type_badge = '<span class="type-badge weekly">WEEKLY</span>'
            elif "HISTORICAL" in t_upper:
                type_badge = '<span class="type-badge historical">HISTORICAL</span>'

            list_html += f"""
            <div class="archive-item">
                <div style="flex-grow: 1;">
                    <div style="display:flex; align-items:center; gap:10px;">
                        <span class="item-date">{item["date"]}</span>
                        {type_badge}
                    </div>
                    <h3 class="item-title">{item["title"]}</h3>
                    <div class="item-summary">{item["summary"]}</div>
                </div>
                <a href="{item["filename"]}" class="read-btn">DECRYPT &rarr;</a>
            </div>
            """

    # Full Page HTML
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

        .archive-list {{ max-width: 1000px; margin: 40px auto; display: flex; flex-direction: column; gap: 15px; padding: 0 20px; }}

        .year-header {{
            font-family: 'JetBrains Mono'; font-size: 2rem; color: #333; font-weight: bold;
            border-bottom: 2px solid #222; margin-top: 40px; margin-bottom: 20px;
        }}

        .archive-item {{
            border: 1px solid #333; background: rgba(255, 255, 255, 0.03); padding: 20px;
            display: flex; justify-content: space-between; align-items: center; gap: 20px;
            transition: all 0.2s ease;
        }}
        .archive-item:hover {{ border-color: var(--accent-color); background: rgba(204, 0, 0, 0.05); transform: translateX(5px); }}

        .item-date {{ font-family: 'JetBrains Mono'; font-size: 0.8rem; color: var(--accent-color); }}
        .item-title {{ font-size: 1.2rem; font-weight: 700; color: #fff; margin: 5px 0; font-family: 'Inter'; }}
        .item-summary {{ color: #888; font-size: 0.85rem; max-width: 600px; }}

        .read-btn {{
            padding: 8px 16px; border: 1px solid var(--accent-color); color: var(--accent-color);
            font-family: 'JetBrains Mono'; text-transform: uppercase; font-size: 0.75rem;
            background: rgba(0,0,0,0.5); text-decoration: none; white-space: nowrap;
        }}
        .read-btn:hover {{ background: var(--accent-color); color: #000; }}

        .type-badge {{ font-size: 0.6rem; padding: 2px 6px; border-radius: 2px; font-weight: bold; font-family: 'JetBrains Mono'; }}
        .flash {{ background: #ff0000; color: white; }}
        .weekly {{ background: #ffa500; color: black; }}
        .historical {{ background: #666; color: white; }}

        @media (max-width: 768px) {{
            .archive-item {{ flex-direction: column; align-items: flex-start; }}
            .read-btn {{ width: 100%; text-align: center; }}
        }}
    </style>
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
    print(f"Updated: {ARCHIVE_FILE}")

if __name__ == "__main__":
    # 1. Regenerate existing/missing files based on source of truth
    generate_files()

    # 2. Update the archive page by scanning everything on disk
    generate_archive_page()
