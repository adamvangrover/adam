import os
import re
import glob
import json
import yaml
from datetime import datetime

# --- Configuration ---
OUTPUT_DIR = "showcase"
SOURCE_DIRS = [
    "core/libraries_and_archives/newsletters",
    "core/libraries_and_archives/reports"
]
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
                    # Handle list of dicts (e.g., dcf_details revenue_growth)
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
                     print(f"Error parsing JSON/YAML {filepath}: {ye}")
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

                    if 'ideas' in section:
                        for idea in section['ideas']:
                            body_html += f"<h3>{idea.get('asset', 'Asset')}</h3>"
                            body_html += f"<p><strong>Rationale:</strong> {idea.get('rationale','')}</p>"

        # Schema 2: 'analysis' dict (NVDA style)
        elif "analysis" in data:
            if summary: body_html += f"<h2>Executive Summary</h2><p>{summary}</p>"
            body_html += format_dict_as_html(data['analysis'])

            # Add other top-level keys like 'trading_levels'
            for key in ['trading_levels', 'financial_performance', 'valuation']:
                if key in data and key not in data['analysis']: # avoid dups
                     body_html += f"<h2>{key.replace('_', ' ').title()}</h2>"
                     if isinstance(data[key], dict):
                         body_html += format_dict_as_html(data[key], 3)
                     else:
                         body_html += f"<p>{data[key]}</p>"

        # Schema 3: Just summary + details (Market Overviews style, though usually list)
        elif summary:
             body_html += f"<p>{summary}</p>"
             # Add any other string fields as paragraphs
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

        return {
            "title": title,
            "date": date_str,
            "summary": summary,
            "type": type_,
            "full_body": body_html,
            "sentiment_score": 50, # Default
            "filename": out_filename,
            "is_sourced": True
        }
    except Exception as e:
        print(f"Error processing {filepath}: {e}")
        return None

def parse_txt_file(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()

        filename = os.path.basename(filepath)

        # Metadata Extraction attempt
        title_match = re.search(r'COMPREHENSIVE.*?:\s*(.*?)$', content, re.MULTILINE)
        title = title_match.group(1).strip() if title_match else filename.replace('.txt', '').replace('_', ' ')

        date_match = re.search(r'DATE:\s*(.*?)$', content, re.MULTILINE)
        date_str = parse_date(date_match.group(1)) if date_match else "2025-01-01"

        # Type
        type_ = "REPORT"
        if "Deep Dive" in content or "DEEP DIVE" in content: type_ = "DEEP_DIVE"

        # Body - just wrap in pre or do basic p splitting
        # Use simple_md_to_html logic but simpler
        body_html = f"<div style='white-space: pre-wrap; font-family: inherit;'>{content}</div>"

        # Summary
        summary = "Comprehensive report."
        summary_match = re.search(r'EXECUTIVE SUMMARY.*?\n\n(.*?)\n', content, re.DOTALL)
        if summary_match:
             summary = summary_match.group(1)[:300] + "..."

        return {
            "title": title,
            "date": date_str,
            "summary": summary,
            "type": type_,
            "full_body": body_html,
            "sentiment_score": 50,
            "filename": filename.replace('.txt', '.html'),
            "is_sourced": True
        }
    except Exception as e:
        print(f"Error parsing TXT {filepath}: {e}")
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

# --- Static Data (Historical & Simulated) ---

NEWSLETTER_DATA = [
    # 2026 The Reflationary Agentic Boom
    {
        "date": "2026-01-12",
        "title": "GLOBAL MACRO-STRATEGIC OUTLOOK 2026: THE REFLATIONARY AGENTIC BOOM",
        "summary": "As markets open on January 12, 2026, the global financial system has entered a new regime: the Reflationary Agentic Boom. Paradoxical growth, sticky inflation, and the 'Binary Big Bang' of Agentic AI.",
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
    # 2025 Monthly
    {
        "date": "2025-09-14",
        "title": "THE LIQUIDITY TRAP",
        "summary": "Fed pauses cuts as core services inflation sticks. Small caps get crushed. The case for 'Quality' factor investing.",
        "type": "MONTHLY",
        "filename": "newsletter_market_mayhem_sep_2025.html",
        "is_sourced": True,
        "full_body": """
        <p>The Federal Reserve's decision to hold rates steady at 5.25% has sent shockwaves through the small-cap sector. The Russell 2000 has plummeted 8% in just two weeks, reflecting growing fears of a credit crunch for regional banks and smaller enterprises dependent on floating-rate debt.</p>
        <p>Core services inflation remains stubbornly high at 4.2%, driven primarily by shelter and healthcare costs. This data point has effectively killed the "pivot" narrative for Q4 2025. The yield curve remains inverted, with the 2s10s spread widening to -45bps.</p>
        <p><strong>Sector Analysis:</strong> We are seeing a massive rotation into 'Quality'—companies with strong balance sheets and high free cash flow yields. Mega-cap tech (AAPL, MSFT) is acting as a safe haven, while unprofitable tech and highly leveraged industrials are being sold off aggressively.</p>
        """
    },
    {
        "date": "2025-08-14",
        "title": "SUMMER DOLDRUMS & AI FATIGUE",
        "summary": "Volume dries up. AI Capex questions emerge. Nvidia earnings preview: Is the bar too high?",
        "type": "MONTHLY",
        "filename": "newsletter_market_mayhem_aug_2025.html",
        "is_sourced": True,
        "full_body": """
        <p>Trading volumes have collapsed to year-to-date lows as Wall Street heads to the Hamptons. However, beneath the calm surface, cracks are forming in the AI narrative. Several hyperscalers (GOOGL, META) have hinted at 'optimizing' capital expenditures, sparking fears of an AI spending slowdown.</p>
        <p>All eyes are on Nvidia's upcoming earnings. The buy-side whisper numbers are astronomical, setting a bar that may be impossible to clear. A disappointment here could trigger a broader correction in the semiconductor index (SOXX).</p>
        <p><strong>Strategy:</strong> We are observing a quiet rotation into Healthcare (XLV) and Utilities (XLU) as investors seek yield and defensive characteristics amidst the uncertainty.</p>
        """
    },
    {
        "date": "2025-07-14",
        "title": "CRYPTO REGULATION SHOCK",
        "summary": "SEC 2.0 launches 'Operation Chokepoint'. DeFi protocols under siege. Bitcoin flight to safety narrative tested.",
        "type": "MONTHLY",
        "filename": "newsletter_market_mayhem_jul_2025.html",
        "is_sourced": True,
        "full_body": """
        <p>The SEC has launched a coordinated enforcement blitz against major DeFi protocols and centralized exchanges, dubbed 'Operation Chokepoint 2.0'. The immediate impact has been a bloodbath in the altcoin market, with ETH/BTC ratios hitting multi-year lows.</p>
        <p>Bitcoin, however, is showing relative strength. The 'flight to safety' narrative within the crypto ecosystem is funneling capital into BTC. Institutional investors view the regulatory purge as a necessary cleansing before true mass adoption can occur.</p>
        <p><strong>Outlook:</strong> Expect continued volatility. The legal battles will take years to resolve. In the meantime, 'Code is Law' is being tested by 'Law is Law'.</p>
        """
    },
    {
        "date": "2025-06-14",
        "title": "THE DOLLAR WRECKING BALL",
        "summary": "DXY breaks 108. Emerging Market currencies collapse. Carry trade unwind begins.",
        "type": "MONTHLY",
        "filename": "newsletter_market_mayhem_jun_2025.html",
        "is_sourced": True,
        "full_body": """
        <p>The US Dollar Index (DXY) has shattered resistance at 108, acting as a wrecking ball for global risk assets. The Japanese Yen has collapsed to 160 against the dollar, forcing the BOJ to intervene—unsuccessfully so far.</p>
        <p>Emerging Market (EM) debt is under severe stress. Countries with high dollar-denominated debt burdens are seeing their credit default swap (CDS) spreads blow out. The 'Carry Trade'—borrowing in Yen to buy tech stocks—is unwinding rapidly, adding selling pressure to the Nasdaq.</p>
        <p><strong>Macro View:</strong> The Fed's 'Higher for Longer' stance is diverging from the ECB and BOJ, creating a rate differential vacuum that sucks capital into the USD.</p>
        """
    },
    {
        "date": "2025-05-14",
        "title": "COMMERCIAL REAL ESTATE: THE RECKONING",
        "summary": "Office vacancy hits 25%. Regional banks take haircuts. The 'Extend and Pretend' game ends.",
        "type": "MONTHLY",
        "filename": "newsletter_market_mayhem_may_2025.html",
        "is_sourced": True,
        "full_body": """
        <p>The 'Extend and Pretend' game is officially over. A landmark sale of a Class-A office tower in San Francisco for $100/sqft—down 75% from its 2019 valuation—has set a terrifying new comparable for the market.</p>
        <p>Regional banks (KRE) are taking massive haircuts on their loan portfolios. We expect a wave of consolidation in the banking sector as smaller players are forced to merge or fail. Meanwhile, Private Credit funds are raising record amounts of dry powder to act as the liquidity providers of last resort.</p>
        <p><strong>Investment Implication:</strong> Avoid regional banks. Look for distress-focused alternative asset managers.</p>
        """
    },
    {
        "date": "2025-04-14",
        "title": "Q1 EARNINGS: PROFIT MARGIN SQUEEZE",
        "summary": "Wage inflation eats into margins. Guidance cut across Consumer Discretionary. The Recession is not cancelled, just delayed.",
        "type": "MONTHLY",
        "filename": "newsletter_market_mayhem_apr_2025.html",
        "is_sourced": True,
        "full_body": """
        <p>Q1 earnings season has revealed a troubling trend: profit margin compression. While top-line revenue remains resilient, bottom-line earnings are being eroded by sticky wage inflation (running at 4.5%) and higher input costs.</p>
        <p>Consumer Discretionary giants like Target and Home Depot have cut full-year guidance, citing 'consumer fatigue' and 'shrink' (theft). The narrative is shifting from 'Goldilocks' to 'Stagflation'—slowing growth with persistent inflation.</p>
        <p><strong>Key Metric:</strong> Operating margins for the S&P 500 ex-Energy have contracted by 120bps year-over-year.</p>
        """
    },
    # 2024 Retrospective / Highlights
    {
        "date": "2024-12-14",
        "title": "2024 POST-MORTEM: THE SOFT LANDING MIRACLE",
        "summary": "How the Fed threaded the needle. Inflation down, unemployment low. Can it last?",
        "type": "YEARLY_REVIEW",
        "filename": "newsletter_market_mayhem_dec_2024.html",
        "is_sourced": True,
        "full_body": """
        <p>2024 will go down in history as the year of the 'Soft Landing Miracle'. Against all odds, the Federal Reserve managed to bring inflation down from 9% to 3% without triggering a recession. Unemployment remained historically low at 3.7%.</p>
        <p>The S&P 500 returned a stunning 24%, driven almost entirely by the 'Magnificent 7' tech giants. The AI revolution fueled a productivity boom narrative that offset higher interest rates.</p>
        <p><strong>Retrospective:</strong> The 'Immaculate Disinflation' was real. Supply chains healed, and the labor market rebalanced without mass layoffs. But valuations are now stretched.</p>
        """
    },

    # Historical - REAL DATA
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
    }
]

# --- Templates ---

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
        <div class="paper-sheet">
            <div style="display:flex; justify-content:space-between; font-family:'JetBrains Mono'; font-size:0.8rem; color:#666; margin-bottom:20px;">
                <span>{date}</span>
                <span>TYPE: {type}</span>
            </div>

            <h1 class="title">{title}</h1>

            {full_body}

            <div style="margin-top: 40px; border-top: 1px solid #ccc; padding-top: 20px; font-style: italic; color: #666;">
                End of Report.
            </div>
        </div>

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

# --- Main Generation ---

def generate_archive():
    print("Starting Archive Generation...")

    # 1. Gather all data sources
    all_items = []

    # Scan Source Directories
    for source_dir in SOURCE_DIRS:
        print(f"Scanning {source_dir}...")
        if not os.path.exists(source_dir):
            print(f"Directory not found: {source_dir}")
            continue

        source_files = glob.glob(os.path.join(source_dir, "*"))

        for filepath in source_files:
            item = None
            if filepath.endswith(".json"):
                item = parse_json_file(filepath)
            elif filepath.endswith(".md"):
                item = parse_markdown_file(filepath)
            elif filepath.endswith(".txt"):
                item = parse_txt_file(filepath)

            if item:
                all_items.append(item)

    # 2. Add Hardcoded/Static Data (from NEWSLETTER_DATA)
    # Check if we already have these dates/titles to avoid duplicates
    existing_dates = set(i['date'] for i in all_items)
    
    for static_item in NEWSLETTER_DATA:
        # Check by title or date overlap
        if static_item['date'] not in existing_dates:
            # Ensure required fields exist for static items
            if 'is_sourced' not in static_item:
                static_item['is_sourced'] = True
            all_items.append(static_item)
            existing_dates.add(static_item['date'])

    # 3. Scan for Orphan HTML Files in Showcase
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
                date_str = date_m.group(1).strip()
                # Attempt fix for bad parsed dates
                parsed_date = parse_date(date_str)

                all_items.append({
                    "date": parsed_date,
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
            full_body=item.get("full_body", "")
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
            type_ = item.get("type", "REPORT")
            if type_ == 'MARKET_PULSE': type_color = "#00f3ff"
            elif type_ == 'DEEP_DIVE': type_color = "#cc0000"
            elif type_ == 'NEWSLETTER': type_color = "#ffff00"
            elif type_ == 'AGENT_NOTE': type_color = "#00ff00" # Matrix Green
            elif type_ == 'PERFORMANCE_REVIEW': type_color = "#60a5fa" # Slate Blue
            elif type_ == 'COMPANY_REPORT': type_color = "#f472b6" # Pink
            elif type_ == 'THEMATIC_REPORT': type_color = "#a78bfa" # Purple
            elif type_ == 'MARKET_OUTLOOK': type_color = "#fca5a5" # Light Red

            list_html += f"""
            <div class="archive-item" data-title="{item['title'].lower()}" data-year="{year}">
                <div style="width: 5px; background: {type_color}; margin-right: 15px;"></div>
                <div style="flex-grow: 1;">
                    <div style="display:flex; align-items:center; gap:10px;">
                        <span class="item-date">{item["date"]}</span>
                        <span class="type-badge" style="background:{type_color}; color:#000;">{type_}</span>
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