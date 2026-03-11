
import sys
import os
import logging
from datetime import datetime
from bs4 import BeautifulSoup

# Add root to pythonpath
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.data_sources.yfinance_market_data import YFinanceMarketData

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def fetch_market_data():
    client = YFinanceMarketData()
    symbols = {
        "S&P 500": "^GSPC",
        "Dow Jones": "^DJI",
        "Nasdaq": "^IXIC",
        "VIX": "^VIX",
        "Bitcoin": "BTC-USD"
    }

    results = {}
    for name, symbol in symbols.items():
        logger.info(f"Fetching data for {name} ({symbol})...")
        snap = client.get_snapshot(symbol)
        if snap and snap.get("current_price"):
            current = snap["current_price"]
            open_price = snap.get("open")

            change_pct = 0.0
            if open_price:
                change_pct = ((current - open_price) / open_price) * 100

            results[name] = {
                "price": current,
                "change_pct": change_pct
            }
        else:
            logger.warning(f"Failed to fetch data for {name}")
            results[name] = {"price": 0.0, "change_pct": 0.0}

    return results

def update_daily_briefing(html_path, market_data):
    if not os.path.exists(html_path):
        logger.error(f"File not found: {html_path}")
        return

    with open(html_path, "r", encoding="utf-8") as f:
        html_content = f.read()

    soup = BeautifulSoup(html_content, "html.parser")

    # 1. Update Date
    current_date = datetime.now().strftime("%b %d, %Y")
    # Find the span with the date. It has style="font-size: 1rem; color: #94a3b8; margin-left: auto;"
    # Since classes are not unique for date, we look for the span inside h1 that is NOT the status badge
    h1 = soup.find("h1")
    if h1:
        spans = h1.find_all("span")
        for span in spans:
            if "status-badge" not in span.get("class", []):
                span.string = current_date
                logger.info(f"Updated date to: {current_date}")

    # 2. Update Signal Integrity / Market Pulse
    # We replace the first paragraph in the first .section div after the h1
    first_section = soup.find("div", class_="section")
    if first_section:
        # Construct the summary string
        summary_parts = []
        for name, data in market_data.items():
            price_fmt = "{:,.2f}".format(data["price"])
            change_fmt = "{:+.2f}%".format(data["change_pct"])
            summary_parts.append(f"<strong>{name}:</strong> {price_fmt} (<span style='color: {'#22c55e' if data['change_pct'] >= 0 else '#f87171'}'>{change_fmt}</span>)")

        market_summary_html = " | ".join(summary_parts)

        # Find the paragraph starting with "Signal Integrity"
        # Or just replace the first paragraph if we assume structure
        paragraphs = first_section.find_all("p")
        if paragraphs:
            # We replace the first paragraph with our new market summary
            # But we want to keep the "Signal Integrity" bold part if we can, or just replace it all

            new_p_content = f"<strong>📡 Live Market Pulse:</strong> {market_summary_html}"

            # Create a new tag for replacement
            new_p = soup.new_tag("p")
            new_p.append(BeautifulSoup(new_p_content, "html.parser"))

            # Replace the first paragraph (which was the Super Bowl one)
            paragraphs[0].replace_with(new_p)
            logger.info("Updated Market Pulse section.")

    # 3. Add Live Views Section
    # We want to insert this before "The Glitch" section or after "Artifacts"
    # Find "Artifacts" section
    artifacts_header = soup.find("h2", string=lambda t: "Artifacts" in t if t else False)
    if artifacts_header:
        artifacts_section = artifacts_header.find_parent("div", class_="section")

        # Create new section
        new_section = soup.new_tag("div", attrs={"class": "section"})

        h2 = soup.new_tag("h2")
        h2.string = "⚡ Live Intelligence Feeds"
        new_section.append(h2)

        # Links container
        links_div = soup.new_tag("div", attrs={"style": "display: grid; grid-template-columns: 1fr 1fr; gap: 15px;"})

        links = [
            ("Market Mayhem Graph (3D)", "market_mayhem_graph.html", "🌊"),
            ("Strategic Command", "apps/strategic_command.html", "🗺️"),
            ("Valuation Lab", "apps/valuation_lab.html", "🧮"),
            ("Structuring Workbench", "apps/structuring_workbench.html", "🏗️")
        ]

        for title, url, icon in links:
            a = soup.new_tag("a", href=url, attrs={
                "style": "display: block; padding: 15px; background: rgba(30, 41, 59, 0.5); border: 1px solid #334155; border-radius: 8px; color: #e2e8f0; text-decoration: none; font-weight: 600; transition: all 0.2s ease;"
            })
            a["onmouseover"] = "this.style.background='rgba(34, 197, 94, 0.1)'; this.style.borderColor='#22c55e';"
            a["onmouseout"] = "this.style.background='rgba(30, 41, 59, 0.5)'; this.style.borderColor='#334155';"

            a.append(BeautifulSoup(f"<span style='margin-right: 8px;'>{icon}</span> {title}", "html.parser"))
            links_div.append(a)

        new_section.append(links_div)

        # Insert after artifacts section
        artifacts_section.insert_after(new_section)
        logger.info("Added Live Intelligence Feeds section.")

    # Write back to file
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(str(soup.prettify())) # prettify ensures valid HTML structure

    logger.info(f"Successfully updated {html_path}")

def main():
    logger.info("Starting Daily Briefing Update...")
    market_data = fetch_market_data()

    html_path = os.path.join("showcase", "daily_briefing.html")
    update_daily_briefing(html_path, market_data)

if __name__ == "__main__":
    main()
