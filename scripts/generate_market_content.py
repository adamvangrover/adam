import json
import os
from datetime import datetime

# Metadata for new reports
reports = [
    {
        "id": "conviction_btc_feb26",
        "title": "Conviction Report: Bitcoin (BTC)",
        "subtitle": "Digital Gold or Digital Leverage?",
        "date": "2026-02-25",
        "type": "CONVICTION_REPORT",
        "ticker": "BTC",
        "content": """<h2>Executive Summary</h2>
<p>Bitcoin has reached a critical juncture. After flushing to $60k, it has reclaimed $68k, but the structure is fragile. The correlation with the Nasdaq is nearing 1.0, destroying its thesis as an uncorrelated hedge.</p>
<h2>The Thesis: Leverage Flush Complete?</h2>
<p>On-chain metrics show a massive reset in Open Interest. The "Tourist Longs" have been liquidated. We are entering a zone of accumulation for long-term holders (LTH).</p>
<h3>Key Drivers</h3>
<ul>
<li><strong>Sovereign Adoption:</strong> Rumors of a G7 central bank adding BTC to reserves are persistent.</li>
<li><strong>ETF Flows:</strong> BlackRock's IBIT is seeing renewed inflows after a week of outflows.</li>
</ul>
<h2>Verdict: ACCUMULATE</h2>
<p>We are raising our conviction to HIGH. Volatility is the price of admission.</p>""",
        "sentiment": 65,
        "conviction": 80
    },
    {
        "id": "conviction_rivn_feb26",
        "title": "Conviction Report: Rivian (RIVN)",
        "subtitle": "The Zombie Bounce",
        "date": "2026-02-25",
        "type": "CONVICTION_REPORT",
        "ticker": "RIVN",
        "content": """<h2>Executive Summary</h2>
<p>Rivian's recent 26% surge is a classic "Short Squeeze" in a "Zombie Company." While the Amazon partnership remains a lifeline, the cash burn is unsustainable without a capital raise in Q3.</p>
<h2>The Trap</h2>
<p>Investors are mistaking a liquidity pump for a fundamental turnaround. The EV sector is undergoing a brutal consolidation.</p>
<h2>Verdict: SELL STRENGTH</h2>
<p>Use this rally to exit. The bankruptcy risk in 2027 is non-zero.</p>""",
        "sentiment": 20,
        "conviction": 90
    },
    {
        "id": "conviction_amat_feb26",
        "title": "Conviction Report: Applied Materials (AMAT)",
        "subtitle": "The Foundry Floor",
        "date": "2026-02-25",
        "type": "CONVICTION_REPORT",
        "ticker": "AMAT",
        "content": """<h2>Executive Summary</h2>
<p>While software AI plays (SaaS) are faltering, the hardware layer remains robust. AMAT is the pick-and-shovel play for the Sovereign AI capex boom.</p>
<h2>Foundry Dominance</h2>
<p>Every new fab built in Arizona or Germany requires AMAT tools. The backlog is at record levels.</p>
<h2>Verdict: STRONG BUY</h2>
<p>The safest way to play the AI Supercycle.</p>""",
        "sentiment": 85,
        "conviction": 95
    },
    {
        "id": "conviction_s_feb26",
        "title": "Conviction Report: SentinelOne (S)",
        "subtitle": "Cyber Debt Refinancing",
        "date": "2026-02-25",
        "type": "CONVICTION_REPORT",
        "ticker": "S",
        "content": """<h2>Executive Summary</h2>
<p>Cybersecurity is no longer discretionary spending; it is a utility. SentinelOne is outperforming CrowdStrike on a relative basis due to its AI-native architecture.</p>
<h2>The Debt Wall</h2>
<p>Concern about 2026 maturities is overblown. Free Cash Flow is turning positive.</p>
<h2>Verdict: BUY</h2>
<p>A prime acquisition target for legacy tech.</p>""",
        "sentiment": 75,
        "conviction": 70
    },
    {
        "id": "conviction_owl_feb26",
        "title": "Conviction Report: Blue Owl (OWL)",
        "subtitle": "Private Credit Fracture",
        "date": "2026-02-20",
        "type": "CONVICTION_REPORT",
        "ticker": "OWL",
        "content": """<h2>Executive Summary</h2>
<p>The halting of redemptions is a massive red flag. Private Credit has acted as a volatility dampener, but liquidity is drying up.</p>
<h2>Contagion Risk</h2>
<p>If OWL cannot meet redemption requests, the entire "Shadow Banking" sector faces a re-pricing event.</p>
<h2>Verdict: AVOID</h2>
<p>Liquidity risk is mispriced.</p>""",
        "sentiment": 10,
        "conviction": 85
    }
]

# Template for Report
TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ADAM v24.0 :: {title}</title>
    <link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Inter:wght@300;400;600;700&display=swap" rel="stylesheet">
    <style>
        body {{ background: #0f172a; color: #e2e8f0; font-family: 'Inter', sans-serif; padding: 40px; }}
        .container {{ max-width: 800px; margin: 0 auto; background: rgba(30,41,59,0.5); padding: 40px; border: 1px solid #334155; border-radius: 12px; }}
        h1 {{ font-family: 'JetBrains Mono'; color: #00f3ff; margin-bottom: 5px; }}
        h2 {{ font-family: 'JetBrains Mono'; color: #22c55e; border-bottom: 1px solid #333; padding-bottom: 10px; margin-top: 30px; }}
        .subtitle {{ color: #94a3b8; margin-bottom: 30px; font-family: 'JetBrains Mono'; }}
        .badge {{ background: #334155; color: #fff; padding: 4px 8px; border-radius: 4px; font-size: 0.8rem; font-family: 'JetBrains Mono'; }}
        .verdict {{ font-size: 1.5rem; font-weight: bold; color: #f59e0b; margin: 20px 0; border: 1px solid #f59e0b; display: inline-block; padding: 10px 20px; }}
    </style>
</head>
<body>
    <div class="container">
        <div style="display:flex; justify-content:space-between; align-items:center;">
            <span class="badge">{date}</span>
            <span class="badge">{type}</span>
        </div>
        <h1>{title}</h1>
        <div class="subtitle">{subtitle}</div>

        {content}

        <div style="margin-top: 50px; border-top: 1px solid #333; padding-top: 20px; color: #666; font-size: 0.8rem; font-family: 'JetBrains Mono';">
            GENERATED BY ADAM v24.0 // SYSTEM 2
        </div>
    </div>
</body>
</html>
"""

INDEX_FILE = "showcase/data/market_mayhem_index.json"

def main():
    # Load Index
    try:
        with open(INDEX_FILE, "r") as f:
            index_data = json.load(f)
    except FileNotFoundError:
        index_data = []

    for r in reports:
        # Generate Filename
        filename = f"{r['id']}.html"
        filepath = f"showcase/{filename}"

        # Write HTML
        html = TEMPLATE.format(
            title=r['title'],
            subtitle=r['subtitle'],
            date=r['date'],
            type=r['type'],
            content=r['content']
        )
        with open(filepath, "w") as f:
            f.write(html)
        print(f"Generated {filepath}")

        # Update Index
        # Check if exists
        exists = False
        for entry in index_data:
            if entry.get("filename") == filename:
                exists = True
                # Update?
                break

        if not exists:
            new_entry = {
                "title": f"{r['title']}: {r['subtitle']}",
                "date": r['date'],
                "summary": r['subtitle'], # Short summary
                "type": r['type'],
                "full_body": r['content'],
                "sentiment_score": r['sentiment'],
                "conviction": r['conviction'],
                "entities": {"tickers": [r['ticker']], "keywords": ["Analysis", "Conviction"]},
                "filename": filename,
                "is_sourced": True,
                "metrics_json": "{}"
            }
            index_data.append(new_entry)

    # Write Index
    with open(INDEX_FILE, "w") as f:
        json.dump(index_data, f, indent=2)
    print("Index updated.")

if __name__ == "__main__":
    main()
