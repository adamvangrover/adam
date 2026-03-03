import json
import os
from datetime import datetime

NEWSLETTER_DATA_PATH = "showcase/data/newsletter_data.json"

new_entries = [
    {
        "date": "2026-03-02",
        "title": "🔴 SYSTEM STATUS: DEGRADED (Volatility Spike)",
        "summary": "Markets open March with significant friction. Manufacturing PMI comes in hot, triggering algorithmic selling.",
        "type": "DAILY_BRIEFING",
        "filename": "Daily_Briefing_2026_03_02.html",
        "is_sourced": True,
        "full_body": """<h3>📡 Signal Integrity: The Growth Glitch</h3>
<p>March begins with a sudden structural failure. The <a href="../market_mayhem_graph.html" style="color: #22d3ee;">S&P 500</a> gap-down opened today, ultimately shedding 1.8% as the ISM Manufacturing PMI printed an unexpected 52.4—the highest expansionary read in over a year.</p>
<p><strong>Credit Dominance Check:</strong> The "good news is bad news" algorithm is back online. Stronger factory orders instantly repainted the rate-cut horizon, causing the 10-Year yield to violently spike 14bps to 4.59%.</p>
<h3>🏮 Artifacts</h3>
<ul>
<li><strong><a href="../market_mayhem_graph.html" style="color: #22d3ee;">Bitcoin</a> ($62,100 | -4.2%):</strong> Risk-off flows bled heavily into the crypto sector.</li>
<li><strong>Semiconductors (-3.1%):</strong> The hardware layer took the brunt of the rate shock.</li>
</ul>
<p><strong>The Glitch:</strong> We are trapped in a feedback loop. Until the economy actually shows signs of breaking, the cost of capital will continue to choke off the valuation multiples of the future.</p>""",
        "source_priority": 3,
        "conviction": 90,
        "sentiment_score": 15
    },
    {
        "date": "2026-03-03",
        "title": "🟡 SYSTEM STATUS: NOMINAL (Mean Reversion)",
        "summary": "The algorithm finds support. Yesterday's panic yields a tentative dead-cat bounce.",
        "type": "DAILY_BRIEFING",
        "filename": "Daily_Briefing_2026_03_03.html",
        "is_sourced": True,
        "full_body": """<h3>📡 Signal Integrity: The Reversion Patch</h3>
<p>The simulation stabilized today. The <a href="../market_mayhem_graph.html" style="color: #22d3ee;">S&P 500</a> recovered 0.9%, finding technical support right at the 6,100 node.</p>
<p><strong>Credit Dominance Check:</strong> Yields paused their ascent. The 10-Year held flat at 4.58%, allowing equity algorithms to safely execute buy-the-dip subroutines.</p>
<h3>🏮 Artifacts</h3>
<ul>
<li><strong>Utilities (+2.1%):</strong> Defensive sectors lead the way, proving this is a nervous recovery, not a bullish breakout.</li>
<li><strong>Tesla (TSLA | +4.5%):</strong> An outlier today after announcing a new factory timeline in India.</li>
</ul>
<p><strong>The Glitch:</strong> This is a low-conviction patch. Volume is light. The system is merely resting before the Non-Farm Payrolls data drop later this week.</p>""",
        "source_priority": 3,
        "conviction": 50,
        "sentiment_score": 55
    }
]

def add_entries():
    if not os.path.exists(NEWSLETTER_DATA_PATH):
        print("Data file not found.")
        return

    with open(NEWSLETTER_DATA_PATH, 'r') as f:
        data = json.load(f)

    existing_keys = {(item.get('date'), item.get('title')) for item in data}
    added_count = 0

    for entry in new_entries:
        if (entry['date'], entry['title']) not in existing_keys:
            data.insert(0, entry) # Add to top
            added_count += 1
            print(f"Added entry for {entry['date']}")

    # Sort by date desc
    data.sort(key=lambda x: x.get('date', ''), reverse=True)

    with open(NEWSLETTER_DATA_PATH, 'w') as f:
        json.dump(data, f, indent=4)

    print(f"Newsletter data updated. Added {added_count} new entries.")

if __name__ == "__main__":
    add_entries()
