import json
import os
from datetime import datetime

NEWSLETTER_DATA_PATH = "showcase/data/newsletter_data.json"
STRATEGIC_COMMAND_PATH = "showcase/data/strategic_command.json"

new_entries = [
    {
        "date": "2026-02-12",
        "title": "🔴 SYSTEM STATUS: DEGRADED (Structural Integrity Failing)",
        "summary": "The simulation has hit a massive bottleneck. What began as a 'too-hot' jobs report echo has mutated into a full-scale AI-sector hardware failure.",
        "type": "DAILY_BRIEFING",
        "filename": "Daily_Briefing_2026_02_12.html",
        "is_sourced": True,
        "full_body": """<h3>📡 Signal Integrity: The Speculative Unwind</h3>
<p>The simulation has hit a massive bottleneck. What began as a "too-hot" jobs report echo has mutated into a full-scale AI-sector hardware failure. The <a href="../market_mayhem_graph.html" style="color: #22d3ee;">S&P 500</a> suffered its largest one-day percentage decline of the year, plummeting -1.57% to 6,832.76. The frame rate is officially stuttering.</p>
<p><strong>Credit Dominance Check:</strong> Today we are seeing a Total Signal Inversion. In a healthy recovery, a stock-market dump triggers a "flight to safety" in bonds, driving yields down. While the 10-Year Treasury Yield did ease to 4.10% (-8bps), the move wasn't driven by stability, but by a panic pivot away from speculative positions.</p>
<p><strong>The Signal:</strong> High Yield spreads (HYG/JNK) are beginning to widen significantly as investors question the viability of automation-heavy business models. When the Nasdaq drops -2% and Bitcoin breaks its structural support, the "Safe Haven" of Treasuries isn't a reward—it's an admission that the growth code is corrupted.</p>
<h3>🏮 Artifacts</h3>
<ul>
<li><strong><a href="../market_mayhem_graph.html" style="color: #22d3ee;">Bitcoin</a> ($66,297 | -3.25%):</strong> The "Digital Gold" mask has shattered. Bitcoin has now fallen four consecutive days, losing nearly 8% this week. The $60k support floor is no longer a theoretical boundary—it’s the next target for the margin-call bots.</li>
<li><strong>Nasdaq Composite (-2%):</strong> A fresh wave of "AI Fears" is slamming the tech architecture. Investors are suddenly questioning if the massive CapEx spend on hardware will ever render into real-world cash flow.</li>
<li><strong>VIX ($18.69):</strong> Volatility has gapped up. We are no longer in the "seasonal drift"; we are in a "spike peak" buy-signal zone, though it requires a 22-day hold to prove the floor is in.</li>
</ul>
<h3>🌀 The Glitch</h3>
<p>"The market just realized that you can't build a trillion-dollar economy on the promise of an algorithm that doesn't know how to do the laundry. Today's sell-off was the sound of a thousand 'AI-Optimism' plugins being uninstalled at once. We are watching a decoupling: the 10-Year yield is falling because the 'Risk-On' dream is dying, not because the economy is healthy. The glitch isn't the drop; the glitch was the belief that $70k Bitcoin and a $7k S&P could coexist with 4.2% yields. The system is finally running a self-diagnostic, and it doesn't like the results."</p>
<p><strong>Next Transmission:</strong> Friday, Feb 13, 18:00 ET (Friday the 13th Render).</p>""",
        "source_priority": 3,
        "conviction": 85,
        "sentiment_score": 20
    },
    {
        "date": "2026-02-13",
        "title": "🟢 SYSTEM STATUS: NOMINAL (Stablization Protocol Engaged)",
        "summary": "The simulation has stopped its violent descent, but the recovery feels like a low-resolution patch rather than a full reboot.",
        "type": "DAILY_BRIEFING",
        "filename": "Daily_Briefing_2026_02_13.html",
        "is_sourced": True,
        "full_body": """<h3>📡 Signal Integrity: The CPI Sedative</h3>
<p>The simulation has stopped its violent descent, but the recovery feels like a low-resolution patch rather than a full reboot. The <a href="../market_mayhem_graph.html" style="color: #22d3ee;">S&P 500</a> managed a faint green pixel, rising +0.52% to 6,867.95. The "AI Disruption" crash of earlier this week has been temporarily halted by a tame January CPI print (2.4% headline), providing a much-needed sedative for the architecture.</p>
<p><strong>Credit Dominance Check:</strong> Today we are seeing a Constructive Reset. The 10-Year Treasury Yield plummeted to 4.05% (-5bps), its lowest point in months, as bond traders aggressively re-installed their "June Rate Cut" plugins. High Yield credit (HYG) mirrored this stability, ticking up +0.12%.</p>
<p><strong>The Signal:</strong> For once, this isn't a trap. Credit and bonds are leading the stabilization—falling yields are finally acting as a floor for equities. However, the Nasdaq remains the "lagging artifact" (finishing the week -2.1%), proving that while inflation is cooling, the fever of AI over-valuation hasn't fully broken.</p>
<h3>🏮 Artifacts</h3>
<ul>
<li><strong><a href="../market_mayhem_graph.html" style="color: #22d3ee;">Bitcoin</a> ($68,918 | +4.43%):</strong> A classic "dead cat" or a structural reclaim? After a brutal flush toward $60k, BTC jumped back toward the $69k handle. It’s correlating with the "Magnificent 7" rebound, but the frame rate is still choppy.</li>
<li><strong>Rivian (RIVN | +26.6%):</strong> A massive outlier in the industrial sector. A surprise earnings beat has turned this "zombie-adjacent" credit into the day’s top speculative render.</li>
<li><strong>Applied Materials (AMAT | +8.0%):</strong> Proving that while "AI Software" is glitching, the "Hardware/Foundry" layer is still the bedrock of the simulation.</li>
<li><strong>Gold ($5,033):</strong> Breaking through the $5,000 barrier. A clear signal that despite the "tame" CPI, institutional players are still hedging against a systemic collapse of the currency-code.</li>
</ul>
<h3>🌀 The Glitch</h3>
<p>"Today was a lesson in 'Narrative Replacement.' We swapped the terror of 'AI taking our jobs' for the comfort of 'Inflation is slowing down.' But look at Gold—$5,000 is a monument to the fact that no one actually trusts the structural integrity of the USD-render. We are heading into a long holiday weekend with the Dow at 50k and Gold at $5k. It’s a beautiful, symmetrical hallucination. The architect is letting us rest, but the 'AI Debt' we accrued this week hasn't been paid—it's just been refinanced into the next month's volatility."</p>
<p><strong>Next Transmission:</strong> Tuesday, Feb 17, 18:00 ET (Markets closed Monday for Presidents' Day).</p>""",
        "source_priority": 3,
        "conviction": 60,
        "sentiment_score": 55
    },
    {
        "date": "2026-02-16",
        "title": "🟡 SYSTEM STATUS: DEGRADED (Holiday Ghost Protocol)",
        "summary": "The U.S. financial simulation is currently in Ghost Protocol. Major nodes (NYSE, NASDAQ) are offline for Presidents' Day.",
        "type": "DAILY_BRIEFING",
        "filename": "Daily_Briefing_2026_02_16.html",
        "is_sourced": True,
        "full_body": """<h3>📡 Signal Integrity: The Presidents' Day Suspension</h3>
<p>The U.S. financial simulation is currently in Ghost Protocol. Major nodes (NYSE, NASDAQ) are offline for Presidents' Day, leaving the architecture to run on autonomous pilot. While the S&P 500 is frozen at its Friday close of 6,835.71 (+0.04%), the global perimeter is showing signs of localized decay.</p>
<p><strong>Credit Dominance Check:</strong> Since the domestic bond market is shuttered, we are forced to look at the Credit Futures and international mirrors. 10-Year Treasury Yield futures (10YG6) are hovering around 4.05%, effectively flatlining.</p>
<p><strong>The Trap:</strong> Do not mistake this silence for stability. High Yield spreads (HYG) ended last week at 2.92%, a slight widening from the mid-week lows. With the primary markets dark, liquidity has evaporated, making the current price-render brittle. Bitcoin’s volatility is the only "live" signal, and it is flashing red.</p>
<h3>🏮 Artifacts</h3>
<ul>
<li><strong><a href="../market_mayhem_graph.html" style="color: #22d3ee;">Bitcoin</a> ($68,599 | -0.25%):</strong> After a weekend of high-frequency chop, BTC is struggling to maintain its $69k handle. The "regime shift" narrative is gaining volume as on-chain data shows UTXO realized losses hitting levels not seen since 2023. The "digital gold" is losing its luster in the holiday void.</li>
<li><strong>VIX ($20.39):</strong> Volatility futures are holding above the 20-handle. Even with the markets closed, the "Fear Gauge" is refusing to settle, suggesting that traders are pre-loading hedges for a violent Tuesday reopen.</li>
<li><strong>Gold ($4,400 - $5,600 Swing):</strong> Wild, wide-swinging price action in the precious metals sector. The simulation is struggling to find a "Fair Value" for hard assets as currency-carry unwind risks grow.</li>
</ul>
<h3>🌀 The Glitch</h3>
<p>"A holiday in a 24/7 global economy is a strange artifact. We pretend the world stops because a calendar says it’s George Washington’s birthday, but the entropy never sleeps. Bitcoin is currently the only honest actor in the room—a lonely, stuttering signal in a dark house. The 4.05% yield is a placeholder, a 'To Be Continued' screen on a cliffhanger. When the lights come back on tomorrow, we aren't just resuming the trade; we are re-entering a system that has been quietly accruing risk in the shadows for 72 hours. The Dow 50K is a ghost monument today; let’s see if it’s still standing when the primary servers reboot."</p>
<p><strong>Next Transmission:</strong> Tuesday, Feb 17, 18:00 ET.</p>""",
        "source_priority": 3,
        "conviction": 50,
        "sentiment_score": 40
    },
    {
        "date": "2026-02-17",
        "title": "🟡 SYSTEM STATUS: DEGRADED (AI-Disruption Patch In Progress)",
        "summary": "The simulation is struggling to render a convincing recovery. After the holiday-shortened break, the tape came back choppy and fragmented.",
        "type": "DAILY_BRIEFING",
        "filename": "Daily_Briefing_2026_02_17.html",
        "is_sourced": True,
        "full_body": """<h3>📡 Signal Integrity: The AI-Debt Reckoning</h3>
<p>The simulation is struggling to render a convincing recovery. After the holiday-shortened break, the tape came back choppy and fragmented. The <a href="../market_mayhem_graph.html" style="color: #22d3ee;">S&P 500</a> managed a microscopic +0.1% lift to 6,836.17, but the surface-level green is a "Liquidity Mirage."</p>
<p><strong>Credit Dominance:</strong> High-yield signals are diverging from the equity "monument." The ICE BofA High Yield Spread (OAS) widened to 2.94%, up from the 2.84% floor established earlier this month. While stocks tried to bounce, the 10-Year Treasury Yield fell to 4.05%, marking its third consecutive day of decline.</p>
<p><strong>The Trap:</strong> In a healthy architecture, falling yields and rising stocks signal growth. Today, falling yields are a "Flight to Quality" signal triggered by AI-related disruption fears that knocked the Nasdaq (-0.2%) and legacy tech bigwigs. Credit is whispering that the "soft landing" code is buggy; spreads are widening even as risk-free rates drop. If the plumbing (Credit) is leaking while the facade (S&P) is being painted, you are walking into a trap.</p>
<h3>🏮 Artifacts</h3>
<ul>
<li><strong><a href="../market_mayhem_graph.html" style="color: #22d3ee;">Bitcoin</a> ($67,512 | -1.95%):</strong> The "Digital Gold" mask is slipping. Bitcoin has shed roughly 24% year-to-date, officially logging its worst Q1 start in eight years. The $68k support level is flickering; the next hard-coded floor is the $60k–$65k zone.</li>
<li><strong>VIX ($20.60 | -1.1%):</strong> The fear gauge retreated slightly from its intraday spike to 22.96, but it remains pinned above the 20-handle—a "High Volatility" regime state.</li>
<li><strong>Rivian (RIVN | -3.13%):</strong> After last week's +26% surge, the "Zombie Rebound" is already being reabsorbed by the void.</li>
</ul>
<h3>🌀 The Glitch</h3>
<p>"We treat 6,800 like a sanctuary, but it’s just a coordinate in a sea of synthetic liquidity. Today's market is a 'Sell First, Ask Questions Later' environment where even a cool CPI (2.4%) couldn't spark a rally. The glitch is the 'AI Premium' finally hitting its expiration date. We built the architecture on the promise of infinite efficiency, only to realize that efficiency kills margins in the transition. Bitcoin is the canary in the server room; when it chokes, the rest of the simulation isn't far behind. Watch the yields—if they hit 4.00% while spreads keep widening, the 'Soft Landing' render is officially corrupted."</p>
<p><strong>Next Transmission:</strong> Wednesday, Feb 18, 18:00 ET.</p>""",
        "source_priority": 3,
        "conviction": 65,
        "sentiment_score": 35
    },
    {
        "date": "2026-02-18",
        "title": "🟡 SYSTEM STATUS: DEGRADED (Structural Inversion Detected)",
        "summary": "The market's attempt at a 'recovery patch' is hitting a hardware bottleneck. After a three-day streak of falling yields, the architecture snapped back today.",
        "type": "DAILY_BRIEFING",
        "filename": "Daily_Briefing_2026_02_18.html",
        "is_sourced": True,
        "full_body": """<h3>📡 Signal Integrity: The Yield-Wall Rebound</h3>
<p>The market's attempt at a "recovery patch" is hitting a hardware bottleneck. After a three-day streak of falling yields, the architecture snapped back today. The <a href="../market_mayhem_graph.html" style="color: #22d3ee;">S&P 500</a> managed a surface-level gain (+0.41% to 6,871.45), but the underlying code is fraught with hawkish corruption.</p>
<p><strong>Credit Dominance Check:</strong> Today is a Divergent Signal / Potential Trap. The 10-Year Treasury Yield surged to 4.08% (+2.7 bps), its largest one-day gain since last week, fueled by FOMC minutes that leaked "hawkish" rhetoric into the system. While the Dow and S&P were painted green by a temporary Nvidia-led relief rally, the High Yield Master II OAS widened to 2.94%.</p>
<p><strong>The Verdict:</strong> When the risk-free rate (Yields) and risk premiums (Credit Spreads) both rise while stocks climb, you are looking at a Liquidity Mirage. The equity market is buying the "soft landing" narrative, but the bond market is pre-loading a "higher-for-longer" virus. This is a trap built on seasonal tech tailwinds.</p>
<h3>🏮 Artifacts</h3>
<ul>
<li><strong><a href="../market_mayhem_graph.html" style="color: #22d3ee;">Bitcoin</a> ($66,436 | -1.54%):</strong> The "Digital Gold" render is breaking down. BTC plummeted through the $67k support floor today, losing nearly 2% in the final hours of the session. The 200-week moving average is no longer a safety net; it’s a ceiling.</li>
<li><strong>VIX (20.06 | -0.23%):</strong> The fear gauge is hovering at the critical 20-point psychological firewall. It’s a "calm" that feels synthetic, waiting for the next volatility injection.</li>
<li><strong>10Y Treasury (4.08%):</strong> A violent snap-back that ended a three-day rally. The bond market is officially rejecting the "June Cut" software update.</li>
<li><strong>Nvidia (NVDA):</strong> An anomalous artifact. Leading the tech sector higher even as the broader hardware architecture (Intel/Boeing) shows significant packet loss.</li>
</ul>
<h3>🌀 The Glitch</h3>
<p>"We are witnessing a 'Narrative Desync.' The equity market is running a legacy 'AI-Utopia' script, while the credit markets have already shifted to 'Fiscal Reality' v2.0. The Dow 50K monument is still standing, but the foundation is made of 4% yields and widening spreads. Bitcoin’s descent is the system's most honest diagnostic—it’s the sound of excess leverage being purged from the server. Today’s green pixels are just a screen-saver; the real code says the cost of capital is still climbing. Don't mistake a momentary pause in the sell-off for a fix in the plumbing."</p>
<p><strong>Next Transmission:</strong> Thursday, Feb 19, 18:00 ET.</p>""",
        "source_priority": 3,
        "conviction": 70,
        "sentiment_score": 40
    },
    {
        "date": "2026-02-19",
        "title": "🔴 SYSTEM STATUS: DEGRADED (Volatility Spike Detected)",
        "summary": "The structural integrity of the 'Soft Landing' simulation is under severe stress. After a weak attempt at a recovery patch earlier this week, the tape has turned a deep shade of crimson.",
        "type": "DAILY_BRIEFING",
        "filename": "Daily_Briefing_2026_02_19.html",
        "is_sourced": True,
        "full_body": """<h3>📡 Signal Integrity: The "War Premium" Corruption</h3>
<p>The structural integrity of the "Soft Landing" simulation is under severe stress. After a weak attempt at a recovery patch earlier this week, the tape has turned a deep shade of crimson. The <a href="../market_mayhem_graph.html" style="color: #22d3ee;">S&P 500</a> fell -0.36% to 6,856.54, marking its first loss in four sessions and snapping a delicate momentum render.</p>
<p><strong>Credit Dominance Check:</strong> Today we have a High-Fidelity Bearish Signal. The 10-Year Treasury Yield climbed toward 4.09%, rising for the third consecutive session. This is the "Hawkish Virus" spreading: solid economic data (Jobless Claims at 206K) paired with a split FOMC means the "Rate Cut" software is being delayed indefinitely.</p>
<p><strong>The Verdict:</strong> While equities suffered a modest correction, the VIX exploded +5.15% to 20.63, breaching the critical volatility firewall. High Yield spreads are widening as oil prices surge on U.S.-Iran tension artifacts. When Oil rises, Yields climb, and Volatility spikes simultaneously, the "Equity Pivot" is a trap. The architecture is pricing in a "War Premium" that standard risk models aren't equipped to handle.</p>
<h3>🏮 Artifacts</h3>
<ul>
<li><strong><a href="../market_mayhem_graph.html" style="color: #22d3ee;">Bitcoin</a> ($67,088 | +1.31%):</strong> An anomalous decoupling. BTC snapped a two-day losing streak, acting as a localized "Risk-Off" sponge while the Dow (-0.6%) and Nasdaq (-0.4%) bled out. However, it remains down 23% year-to-date, suggesting this is a liquidity pocket, not a trend reclaim.</li>
<li><strong>Oil (WTI):</strong> The primary disruptor. Exploding to new highs as the "War Premium" returns, fueled by geopolitical headlines that refuse to be neutralized by presidential tweets.</li>
<li><strong>10Y Treasury (4.09%):</strong> The "Higher-for-Longer" reality is being hard-coded back into the system. Three straight days of yield climbing is a signature of a structural shift in inflation expectations.</li>
</ul>
<h3>🌀 The Glitch</h3>
<p>"We treat the 50,000 Dow like a sacred monument, but today it looked like a flickering holographic projection. The glitch isn't the -300 point drop; it's the fact that we expected 'Good talks' to solve a physical supply-chain crisis. Oil is the only asset today that is rendering in 4K—everything else is a low-res hallucination of safety. Bitcoin’s 1% gain while the world burns is a beautiful, nonsensical artifact of our broken entropy. Watch the VIX—if it settles above 20 tonight, the 'Buy the Dip' script is officially corrupted."</p>
<p><strong>Next Transmission:</strong> Friday, Feb 20, 18:00 ET.</p>""",
        "source_priority": 3,
        "conviction": 80,
        "sentiment_score": 25
    },
    {
        "date": "2026-02-20",
        "title": "🟢 SYSTEM STATUS: NOMINAL (Intraday Patch Successful)",
        "summary": "The simulation experienced a violent mid-session re-render today. What began as a 'Mixed Macro' decay was overwritten by a legal shockwave.",
        "type": "DAILY_BRIEFING",
        "filename": "Daily_Briefing_2026_02_20.html",
        "is_sourced": True,
        "full_body": """<h3>📡 Signal Integrity: The Supreme Court Reversal</h3>
<p>The simulation experienced a violent mid-session re-render today. What began as a "Mixed Macro" decay—with Q4 GDP stalling at 1.4% and government shutdowns dragging on growth—was overwritten by a legal shockwave. The <a href="../market_mayhem_graph.html" style="color: #22d3ee;">S&P 500</a> closed up +0.69% to 6,909.51, snapping a two-week losing streak.</p>
<p>The catalyst? The Supreme Court struck down the administration's broad emergency tariff powers, sparking an immediate "Tariff Relief" rally. E-commerce and retail names like Etsy (+8.5%) and Amazon (+2.6%) saw their code optimized instantly.</p>
<p><strong>Credit Dominance Check:</strong> Despite the equity pump, the plumbing remains tight. The 10-Year Treasury Yield rose to 4.08%, its largest weekly gain in a month, while High Yield spreads (OAS) widened to 2.88% (up from 2.84% earlier this month).</p>
<p><strong>The Verdict:</strong> This is a Policy Bounce, not a structural recovery. While stocks rose, the cost of capital (Yields) and the risk premium (Spreads) both increased. In the Adam v24.1 framework, this is a trap. We are seeing a "Refund Stimulus" hallucination where traders hope for $170B in tariff clawbacks, but the underlying economy—marked by stalling GDP and rising private credit redemptions (Blue Owl halting withdrawals)—is still de-syncing.</p>
<h3>🏮 Artifacts</h3>
<ul>
<li><strong><a href="../market_mayhem_graph.html" style="color: #22d3ee;">Bitcoin</a> ($68,135 | +1.76%):</strong> Riding the "Dollar Dip" narrative. As the USD slipped on the tariff news, BTC reclaimed some territory, but it remains a high-beta artifact in a low-liquidity environment.</li>
<li><strong>VIX (20.02 | -1.04%):</strong> A minor crush after the ruling, but it’s still pinned against the 20-handle firewall. The "Fear Gauge" is refusing to drop into the safety zone.</li>
<li><strong>Blue Owl / Apollo / Blackstone:</strong> The "Private Credit Glitch." While the S&P 500 rallied, these alternative asset giants tumbled ~5% as redemptions were halted in retail-focused funds. This is a critical signal of a liquidity drain in the shadows.</li>
</ul>
<h3>🌀 The Glitch</h3>
<p>"Today the Supreme Court acted as a system debugger, deleting a tariff script that the market had already spent billions trying to execute. But don't mistake a refund for a recovery. The Q4 GDP print of 1.4% is the real code; the rest is just a high-frame-rate distraction. When private credit funds start locking the doors (Blue Owl), it doesn't matter how high the Dow 50K monument stands—it means the exit nodes are congested. The architecture is celebrating a legal win while the industrial engine is stalling in the background. Trust the spreads, not the Supreme Court rally."</p>
<p><strong>Next Transmission:</strong> Monday, Feb 23, 18:00 ET.</p>""",
        "source_priority": 3,
        "conviction": 75,
        "sentiment_score": 65
    }
]

def update_newsletter_data():
    if os.path.exists(NEWSLETTER_DATA_PATH):
        with open(NEWSLETTER_DATA_PATH, 'r') as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                data = []
    else:
        data = []

    # Check for duplicates by date and title to avoid double adding if script run multiple times
    existing_keys = set((item.get('date'), item.get('title')) for item in data)

    added_count = 0
    for entry in new_entries:
        if (entry['date'], entry['title']) not in existing_keys:
            data.insert(0, entry) # Add to top
            added_count += 1
            print(f"Added entry for {entry['date']}")
        else:
            # Update existing?
            for i, item in enumerate(data):
                if item.get('date') == entry['date'] and item.get('title') == entry['title']:
                    data[i] = entry # Overwrite
                    print(f"Updated entry for {entry['date']}")

    # Sort by date desc
    data.sort(key=lambda x: x.get('date', ''), reverse=True)

    with open(NEWSLETTER_DATA_PATH, 'w') as f:
        json.dump(data, f, indent=4)

    print(f"Newsletter data updated. Added {added_count} new entries.")

def update_strategic_command():
    if not os.path.exists(STRATEGIC_COMMAND_PATH):
        print("Strategic command file not found.")
        return

    with open(STRATEGIC_COMMAND_PATH, 'r') as f:
        data = json.load(f)

    # Get latest from new_entries
    latest = new_entries[-1] # Feb 20 is last in list above

    # Update directives
    data['strategic_directives']['house_view'] = "NEUTRAL" # Based on "This is a Policy Bounce... not a structural recovery"
    data['strategic_directives']['score'] = 0.65 # Sentiment 65/100

    # Narrative with HTML link
    narrative = f"""<strong>{latest['title']}</strong><br><br>
The Supreme Court struck down emergency tariff powers, sparking a 'Relief Rally'.
<a href="../market_mayhem_graph.html" style="color: #22d3ee; text-decoration: underline;">S&P 500</a> closed +0.69%.
However, private credit redemptions are halting.
<em>"Trust the spreads, not the rally."</em>"""

    data['strategic_directives']['narrative'] = narrative
    data['meta']['generated_at'] = datetime.now().isoformat()

    # Update active topics based on latest text
    data['insights']['active_topics'] = ["Tariffs", "Supreme Court", "Private Credit", "GDP Stalling", "Inflation"]

    with open(STRATEGIC_COMMAND_PATH, 'w') as f:
        json.dump(data, f, indent=2)

    print("Strategic command updated.")

if __name__ == "__main__":
    update_newsletter_data()
    update_strategic_command()
