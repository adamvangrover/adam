import json
import os
import re
from datetime import datetime

# Raw data from the prompt
briefings_data = [
    {
        "date": "2026-02-12",
        "title": "🔴 SYSTEM STATUS: DEGRADED (Structural Integrity Failing)",
        "subtitle": "Signal Integrity: The Speculative Unwind",
        "content": """The simulation has hit a massive bottleneck. What began as a "too-hot" jobs report echo has mutated into a full-scale AI-sector hardware failure. The S&P 500 suffered its largest one-day percentage decline of the year, plummeting -1.57% to 6,832.76. The frame rate is officially stuttering.
Credit Dominance Check: Today we are seeing a Total Signal Inversion. In a healthy recovery, a stock-market dump triggers a "flight to safety" in bonds, driving yields down. While the 10-Year Treasury Yield did ease to 4.10% (-8bps), the move wasn't driven by stability, but by a panic pivot away from speculative positions.
The Signal: High Yield spreads (HYG/JNK) are beginning to widen significantly as investors question the viability of automation-heavy business models. When the Nasdaq drops -2% and Bitcoin breaks its structural support, the "Safe Haven" of Treasuries isn't a reward—it's an admission that the growth code is corrupted.""",
        "artifacts": [
            {"title": "Bitcoin ($66,297 | -3.25%)", "desc": "The \"Digital Gold\" mask has shattered. Bitcoin has now fallen four consecutive days, losing nearly 8% this week. The $60k support floor is no longer a theoretical boundary—it’s the next target for the margin-call bots."},
            {"title": "Nasdaq Composite (-2%)", "desc": "A fresh wave of \"AI Fears\" is slamming the tech architecture. Investors are suddenly questioning if the massive CapEx spend on hardware will ever render into real-world cash flow."},
            {"title": "VIX ($18.69)", "desc": "Volatility has gapped up. We are no longer in the \"seasonal drift\"; we are in a \"spike peak\" buy-signal zone, though it requires a 22-day hold to prove the floor is in."}
        ],
        "glitch": "\"The market just realized that you can't build a trillion-dollar economy on the promise of an algorithm that doesn't know how to do the laundry. Today's sell-off was the sound of a thousand 'AI-Optimism' plugins being uninstalled at once. We are watching a decoupling: the 10-Year yield is falling because the 'Risk-On' dream is dying, not because the economy is healthy. The glitch isn't the drop; the glitch was the belief that $70k Bitcoin and a $7k S&P could coexist with 4.2% yields. The system is finally running a self-diagnostic, and it doesn't like the results.\"",
        "sentiment": 20
    },
    {
        "date": "2026-02-13",
        "title": "🟢 SYSTEM STATUS: NOMINAL (Stablization Protocol Engaged)",
        "subtitle": "Signal Integrity: The CPI Sedative",
        "content": """The simulation has stopped its violent descent, but the recovery feels like a low-resolution patch rather than a full reboot. The S&P 500 managed a faint green pixel, rising +0.52% to 6,867.95. The "AI Disruption" crash of earlier this week has been temporarily halted by a tame January CPI print (2.4% headline), providing a much-needed sedative for the architecture.
Credit Dominance Check: Today we are seeing a Constructive Reset.
The 10-Year Treasury Yield plummeted to 4.05% (-5bps), its lowest point in months, as bond traders aggressively re-installed their "June Rate Cut" plugins. High Yield credit (HYG) mirrored this stability, ticking up +0.12%.
The Signal: For once, this isn't a trap. Credit and bonds are leading the stabilization—falling yields are finally acting as a floor for equities. However, the Nasdaq remains the "lagging artifact" (finishing the week -2.1%), proving that while inflation is cooling, the fever of AI over-valuation hasn't fully broken.""",
        "artifacts": [
            {"title": "Bitcoin ($68,918 | +4.43%)", "desc": "A classic \"dead cat\" or a structural reclaim? After a brutal flush toward $60k, BTC jumped back toward the $69k handle. It’s correlating with the \"Magnificent 7\" rebound, but the frame rate is still choppy."},
            {"title": "Rivian (RIVN | +26.6%)", "desc": "A massive outlier in the industrial sector. A surprise earnings beat has turned this \"zombie-adjacent\" credit into the day’s top speculative render."},
            {"title": "Applied Materials (AMAT | +8.0%)", "desc": "Proving that while \"AI Software\" is glitching, the \"Hardware/Foundry\" layer is still the bedrock of the simulation."},
            {"title": "Gold ($5,033)", "desc": "Breaking through the $5,000 barrier. A clear signal that despite the \"tame\" CPI, institutional players are still hedging against a systemic collapse of the currency-code."}
        ],
        "glitch": "\"Today was a lesson in 'Narrative Replacement.' We swapped the terror of 'AI taking our jobs' for the comfort of 'Inflation is slowing down.' But look at Gold—$5,000 is a monument to the fact that no one actually trusts the structural integrity of the USD-render. We are heading into a long holiday weekend with the Dow at 50k and Gold at $5k. It’s a beautiful, symmetrical hallucination. The architect is letting us rest, but the 'AI Debt' we accrued this week hasn't been paid—it's just been refinanced into the next month's volatility.\"",
        "sentiment": 60
    },
    {
        "date": "2026-02-16",
        "title": "🟡 SYSTEM STATUS: DEGRADED (Holiday Ghost Protocol)",
        "subtitle": "Signal Integrity: The Presidents' Day Suspension",
        "content": """The U.S. financial simulation is currently in Ghost Protocol. Major nodes (NYSE, NASDAQ) are offline for Presidents' Day, leaving the architecture to run on autonomous pilot. While the S&P 500 is frozen at its Friday close of 6,835.71 (+0.04%), the global perimeter is showing signs of localized decay.
Credit Dominance Check: Since the domestic bond market is shuttered, we are forced to look at the Credit Futures and international mirrors. 10-Year Treasury Yield futures (10YG6) are hovering around 4.05%, effectively flatlining.
The Trap: Do not mistake this silence for stability. High Yield spreads (HYG) ended last week at 2.92%, a slight widening from the mid-week lows. With the primary markets dark, liquidity has evaporated, making the current price-render brittle. Bitcoin’s volatility is the only "live" signal, and it is flashing red.""",
        "artifacts": [
            {"title": "Bitcoin ($68,599 | -0.25%)", "desc": "After a weekend of high-frequency chop, BTC is struggling to maintain its $69k handle. The \"regime shift\" narrative is gaining volume as on-chain data shows UTXO realized losses hitting levels not seen since 2023. The \"digital gold\" is losing its luster in the holiday void."},
            {"title": "VIX ($20.39)", "desc": "Volatility futures are holding above the 20-handle. Even with the markets closed, the \"Fear Gauge\" is refusing to settle, suggesting that traders are pre-loading hedges for a violent Tuesday reopen."},
            {"title": "Gold ($4,400 - $5,600 Swing)", "desc": "Wild, wide-swinging price action in the precious metals sector. The simulation is struggling to find a \"Fair Value\" for hard assets as currency-carry unwind risks grow."}
        ],
        "glitch": "\"A holiday in a 24/7 global economy is a strange artifact. We pretend the world stops because a calendar says it’s George Washington’s birthday, but the entropy never sleeps. Bitcoin is currently the only honest actor in the room—a lonely, stuttering signal in a dark house. The 4.05% yield is a placeholder, a 'To Be Continued' screen on a cliffhanger. When the lights come back on tomorrow, we aren't just resuming the trade; we are re-entering a system that has been quietly accruing risk in the shadows for 72 hours. The Dow 50K is a ghost monument today; let’s see if it’s still standing when the primary servers reboot.\"",
        "sentiment": 40
    },
    {
        "date": "2026-02-17",
        "title": "🟡 SYSTEM STATUS: DEGRADED (AI-Disruption Patch In Progress)",
        "subtitle": "Signal Integrity: The AI-Debt Reckoning",
        "content": """The simulation is struggling to render a convincing recovery. After the holiday-shortened break, the tape came back choppy and fragmented. The S&P 500 managed a microscopic +0.1% lift to 6,836.17, but the surface-level green is a "Liquidity Mirage."
Credit Dominance: High-yield signals are diverging from the equity "monument." The ICE BofA High Yield Spread (OAS) widened to 2.94%, up from the 2.84% floor established earlier this month. While stocks tried to bounce, the 10-Year Treasury Yield fell to 4.05%, marking its third consecutive day of decline.
The Trap: In a healthy architecture, falling yields and rising stocks signal growth. Today, falling yields are a "Flight to Quality" signal triggered by AI-related disruption fears that knocked the Nasdaq (-0.2%) and legacy tech bigwigs. Credit is whispering that the "soft landing" code is buggy; spreads are widening even as risk-free rates drop. If the plumbing (Credit) is leaking while the facade (S&P) is being painted, you are walking into a trap.""",
        "artifacts": [
            {"title": "Bitcoin ($67,512 | -1.95%)", "desc": "The \"Digital Gold\" mask is slipping. Bitcoin has shed roughly 24% year-to-date, officially logging its worst Q1 start in eight years. The $68k support level is flickering; the next hard-coded floor is the $60k–$65k zone."},
            {"title": "VIX ($20.60 | -1.1%)", "desc": "The fear gauge retreated slightly from its intraday spike to 22.96, but it remains pinned above the 20-handle—a \"High Volatility\" regime state."},
            {"title": "Rivian (RIVN | -3.13%)", "desc": "After last week's +26% surge, the \"Zombie Rebound\" is already being reabsorbed by the void."}
        ],
        "glitch": "\"We treat 6,800 like a sanctuary, but it’s just a coordinate in a sea of synthetic liquidity. Today's market is a 'Sell First, Ask Questions Later' environment where even a cool CPI (2.4%) couldn't spark a rally. The glitch is the 'AI Premium' finally hitting its expiration date. We built the architecture on the promise of infinite efficiency, only to realize that efficiency kills margins in the transition. Bitcoin is the canary in the server room; when it chokes, the rest of the simulation isn't far behind. Watch the yields—if they hit 4.00% while spreads keep widening, the 'Soft Landing' render is officially corrupted.\"",
        "sentiment": 30
    },
    {
        "date": "2026-02-18",
        "title": "🟡 SYSTEM STATUS: DEGRADED (Structural Inversion Detected)",
        "subtitle": "Signal Integrity: The Yield-Wall Rebound",
        "content": """The market's attempt at a "recovery patch" is hitting a hardware bottleneck. After a three-day streak of falling yields, the architecture snapped back today. The S&P 500 managed a surface-level gain (+0.41% to 6,871.45), but the underlying code is fraught with hawkish corruption.
Credit Dominance Check: Today is a Divergent Signal / Potential Trap. The 10-Year Treasury Yield surged to 4.08% (+2.7 bps), its largest one-day gain since last week, fueled by FOMC minutes that leaked "hawkish" rhetoric into the system. While the Dow and S&P were painted green by a temporary Nvidia-led relief rally, the High Yield Master II OAS widened to 2.94%.
The Verdict: When the risk-free rate (Yields) and risk premiums (Credit Spreads) both rise while stocks climb, you are looking at a Liquidity Mirage. The equity market is buying the "soft landing" narrative, but the bond market is pre-loading a "higher-for-longer" virus. This is a trap built on seasonal tech tailwinds.""",
        "artifacts": [
            {"title": "Bitcoin ($66,436 | -1.54%)", "desc": "The \"Digital Gold\" render is breaking down. BTC plummeted through the $67k support floor today, losing nearly 2% in the final hours of the session. The 200-week moving average is no longer a safety net; it’s a ceiling."},
            {"title": "VIX (20.06 | -0.23%)", "desc": "The fear gauge is hovering at the critical 20-point psychological firewall. It’s a \"calm\" that feels synthetic, waiting for the next volatility injection."},
            {"title": "10Y Treasury (4.08%)", "desc": "A violent snap-back that ended a three-day rally. The bond market is officially rejecting the \"June Cut\" software update."},
            {"title": "Nvidia (NVDA)", "desc": "An anomalous artifact. Leading the tech sector higher even as the broader hardware architecture (Intel/Boeing) shows significant packet loss."}
        ],
        "glitch": "\"We are witnessing a 'Narrative Desync.' The equity market is running a legacy 'AI-Utopia' script, while the credit markets have already shifted to 'Fiscal Reality' v2.0. The Dow 50K monument is still standing, but the foundation is made of 4% yields and widening spreads. Bitcoin’s descent is the system's most honest diagnostic—it’s the sound of excess leverage being purged from the server. Today’s green pixels are just a screen-saver; the real code says the cost of capital is still climbing. Don't mistake a momentary pause in the sell-off for a fix in the plumbing.\"",
        "sentiment": 35
    },
    {
        "date": "2026-02-19",
        "title": "🔴 SYSTEM STATUS: DEGRADED (Volatility Spike Detected)",
        "subtitle": "Signal Integrity: The \"War Premium\" Corruption",
        "content": """The structural integrity of the "Soft Landing" simulation is under severe stress. After a weak attempt at a recovery patch earlier this week, the tape has turned a deep shade of crimson. The S&P 500 fell -0.36% to 6,856.54, marking its first loss in four sessions and snapping a delicate momentum render.
Credit Dominance Check: Today we have a High-Fidelity Bearish Signal. The 10-Year Treasury Yield climbed toward 4.09%, rising for the third consecutive session. This is the "Hawkish Virus" spreading: solid economic data (Jobless Claims at 206K) paired with a split FOMC means the "Rate Cut" software is being delayed indefinitely.
The Verdict: While equities suffered a modest correction, the VIX exploded +5.15% to 20.63, breaching the critical volatility firewall. High Yield spreads are widening as oil prices surge on U.S.-Iran tension artifacts. When Oil rises, Yields climb, and Volatility spikes simultaneously, the "Equity Pivot" is a trap. The architecture is pricing in a "War Premium" that standard risk models aren't equipped to handle.""",
        "artifacts": [
            {"title": "Bitcoin ($67,088 | +1.31%)", "desc": "An anomalous decoupling. BTC snapped a two-day losing streak, acting as a localized \"Risk-Off\" sponge while the Dow (-0.6%) and Nasdaq (-0.4%) bled out. However, it remains down 23% year-to-date, suggesting this is a liquidity pocket, not a trend reclaim."},
            {"title": "Oil (WTI)", "desc": "The primary disruptor. Exploding to new highs as the \"War Premium\" returns, fueled by geopolitical headlines that refuse to be neutralized by presidential tweets."},
            {"title": "10Y Treasury (4.09%)", "desc": "The \"Higher-for-Longer\" reality is being hard-coded back into the system. Three straight days of yield climbing is a signature of a structural shift in inflation expectations."}
        ],
        "glitch": "\"We treat the 50,000 Dow like a sacred monument, but today it looked like a flickering holographic projection. The glitch isn't the -300 point drop; it's the fact that we expected 'Good talks' to solve a physical supply-chain crisis. Oil is the only asset today that is rendering in 4K—everything else is a low-res hallucination of safety. Bitcoin’s 1% gain while the world burns is a beautiful, nonsensical artifact of our broken entropy. Watch the VIX—if it settles above 20 tonight, the 'Buy the Dip' script is officially corrupted.\"",
        "sentiment": 25
    },
    {
        "date": "2026-02-20",
        "title": "🟢 SYSTEM STATUS: NOMINAL (Intraday Patch Successful)",
        "subtitle": "Signal Integrity: The Supreme Court Reversal",
        "content": """The simulation experienced a violent mid-session re-render today. What began as a "Mixed Macro" decay—with Q4 GDP stalling at 1.4% and government shutdowns dragging on growth—was overwritten by a legal shockwave. The S&P 500 closed up +0.69% to 6,909.51, snapping a two-week losing streak.
The catalyst? The Supreme Court struck down the administration's broad emergency tariff powers, sparking an immediate "Tariff Relief" rally. E-commerce and retail names like Etsy (+8.5%) and Amazon (+2.6%) saw their code optimized instantly.
Credit Dominance Check: Despite the equity pump, the plumbing remains tight. The 10-Year Treasury Yield rose to 4.08%, its largest weekly gain in a month, while High Yield spreads (OAS) widened to 2.88% (up from 2.84% earlier this month).
The Verdict: This is a Policy Bounce, not a structural recovery. While stocks rose, the cost of capital (Yields) and the risk premium (Spreads) both increased. In the Adam v24.1 framework, this is a trap. We are seeing a "Refund Stimulus" hallucination where traders hope for $170B in tariff clawbacks, but the underlying economy—marked by stalling GDP and rising private credit redemptions (Blue Owl halting withdrawals)—is still de-syncing.""",
        "artifacts": [
            {"title": "Bitcoin ($68,135 | +1.76%)", "desc": "Riding the \"Dollar Dip\" narrative. As the USD slipped on the tariff news, BTC reclaimed some territory, but it remains a high-beta artifact in a low-liquidity environment."},
            {"title": "VIX (20.02 | -1.04%)", "desc": "A minor crush after the ruling, but it’s still pinned against the 20-handle firewall. The \"Fear Gauge\" is refusing to drop into the safety zone."},
            {"title": "Blue Owl / Apollo / Blackstone", "desc": "The \"Private Credit Glitch.\" While the S&P 500 rallied, these alternative asset giants tumbled ~5% as redemptions were halted in retail-focused funds. This is a critical signal of a liquidity drain in the shadows."}
        ],
        "glitch": "\"Today the Supreme Court acted as a system debugger, deleting a tariff script that the market had already spent billions trying to execute. But don't mistake a refund for a recovery. The Q4 GDP print of 1.4% is the real code; the rest is just a high-frame-rate distraction. When private credit funds start locking the doors (Blue Owl), it doesn't matter how high the Dow 50K monument stands—it means the exit nodes are congested. The architecture is celebrating a legal win while the industrial engine is stalling in the background. Trust the spreads, not the Supreme Court rally.\"",
        "sentiment": 65
    },
    {
        "date": "2026-02-25",
        "title": "📡 Signal Integrity: The ADP Artifact",
        "subtitle": "Signal Integrity: The ADP Artifact",
        "content": """The simulation has regained its luster today, bolstered by an "ADP Artifact"—a robust private jobs report (+12,750 weekly avg.) that served as a stabilizer for the broader architecture. The S&P 500 climbed +0.81% to 6,890.07, while the tech-heavy Nasdaq (+1.1%) was invigorated by Anthropic’s "Enterprise-Grade" AI patch.
Credit Dominance Check: We are seeing Constructive Alignment. The 10-Year Treasury Yield rose to 4.05% (+2bps), snapping back from three-month lows as safe-haven demand evaporated in favor of the 10% tariff "relief" narrative. More importantly, the VIX plummeted -6.55% to 18.27, retreating from the 20-point firewall. High-yield credit spreads (HYG/JNK) tightened moderately, confirming the equity rally.
The Signal: This is not a trap. For the first time in several sessions, the plumbing (Credit) and the facade (Equities) are rendering the same scene: a "Risk-On" pivot driven by economic resilience and a manageable 10% tariff floor. However, with money markets scaling back Fed cut expectations to just a 50% chance in June, the architecture is now running on "Pure Growth" rather than "Rate-Cut Hope.\"""",
        "artifacts": [
            {"title": "Anthropic integrations", "desc": "The primary driver of today's tech-lift. While legacy software has been glitching, the market is betting on the next layer of the AI stack to justify current multiples."},
            {"title": "Bitcoin ($68,421 | +6.82%)", "desc": "A massive high-fidelity reversal. BTC reclaimed the $68k handle after being left for dead yesterday. It’s no longer a \"Digital Gold\" hedge—it's trading like a 3x leveraged version of the Nasdaq."},
            {"title": "SentinelOne (S | +1.25%)", "desc": "Outperforming the S&P 500. A sign that the \"Cybersecurity Debt\" we discussed is being aggressively refinanced by optimistic investors ahead of March earnings."}
        ],
        "glitch": "\"Today the market decided that 10% tariffs are 'the new normal' and that we can grow our way out of any structural debt. We celebrated the ADP jobs beat as if the consumer is immortal, but look at the VIX—even at 18.27, it’s still significantly higher than the 2025 baseline. The glitch is the 'Certainty Mirage.' We think we’ve solved the tariff problem because the SCOTUS stepped in, but the Fed just deleted your June rate-cut software in the background. We are building a new monument on the 50k Dow, but the foundation is shifting from 'Cheap Money' to 'AI-Efficiency.' Let’s hope the code doesn't have a bug.\"",
        "sentiment": 70
    }
]

# Paths
INDEX_FILE = "showcase/data/market_mayhem_index.json"
TEMPLATE_FILE = "showcase/daily_briefing.html"

# Load index
try:
    with open(INDEX_FILE, "r") as f:
        index_data = json.load(f)
except FileNotFoundError:
    index_data = []

# Load template
try:
    with open(TEMPLATE_FILE, "r") as f:
        template_content = f.read()
except FileNotFoundError:
    print(f"Error: {TEMPLATE_FILE} not found. Ensure it exists.")
    exit(1)

# Helper to generate HTML
def generate_html(briefing):
    html = template_content

    # Simple replacement logic (assuming template has specific markers or we replace sections)
    # Since we are using an existing file as template, we need to replace specific content blocks.
    # We will look for markers like the Title, Date, Content paragraphs, Artifacts, Glitch.

    # Title/Status
    html = re.sub(r'<span class="status-badge">.*?</span>', f'<span class="status-badge">{briefing["title"]}</span>', html, flags=re.DOTALL)

    # Date
    date_obj = datetime.strptime(briefing["date"], "%Y-%m-%d")
    date_str = date_obj.strftime("%b %d, %Y")
    html = re.sub(r'<span style="font-size: 1rem; color: #94a3b8; margin-left: auto;">.*?</span>', f'<span style="font-size: 1rem; color: #94a3b8; margin-left: auto;">{date_str}</span>', html, flags=re.DOTALL)

    # Content (We replace the Live Market Pulse paragraph and subsequent ones)
    # This is tricky with regex on a full file. We'll reconstruct the "container" div's first section.
    # Actually, simpler to just replace the specific text blocks if we can identify them, but they change.
    # Let's rebuild the body content roughly.

    # Strategy: Replace the entire first .section content after <h1>
    # We will locate the first <div class="section"> ... </div> and inside it replace paragraphs.

    # Construct new section 1 body
    # Extract subtitle if any
    subtitle = briefing.get("subtitle", "")

    # We assume the content provided is HTML-ready text (paragraphs).
    # We'll just wrap lines in <p> if they aren't.
    content_html = ""
    for para in briefing["content"].split("\n"):
        if para.strip():
            content_html += f"<p>{para.strip()}</p>\n"

    # Artifacts HTML
    artifacts_html = ""
    for art in briefing["artifacts"]:
        artifacts_html += f"""
    <div class="artifact-card">
     <div class="artifact-title">
      {art['title']}
     </div>
     <p>
      {art['desc']}
     </p>
    </div>"""

    # Glitch HTML
    glitch_html = f"<p>{briefing['glitch']}</p>"

    # Now we need to inject these into the HTML.
    # We'll use specific replacement targets based on the *current* file content which is the ADP one.

    # Replace content after H1 in first section
    # Find the first section
    section_match = re.search(r'(<div class="section">.*?<h1>.*?</h1>)(.*?)(</div>)', html, flags=re.DOTALL)
    if section_match:
        header = section_match.group(1) # <div class="section">...<h1>...</h1>
        # The rest of the section 1 is replaced
        new_section_1 = f"{header}\n{content_html}"
        html = html.replace(section_match.group(0), f"{new_section_1}</div>")

    # Replace Artifacts section
    # Look for <h2>🏮 Artifacts</h2> and replace the following divs until next section or end of div
    artifacts_match = re.search(r'(<h2>\s*🏮 Artifacts\s*</h2>)(.*?)(</div>\s*<div class="section">)', html, flags=re.DOTALL)
    if artifacts_match:
        html = html.replace(artifacts_match.group(2), f"\n{artifacts_html}\n")

    # Replace Glitch section
    glitch_match = re.search(r'(<h2 class="glitch-text">\s*🌀 The Glitch\s*</h2>)(.*?)(</div>)', html, flags=re.DOTALL)
    # Note: Glitch is usually in its own section.
    # Let's find the section containing "The Glitch"
    if glitch_match:
         html = html.replace(glitch_match.group(2), f"\n{glitch_html}\n")

    return html

# Process briefings
for briefing in briefings_data:
    filename = f"Daily_Briefing_{briefing['date'].replace('-', '_')}.html"
    filepath = f"showcase/{filename}"

    # Generate HTML
    full_html = generate_html(briefing)

    # Write HTML file
    with open(filepath, "w") as f:
        f.write(full_html)
    print(f"Generated {filepath}")

    # Update Index
    # Check if exists
    exists = False
    for entry in index_data:
        if entry.get("filename") == filename:
            exists = True
            # Update fields
            entry["title"] = briefing["subtitle"] if briefing.get("subtitle") else briefing["title"]
            entry["date"] = briefing["date"]
            entry["summary"] = briefing["content"][:150] + "..."
            entry["full_body"] = briefing["content"] # Store full text for search
            entry["sentiment_score"] = briefing["sentiment"]
            entry["type"] = "DAILY_BRIEFING"
            break

    if not exists:
        new_entry = {
            "title": briefing["subtitle"] if briefing.get("subtitle") else briefing["title"],
            "date": briefing["date"],
            "summary": briefing["content"][:150] + "...",
            "type": "DAILY_BRIEFING",
            "full_body": briefing["content"],
            "sentiment_score": briefing["sentiment"],
            "entities": {"keywords": ["AI", "Crypto", "Market", "Briefing"]}, # simplified
            "provenance_hash": "generated_archive",
            "filename": filename,
            "is_sourced": True,
            "metrics_json": "{}",
            "source_priority": 2,
            "conviction": 50
        }
        index_data.append(new_entry)

# Write index
with open(INDEX_FILE, "w") as f:
    json.dump(index_data, f, indent=2)

print("Archive complete.")
