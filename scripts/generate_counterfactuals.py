import os
import json
import hashlib
from datetime import datetime

OUTPUT_DIR = "showcase"

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ADAM v23.5 :: {title}</title>
    <link rel="stylesheet" href="css/style.css">
    <link rel="stylesheet" href="css/market_mayhem_tiers.css">
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
    <link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Inter:wght@300;400;600;700&family=Playfair+Display:ital,wght@0,400;0,700;1,400&display=swap" rel="stylesheet">
    <script src="https://cdn.tailwindcss.com"></script>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <script src="js/nav.js" defer></script>
    <script src="js/market_mayhem_viewer.js" defer></script>
</head>
<body class="tier-high">

    <!-- System 2 Verification Overlay -->
    <div id="system2-overlay" class="system2-overlay">
        <div class="system2-terminal">
            <div id="sys2-content"></div>
        </div>
    </div>

    <!-- Toast Container -->
    <div id="toast-container" class="cyber-toast-container"></div>

    <!-- Raw Source Modal -->
    <div id="raw-source-modal" class="raw-source-modal">
        <div class="raw-source-header">
            <span>SOURCE_DATA.JSON</span>
            <span style="cursor:pointer; color:#cc0000;" onclick="toggleSource()">[CLOSE]</span>
        </div>
        <div class="raw-source-content" id="raw-source-content"></div>
    </div>

    <!-- Virtual Toolbar for High Tier -->
    <div class="doc-toolbar" style="display: flex;">
        <div class="toolbar-item" onclick="window.location.href='market_mayhem_archive.html'"><strong>FILE</strong></div>
        <div class="toolbar-item" onclick="showToast('EDIT MODE: READ ONLY (SIMULATION)')">EDIT</div>
        <div class="toolbar-item" onclick="toggleTheme()">VIEW</div>
        <div class="toolbar-item" onclick="showToast('INSERT: PERMISSION DENIED')">INSERT</div>
        <div class="toolbar-item" onclick="showToast('FORMAT: LOCKED')">FORMAT</div>
        <div class="toolbar-item" onclick="runVerification()" style="color: #0078d7;"><strong>VERIFY</strong></div>
        <div style="flex-grow:1;"></div>
        <div class="toolbar-item" style="color:#666;">ADAM_V26_EDITOR</div>
    </div>

    <div class="newsletter-wrapper">
        <div class="paper-sheet" id="paper-sheet">

            <!-- High Tier Header UI -->
            <div class="doc-header-ui" style="display: flex;">
                <span id="verification-status" style="color: #ffaa00;">SIMULATION // COUNTERFACTUAL</span>
                <span>{prov_short}</span>
            </div>

            <div style="display:flex; justify-content:space-between; font-family:'JetBrains Mono'; font-size:0.8rem; color:#666; margin-bottom:20px; margin-top: 20px;">
                <span>{date}</span>
                <span>ID: {prov_short}</span>
            </div>

            <h1 class="title">{title}</h1>

            <!-- Metadata Panel for High Tier (Embedded in Doc) -->
            <div class="doc-metadata-panel" style="display: grid;">
                <div class="doc-metadata-item"><strong>Classification</strong>RESTRICTED</div>
                <div class="doc-metadata-item"><strong>Confidence</strong>{conviction}/100</div>
                <div class="doc-metadata-item"><strong>Divergence</strong>{divergence}/100</div>
                <div class="doc-metadata-item"><strong>Simulator</strong>Nexus_Core</div>
            </div>

            <div id="financialChartContainer" style="display:none; margin-bottom: 30px;">
                <canvas id="financialChart"></canvas>
            </div>

            {full_body}

            <!-- Provenance Log (All Tiers) -->
            <div class="provenance-log">
                <div class="log-entry">
                    <span>> HASH_CHECK</span>
                    <span style="color:var(--terminal-green);">{prov_hash}</span>
                </div>
                <div class="log-entry">
                    <span>> DIVERGENCE_INDEX</span>
                    <span>{divergence}% (BASELINE: 2024_REALITY)</span>
                </div>
                <div class="log-entry">
                    <span>> SCENARIO_LOCK</span>
                    <span>{conviction}%</span>
                </div>
                <div class="log-entry">
                    <span>> CRITIQUE_LOG</span>
                    <span>"{critique}"</span>
                </div>
            </div>

            <a href="#" class="source-jump-btn" onclick="toggleSource(); return false;" title="View Source JSON">JUMP TO SOURCE</a>

            <div style="margin-top: 40px; border-top: 1px solid #ccc; padding-top: 20px; font-style: italic; color: #666;">
                End of Simulation.
            </div>
        </div>

        <aside class="cyber-sidebar">
            <div class="sidebar-widget">
                <div class="sidebar-title">Simulation Metadata</div>
                <div class="stat-row"><span>DATE</span> <span class="stat-val">{date}</span></div>
                <div class="stat-row"><span>TYPE</span> <span class="stat-val">{type}</span></div>
                <div class="stat-row"><span>CLASS</span> <span class="stat-val">RESTRICTED</span></div>
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
                <div class="sidebar-title">Simulation Confidence</div>
                 <div class="stat-row"><span>CONFIDENCE</span> <span class="stat-val">{conviction}/100</span></div>
                 <div class="sentiment-bar">
                    <div class="sentiment-fill" style="width: {conviction}%; background: {conviction_color};"></div>
                </div>
            </div>

             <div class="sidebar-widget" style="border-color: #555;">
                <div class="sidebar-title">Nexus Agent Log</div>
                 <div style="font-size: 0.75rem; color: #ccc; font-style: italic;">
                    "{critique}"
                </div>
            </div>
        </aside>
    </div>

    <!-- Hidden Raw Data -->
    <textarea id="hidden-raw-data" style="display:none;">{raw_source}</textarea>

    <script>
        const metricsData = {metrics_json};
        if (metricsData && Object.keys(metricsData).length > 0) {{
            const ctx = document.getElementById('financialChart');
            const container = document.getElementById('financialChartContainer');
            let labels = [];
            let values = [];
            let labelName = "Metric";
            for (const [key, val] of Object.entries(metricsData)) {{
                if (typeof val === 'object') {{
                    labels = Object.keys(val);
                    values = Object.values(val).map(v => parseFloat(v));
                    labelName = key;
                    break;
                }}
            }}
            if (labels.length > 0) {{
                container.style.display = 'block';
                new Chart(ctx, {{
                    type: 'line',
                    data: {{
                        labels: labels,
                        datasets: [{{
                            label: labelName,
                            data: values,
                            borderColor: '#ffaa00',
                            backgroundColor: 'rgba(255, 170, 0, 0.1)',
                            borderWidth: 2,
                            tension: 0.4,
                            fill: true
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

SCENARIOS = [
    {
        "title": "COUNTERFACTUAL: THE 2008 LEHMAN BAILOUT",
        "date": "2008-09-15",
        "summary": "Simulation: What if the US Treasury had backstopped Lehman Brothers? The moral hazard debate vs. systemic collapse.",
        "full_body": """
        <h3>Scenario Parameters: Intervention</h3>
        <p><strong>Divergence Point:</strong> September 14, 2008. Treasury Secretary Paulson agrees to a $30 billion backstop for Barclays to acquire Lehman Brothers, citing "systemic risk" over "moral hazard".</p>

        <h3>Outcome: The 'Soft' Landing?</h3>
        <p>Markets rallied 4% on the news. The Reserve Primary Fund did not break the buck. The commercial paper market remained liquid. However, the underlying toxic assets (subprime mortgages) remained on balance sheets across the street.</p>

        <h3>Long-Term Consequence: The Zombie Economy</h3>
        <p>Without the cleansing fire of a crisis, strict banking regulations (Dodd-Frank equivalent) were never passed. Leverage ratios remained high. By 2011, a larger, more systemic sovereign debt crisis emerged, centering on US Treasuries themselves rather than just mortgage-backed securities.</p>

        <h3>Metrics: S&P 500 Simulation</h3>
        <p>The index avoided the 2008 crash low of 666, bottoming instead at 1,100 in 2009. However, growth remained anemia (1.2% GDP) for a decade due to debt overhang.</p>
        """,
        "metrics": {"S&P_500_Simulated": {"2008": "1250", "2009": "1100", "2010": "1150", "2011": "1050", "2012": "1200"}},
        "entities": ["$LEH", "$XLF", "US_FED", "Moral_Hazard", "Bailout"],
        "sentiment": 45,
        "divergence": 85,
        "conviction": 92
    },
    {
        "title": "COUNTERFACTUAL: THE 2000 SOFT LANDING",
        "date": "2000-03-10",
        "summary": "Simulation: What if the Fed had not tightened rates in 1999-2000? The Dot Com bubble deflates slowly rather than popping.",
        "full_body": """
        <h3>Scenario Parameters: Rate Stability</h3>
        <p><strong>Divergence Point:</strong> February 2000. Greenspan announces a pause in rate hikes, citing productivity gains from the internet revolution.</p>

        <h3>Outcome: The Plateau</h3>
        <p>The NASDAQ Composite peaked at 5,048 but retreated orderly to 4,000. Capital continued to flow into fiber optics and telecom. Companies like Pets.com still failed, but Amazon and Cisco consolidated power faster.</p>

        <h3>Long-Term Consequence: Faster Digitization</h3>
        <p>With capital markets still open, broadband infrastructure was built out 3 years ahead of schedule. Streaming video and cloud computing emerged in 2003 rather than 2006. The Great Financial Crisis of 2008 was potentially exacerbated by an even larger tech-driven wealth effect masking housing issues.</p>
        """,
        "metrics": {"NASDAQ_Simulated": {"2000": "4500", "2001": "4200", "2002": "3800", "2003": "4100"}},
        "entities": ["$NDX", "$AMZN", "$CSCO", "US_FED", "Tech_Bubble"],
        "sentiment": 60,
        "divergence": 70,
        "conviction": 85
    },
    {
        "title": "COUNTERFACTUAL: AL GORE WINS 2000",
        "date": "2001-01-20",
        "summary": "Simulation: What if the Supreme Court halted the recount in favor of Gore? The Green Energy transition begins two decades early.",
        "full_body": """
        <h3>Scenario Parameters: Policy Shift</h3>
        <p><strong>Divergence Point:</strong> December 12, 2000. SCOTUS rules 5-4 to allow Florida recount to proceed. Gore wins by 537 votes.</p>

        <h3>Outcome: The Carbon Tax</h3>
        <p>The Gore administration passes a Carbon Tax in 2002. Oil majors ($XOM, $CVX) crash while early solar and wind plays skyrocket. The US never invades Iraq in 2003, focusing military spending on "Energy Independence" via renewables.</p>

        <h3>Long-Term Consequence: Geopolitical Realignment</h3>
        <p>Oil prices remain suppressed. The Middle East influence wanes earlier. Russia's economy struggles in the 2000s, potentially altering the trajectory of Putin's consolidation of power. The 2008 crisis still occurs but is driven by Green Tech speculation rather than housing.</p>
        """,
        "metrics": {"Oil_Price_Simulated": {"2001": "25", "2002": "30", "2003": "28", "2004": "35"}},
        "entities": ["$XOM", "$TAN", "US_GOV", "Climate_Change", "Geopolitics"],
        "sentiment": 55,
        "divergence": 95,
        "conviction": 88
    },
    {
        "title": "COUNTERFACTUAL: BITCOIN BANNED 2013",
        "date": "2013-10-02",
        "summary": "Simulation: What if the Silk Road seizure led to a coordinated G20 ban on cryptocurrency ownership and mining?",
        "full_body": """
        <h3>Scenario Parameters: The G20 Ban</h3>
        <p><strong>Divergence Point:</strong> October 2013. Following the Silk Road bust, the US, EU, and China announce a simultaneous ban on all crypto-assets, labeling them "Tools of Terror".</p>

        <h3>Outcome: The Dark Age</h3>
        <p>Bitcoin price collapses from $100 to $5. Development goes strictly underground (Tor-only). Ethereum never launches in 2015 as VC funding evaporates.</p>

        <h3>Long-Term Consequence: CBDC Dominance</h3>
        <p>Without private crypto innovation, Central Bank Digital Currencies (CBDCs) are rolled out unchallenged by 2018. Financial privacy is effectively eliminated. Blockchain technology is used solely for enterprise permissioned ledgers (IBM, Maersk).</p>
        """,
        "metrics": {"BTC_Price_Simulated": {"2013": "120", "2014": "10", "2015": "5", "2016": "8"}},
        "entities": ["$BTC", "CBDC", "Regulation", "Privacy", "Crypto"],
        "sentiment": 20,
        "divergence": 90,
        "conviction": 95
    },
    {
        "title": "COUNTERFACTUAL: THE 2024 AI WINTER",
        "date": "2024-06-15",
        "summary": "Simulation: What if LLM scaling laws hit a hard wall in mid-2024? The trillion-dollar Capex spend evaporates.",
        "full_body": """
        <h3>Scenario Parameters: The Plateau</h3>
        <p><strong>Divergence Point:</strong> June 2024. GPT-5 training runs fail to show significant improvement over GPT-4. Researchers conclude that "transformer architecture has peaked".</p>

        <h3>Outcome: The Nvidia Crash</h3>
        <p>Nvidia ($NVDA) stock crashes 60% in a month as hyperscalers (Microsoft, Google) cancel GPU orders. The "AI Safety" debate vanishes, replaced by "AI Disappointment".</p>

        <h3>Long-Term Consequence: Tech Depression</h3>
        <p>The Nasdaq enters a prolonged bear market. Venture capital pivots back to... nothing. There is no "Next Big Thing". The US economy, deprived of the productivity narrative, enters a standard recession in late 2024.</p>
        """,
        "metrics": {"NVDA_Price_Simulated": {"2024_Q1": "900", "2024_Q2": "850", "2024_Q3": "350", "2024_Q4": "300"}},
        "entities": ["$NVDA", "$MSFT", "AI_Winter", "Recession", "Tech"],
        "sentiment": 30,
        "divergence": 60,
        "conviction": 80
    }
]

def generate_counterfactuals():
    print("Generating Counterfactual Scenarios...")

    generated_count = 0
    for scenario in SCENARIOS:
        filename = f"Counterfactual_{scenario['title'].replace(' ', '_').replace(':', '')}.html"
        filepath = os.path.join(OUTPUT_DIR, filename)

        # Provenance Mock
        prov_hash = hashlib.sha256(scenario['full_body'].encode()).hexdigest()

        # Entity HTML
        entity_html = ""
        for e in scenario['entities']:
            if e.startswith("$"):
                entity_html += f'<a href="market_mayhem_archive.html?search={e[1:]}" class="tag ticker" style="text-decoration:none;">{e}</a>'
            else:
                entity_html += f'<a href="market_mayhem_archive.html?search={e}" class="tag" style="text-decoration:none;">{e}</a>'

        metrics_json = json.dumps(scenario.get("metrics", {}))

        conviction_color = "#00ff00" if scenario['conviction'] > 70 else "#ffff00"

        content = HTML_TEMPLATE.format(
            title=scenario['title'],
            date=scenario['date'],
            summary=scenario['summary'],
            type="COUNTERFACTUAL",
            full_body=scenario['full_body'],
            sentiment_score=scenario['sentiment'],
            prov_hash=prov_hash,
            prov_short=prov_hash[:8],
            entity_html=entity_html,
            metrics_json=metrics_json,
            conviction=scenario['conviction'],
            conviction_color=conviction_color,
            divergence=scenario['divergence'],
            critique=f"Nexus Simulation Engine verified divergence of {scenario['divergence']}%. Scenario Plausible.",
            raw_source=json.dumps(scenario, indent=2)
        )

        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)

        generated_count += 1
        print(f"Generated: {filename}")

    print(f"Complete. {generated_count} scenarios generated.")

if __name__ == "__main__":
    generate_counterfactuals()
