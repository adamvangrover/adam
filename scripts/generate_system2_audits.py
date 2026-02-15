import os
import json
import hashlib
from datetime import datetime
import glob

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
        <div class="toolbar-item" onclick="showToast('EDIT MODE: READ ONLY (AUDIT)')">EDIT</div>
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
                <span id="verification-status" style="color: #ff00ff;">SYSTEM 2 // AUDIT PROTOCOL</span>
                <span>{prov_short}</span>
            </div>

            <div style="display:flex; justify-content:space-between; font-family:'JetBrains Mono'; font-size:0.8rem; color:#666; margin-bottom:20px; margin-top: 20px;">
                <span>{date}</span>
                <span>ID: {prov_short}</span>
            </div>

            <h1 class="title">{title}</h1>

            <!-- Metadata Panel for High Tier (Embedded in Doc) -->
            <div class="doc-metadata-panel" style="display: grid;">
                <div class="doc-metadata-item"><strong>Classification</strong>INTERNAL_ONLY</div>
                <div class="doc-metadata-item"><strong>Accuracy</strong>{accuracy}/100</div>
                <div class="doc-metadata-item"><strong>Bias</strong>{bias}/100</div>
                <div class="doc-metadata-item"><strong>Reviewer</strong>Overseer</div>
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
                    <span>> PREDICTION_ACCURACY</span>
                    <span>{accuracy}% (DELTA: -{bias})</span>
                </div>
                <div class="log-entry">
                    <span>> CRITIQUE_LOG</span>
                    <span>"{critique}"</span>
                </div>
            </div>

            <a href="#" class="source-jump-btn" onclick="toggleSource(); return false;" title="View Source JSON">JUMP TO SOURCE</a>

            <div style="margin-top: 40px; border-top: 1px solid #ccc; padding-top: 20px; font-style: italic; color: #666;">
                End of Audit.
            </div>
        </div>

        <aside class="cyber-sidebar">
            <div class="sidebar-widget">
                <div class="sidebar-title">Audit Metadata</div>
                <div class="stat-row"><span>DATE</span> <span class="stat-val">{date}</span></div>
                <div class="stat-row"><span>TYPE</span> <span class="stat-val">{type}</span></div>
                <div class="stat-row"><span>CLASS</span> <span class="stat-val">INTERNAL</span></div>
                <div class="stat-row"><span>PROVENANCE</span> <span class="stat-val" title="{prov_hash}">{prov_short}</span></div>
            </div>

            <div class="sidebar-widget">
                <div class="sidebar-title">System Accuracy</div>
                <div class="stat-row"><span>SCORE</span> <span class="stat-val">{accuracy}/100</span></div>
                <div class="sentiment-bar">
                    <div class="sentiment-fill" style="width: {accuracy}%; background: #00ff00;"></div>
                </div>
            </div>

            <div class="sidebar-widget">
                <div class="sidebar-title">Detected Entities</div>
                <div class="tag-cloud">
                    {entity_html}
                </div>
            </div>

             <div class="sidebar-widget">
                <div class="sidebar-title">Bias Detection</div>
                 <div class="stat-row"><span>OPTIMISM BIAS</span> <span class="stat-val">{bias}/100</span></div>
                 <div class="sentiment-bar">
                    <div class="sentiment-fill" style="width: {bias}%; background: #ff0000;"></div>
                </div>
            </div>

             <div class="sidebar-widget" style="border-color: #555;">
                <div class="sidebar-title">Audit Log</div>
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
                    type: 'bar',
                    data: {{
                        labels: labels,
                        datasets: [{{
                            label: labelName,
                            data: values,
                            backgroundColor: 'rgba(255, 0, 255, 0.5)',
                            borderColor: '#ff00ff',
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

def generate_audits():
    print("Generating System 2 Audits...")

    # 1. Load Index
    index_path = "showcase/data/market_mayhem_index.json"
    if not os.path.exists(index_path):
        print("Error: Index not found. Run archive generation first.")
        return

    with open(index_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 2. Filter for recent reports (2024-2026) to "audit"
    recent_reports = [d for d in data if d.get('date', '').startswith('2025') or d.get('date', '').startswith('2026')]
    recent_reports.sort(key=lambda x: x['date'], reverse=True)

    # Generate 3 Audits: Q3 2025, Q4 2025, Q1 2026
    audits = [
        {
            "period": "Q1 2026",
            "date": "2026-03-31",
            "focus": "Macro Strategy",
            "accuracy": 88,
            "bias": 12,
            "critique": "Model correctly anticipated the 'Reflationary Boom' but underestimated the persistence of services inflation (Sticky CPI > 3%). Sovereign AI thesis fully validated.",
            "metrics": {"Prediction_Accuracy": {"Jan": "85", "Feb": "88", "Mar": "92"}}
        },
        {
            "period": "Q4 2025",
            "date": "2025-12-31",
            "focus": "Tech Sector",
            "accuracy": 76,
            "bias": 24,
            "critique": "Optimism bias detected in semiconductor projections. Agent swarms overestimated demand elasticity for edge-AI devices. Correction in December was missed by momentum models.",
            "metrics": {"Prediction_Accuracy": {"Oct": "80", "Nov": "75", "Dec": "72"}}
        },
        {
            "period": "Q3 2025",
            "date": "2025-09-30",
            "focus": "Geopolitics",
            "accuracy": 94,
            "bias": 5,
            "critique": "High fidelity in geopolitical risk mapping. Energy volatility predicted within 2% margin. System 2 override correctly flagged 'Black Swan' signal in MENA region.",
            "metrics": {"Prediction_Accuracy": {"Jul": "92", "Aug": "95", "Sep": "94"}}
        }
    ]

    generated_count = 0
    for audit in audits:
        filename = f"System_2_Audit_{audit['period'].replace(' ', '_')}.html"
        filepath = os.path.join(OUTPUT_DIR, filename)

        full_body = f"""
        <h3>Executive Audit Summary: {audit['period']}</h3>
        <p>This report constitutes a meta-analysis of agentic outputs for the period {audit['period']}. System 2 Overseer protocols were engaged to evaluate predictive accuracy and detect latent biases in the generative models.</p>

        <h3>Performance Analysis</h3>
        <p>The aggregate predictive accuracy for this quarter was <strong>{audit['accuracy']}%</strong>. The primary driver of deviation was {audit['focus'].lower()} volatility.</p>

        <h3>Bias Detection</h3>
        <p>Latent optimism bias was detected at a level of <strong>{audit['bias']}%</strong>. Corrective weights have been applied to the 'Risk_Engine' and 'Sentiment_Crawler' agents for future iterations.</p>

        <h3>Key Learning</h3>
        <p>{audit['critique']}</p>
        """

        prov_hash = hashlib.sha256(full_body.encode()).hexdigest()

        # Entities
        entity_html = ""
        entities = ["System_2", "Overseer", "Audit", "Bias_Check", "Accuracy"]
        for e in entities:
             entity_html += f'<a href="market_mayhem_archive.html?search={e}" class="tag" style="text-decoration:none;">{e}</a>'

        conviction_color = "#00ff00"
        metrics_json = json.dumps(audit.get("metrics", {}))

        content = HTML_TEMPLATE.format(
            title=f"SYSTEM 2 AUDIT: {audit['period']}",
            date=audit['date'],
            summary=f"Meta-analysis of system performance for {audit['period']}. Accuracy: {audit['accuracy']}%.",
            type="SYSTEM_AUDIT",
            full_body=full_body,
            sentiment_score=50, # Neutral audit
            prov_hash=prov_hash,
            prov_short=prov_hash[:8],
            entity_html=entity_html,
            metrics_json=metrics_json,
            accuracy=audit['accuracy'],
            bias=audit['bias'],
            conviction=100,
            conviction_color=conviction_color,
            critique=audit['critique'],
            raw_source=json.dumps(audit, indent=2)
        )

        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)

        generated_count += 1
        print(f"Generated: {filename}")

    print(f"Complete. {generated_count} audits generated.")

if __name__ == "__main__":
    generate_audits()
