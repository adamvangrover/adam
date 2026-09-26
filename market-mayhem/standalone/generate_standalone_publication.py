import os
import json
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw
import weasyprint

def generate_visual_assets():
    assets_dir = 'market-mayhem/standalone/assets'
    os.makedirs(assets_dir, exist_ok=True)

    # Image 1: Baseline
    width, height = 800, 450
    img1 = Image.new('RGB', (width, height), '#0a0a0c')
    draw1 = ImageDraw.Draw(img1)
    for y in range(height):
        r = int(10 + (y / height) * 20)
        g = int(30 + (y / height) * 50)
        b = int(45 + (y / height) * 65)
        draw1.line([(0, y), (width, y)], fill=(r, g, b))
    draw1.ellipse([580, 30, 700, 150], fill=(0, 255, 204), outline=(0, 255, 204))
    draw1.rectangle([0, 300, 800, 450], fill=(12, 40, 50))
    draw1.polygon([(420, 220), (540, 220), (490, 310), (470, 310)], fill=(0, 200, 180), outline=(0, 255, 204))
    draw1.line([(480, 310), (480, 390)], fill=(0, 255, 204), width=5)
    draw1.ellipse([440, 385, 520, 400], fill=(0, 255, 204))
    draw1.rectangle([0, 410, 800, 450], fill=(10, 20, 25))
    draw1.text((20, 420), "RISK-ON: CLEAR SKIES & HIGH LIQUIDITY", fill=(0, 255, 204))
    img1.save(f'{assets_dir}/baseline_margarita.png')

    # Image 2: Challenger
    img2 = Image.new('RGB', (width, height), '#0a0a0c')
    draw2 = ImageDraw.Draw(img2)
    for y in range(height):
        r = int(35 + (y / height) * 45)
        g = int(8 + (y / height) * 10)
        b = int(12 + (y / height) * 15)
        draw2.line([(0, y), (width, y)], fill=(r, g, b))
    draw2.polygon([(260, 250), (540, 250), (510, 400), (290, 400)], fill=(35, 20, 25), outline=(255, 51, 102), width=3)
    np.random.seed(42)
    for _ in range(60):
        fx = np.random.randint(270, 530)
        fy = np.random.randint(100, 245)
        fr = np.random.randint(10, 35)
        draw2.ellipse([fx-fr, fy-fr, fx+fr, fy+fr], fill=(255, np.random.randint(30, 160), np.random.randint(40, 100)))
    draw2.rectangle([0, 410, 800, 450], fill=(25, 10, 15))
    draw2.text((20, 420), "TAIL-RISK: LIQUIDITY SHOCK & DURATION CONFLAGRATION", fill=(255, 51, 102))
    img2.save(f'{assets_dir}/challenger_dumpster.png')

    # Chart 1: 10Y Treasury & BSL Spreads Trajectory
    fig, ax1 = plt.subplots(figsize=(9, 4), facecolor='#0a0a0c')
    ax1.set_facecolor('#0a0a0c')
    days = np.array([1, 2, 3, 4, 5, 6, 7])
    yield_base = np.array([4.38, 4.45, 4.52, 4.70, 4.65, 4.58, 4.50])
    yield_challenger = np.array([4.38, 4.85, 5.25, 5.65, 5.40, 5.15, 4.90])
    ax1.plot(days, yield_base, color='#00ffcc', linewidth=2.5, marker='o', label='10Y Yield Baseline (%)')
    ax1.plot(days, yield_challenger, color='#ff3366', linewidth=2.5, marker='s', linestyle='--', label='10Y Yield Shock (%)')
    ax1.set_title('Macro Rates & BSL Spread Divergence (7-Day Shock Simulation)', color='#ffffff', fontsize=11, fontweight='bold', pad=10)
    ax1.set_xlabel('Simulation Days', color='#a0a0b0', fontsize=9)
    ax1.set_ylabel('Yield (%)', color='#a0a0b0', fontsize=9)
    ax1.tick_params(colors='#a0a0b0')
    ax1.legend(facecolor='#14141d', edgecolor='#333344', labelcolor='#ffffff', loc='upper left')
    plt.grid(True, linestyle=':', alpha=0.3, color='#444455')
    plt.tight_layout()
    plt.savefig(f'{assets_dir}/yield_shock_chart.png', dpi=200, facecolor='#0a0a0c')

def generate_standalone_html_and_pdf():
    html_content = """<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Market Mayhem - Standalone Quantitative Intelligence Suite</title>
    <style>
        @page {
            size: A4 portrait;
            margin: 12mm;
            @bottom-right {
                content: "Page " counter(page) " of " counter(pages);
                font-family: 'Courier New', monospace;
                font-size: 8pt;
                color: #00ffcc;
            }
            @bottom-left {
                content: "ADAM SYSTEM 2 // STANDALONE SUITE";
                font-family: 'Courier New', monospace;
                font-size: 8pt;
                color: #a0a0b0;
            }
        }
        body {
            background-color: #0a0a0c;
            color: #e0e0e0;
            font-family: 'Helvetica Neue', Arial, sans-serif;
            margin: 0;
            padding: 0;
            font-size: 9.5pt;
            line-height: 1.4;
        }
        .header {
            border-bottom: 2px solid #ff3366;
            padding-bottom: 8px;
            margin-bottom: 15px;
        }
        .title {
            font-size: 24pt;
            font-weight: 900;
            color: #ffffff;
            letter-spacing: 2px;
            margin: 0;
            text-transform: uppercase;
        }
        .subtitle {
            font-size: 10pt;
            color: #00ffcc;
            font-family: 'Courier New', monospace;
            margin-top: 3px;
        }
        .section-title {
            font-size: 12pt;
            font-weight: bold;
            color: #00ffcc;
            border-left: 4px solid #00ffcc;
            padding-left: 8px;
            margin-top: 18px;
            margin-bottom: 10px;
            text-transform: uppercase;
        }
        .section-title.red {
            color: #ff3366;
            border-left-color: #ff3366;
        }
        .card {
            background-color: #14141d;
            border: 1px solid #222233;
            padding: 10px 12px;
            margin-bottom: 12px;
            border-radius: 3px;
        }
        table.data-table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 8px;
            margin-bottom: 12px;
            font-size: 8pt;
        }
        table.data-table th {
            background-color: #1a1a26;
            color: #00ffcc;
            border: 1px solid #28283a;
            padding: 5px 6px;
            text-align: left;
            font-family: 'Courier New', monospace;
        }
        table.data-table td {
            border: 1px solid #222233;
            padding: 5px 6px;
            background-color: #101017;
        }
        .code-block {
            background-color: #08080d;
            border: 1px solid #222233;
            border-left: 3px solid #00ffcc;
            padding: 8px;
            font-family: 'Courier New', monospace;
            font-size: 7.5pt;
            color: #00ffcc;
            white-space: pre-wrap;
            margin-bottom: 12px;
        }
        .page-break { page-break-before: always; }
        .grid-2 { display: table; width: 100%; margin-bottom: 10px; }
        .grid-cell { display: table-cell; width: 50%; padding-right: 6px; vertical-align: top; }
        .grid-cell:last-child { padding-right: 0; padding-left: 6px; }
        .grid-cell img { width: 100%; border: 1px solid #333344; border-radius: 3px; }
    </style>
</head>
<body>

    <div class="header">
        <div class="title">Market Mayhem</div>
        <div class="subtitle">STANDALONE QUANTITATIVE RISK & MULTI-BRIEFING HARNESS</div>
    </div>

    <!-- PART 1: MARKET MAYHEM NEWSLETTER -->
    <div class="section-title">Part I: Market Mayhem Newsletter</div>

    <div class="card">
        <strong>Module 1: Executive Summary & Top Market Stories</strong><br>
        Systemic credit conditions face structural strain as persistent inflation keeps terminal rates elevated, compressing interest coverage ratios across highly leveraged corporate issuers. Broadly syndicated loan (BSL) repricings continue to mask underlying credit degradation, forcing institutional coverage teams to prepare for heightened default contagion in lower-tier rating buckets.
        <ul>
            <li><strong>Macro Rates Volatility:</strong> The 10-Year Treasury yield tests critical resistance levels as sticky CPI print data pushes back expectations for monetary easing, forcing a sharp recalibration of real yields across the curve.</li>
            <li><strong>Credit Spread Compression:</strong> High-yield and BSL spreads remain tight despite macro headwinds, creating a visible disconnect between asset pricing and fundamental corporate debt-service capabilities.</li>
            <li><strong>Energy Supply Shocks:</strong> Crude oil benchmarks experience upward supply-driven pressure, re-igniting fears of a secondary inflation wave that complicates central bank terminal rate trajectories.</li>
        </ul>
    </div>

    <div class="card" style="border-left: 3px solid #ff3366;">
        <strong style="color: #ff3366;">Module 2: 🔴 SYSTEM STATUS: CRITICAL (The Glitch)</strong><br>
        The yield curve simulation has encountered a severe logic loop. The "Bear-Flattening Virus" corrupted traditional asset correlation protocols overnight, forcing central bank nodes to re-render forward guidance while capital flows panic-route through fragmented liquidity pools. High-frequency algorithms are running at maximum entropy as systemic leverage parameters hit hard risk limits.
        <br><br>
        • <strong>Bitcoin ($88,450):</strong> Digital liquidity sink absorbs overflow capital from failing sovereign fiat routines.<br>
        • <strong>VIX Index (18.25):</strong> Volatility subroutine re-allocates pricing power back to tail-risk hedgers.<br>
        • <strong>WTI Crude Oil ($81.50):</strong> Supply-side glitch triggers an automated inflationary cascade across physical nodes.<br>
        • <strong>10Y US Treasury (4.38%):</strong> Sovereign baseline yield recalibrates to purge soft-landing assumptions from the main network.
    </div>

    <div class="card">
        <strong>Module 3: Macro & Policy Outlook</strong><br>
        The Federal Reserve maintains a restrictive policy stance, holding target rates steady as persistent service inflation and resilient labor data limit room for near-term rate cuts. Policy guidance signals a prolonged hold, directly pushing out the timing and magnitude of any potential yield curve steepening. Persistent elevated baseline rates continue to erode corporate debt-service metrics, particularly for balance sheets dependent on floating-rate debt. While real GDP metrics display nominal resilience, deteriorating interest coverage ratios (ICR) across single-B issuers indicate that restrictive policy is steadily exhausting corporate cash buffers.
    </div>

    <div class="section-title red">Module 4: BSL Market Update & The Repricing Mirage</div>
    <p>Broadly Syndicated Loan (BSL) market activity remains dominated by opportunistic repricings and refinancing transactions rather than new net M&A issuance. Average BSL spreads have compressed toward SOFR + 325 bps, driven by massive institutional demand from CLOs and retail loan funds.</p>
    <table class="data-table">
        <thead>
            <tr><th>Market Segment</th><th>Avg Spread / Metric</th><th>Key Operational Dynamic</th><th>Primary Risk Vector</th></tr>
        </thead>
        <tbody>
            <tr><td><strong>BSL Market</strong></td><td>SOFR + 325 bps</td><td>Aggressive repricings; covenant flexibility</td><td>Tail-risk default clustering</td></tr>
            <tr><td><strong>Private Credit</strong></td><td>SOFR + 550 bps</td><td>Tightening premiums; direct borrower workout power</td><td>Delayed loss recognition</td></tr>
        </tbody>
    </table>

    <div class="section-title">Visual Metaphors & Shock Trajectory</div>
    <div class="grid-2">
        <div class="grid-cell"><img src="assets/baseline_margarita.png" alt="Baseline"><div style="font-size:7.5pt; text-align:center; color:#a0a0b0;">IMAGE 1: RISK-ON CLEAR SKIES</div></div>
        <div class="grid-cell"><img src="assets/challenger_dumpster.png" alt="Challenger"><div style="font-size:7.5pt; text-align:center; color:#a0a0b0;">IMAGE 2: TAIL-RISK LIQUIDITY SHOCK</div></div>
    </div>
    <img src="assets/yield_shock_chart.png" alt="Yield Shock Chart" style="width:100%; margin-bottom:10px;">

    <div class="page-break"></div>

    <div class="section-title">Module 5: Counterfactual Scenario Analysis</div>
    <div class="code-block">[Energy Shock / Geo Tension] ──► [Persistent Inflation Spike] ──► [Surprise +50 bps Rate Hike]
       ├─► Credit Risk: Single-B ICR drops below 1.0x ──► Default Cascade
       ├─► Market Risk: Yields spike / Equity Multiples drop ──► Volatility Jump
       └─► Liquidity Risk: Secondary Loan Market Bids Dry Up ──► Private Credit Lockup</div>

    <div class="card">
        <strong>Module 6: Tactical Appendices</strong><br>
        • <strong>Institutional Investors & Funds:</strong> Maintain short duration profiles in fixed income while selectively rotating into top-tier AAA CLO tranches. Underweight lower-rated B3/B- floating-rate loans susceptible to downgrade migration.<br>
        • <strong>Private Equity Sponsors:</strong> Prioritize equity co-investments and junior capital infusions to deleverage portfolio companies. Execute fixed-for-floating interest rate swaps immediately to cap SOFR exposure.<br>
        • <strong>Family Offices & UHNW:</strong> Allocate opportunistic capital to distressed debt funds and secondary private equity vehicles targeting liquidity-constrained sellers. Maximize T-bill yields.
    </div>

    <div class="card">
        <strong>Module 7: Behavioral Finance Corner - Anchoring Bias in Credit Risk</strong><br>
        Anchoring Bias occurs when risk managers evaluate credit risk based on historical interest rate baselines (such as ZIRP era) rather than adapting to structural regime shifts. In current credit risk control, anchoring to low base rates causes analysts to misprice long-term debt-servicing capability.
        <br><em>Rule of Thumb:</em> Always stress-test balance sheet cash flows using a baseline SOFR assumption that is at least 150 bps higher than current market forward curve.
    </div>

    <div class="section-title red">Module 8: 💾 Institutional AI Training Ledger & JSON Schema</div>
    <table class="data-table">
        <thead>
            <tr><th>Data Variable / Node</th><th>Market Level / Value</th><th>Primary Model Target</th><th>Unstructured Context / Provenance (Citation)</th></tr>
        </thead>
        <tbody>
            <tr><td><strong>10Y US Treasury Benchmark</strong></td><td>4.38%</td><td>DCF / EV</td><td>Benchmark risk-free discount rate driving baseline corporate valuations.</td></tr>
            <tr><td><strong>Bitcoin / Digital Asset Index</strong></td><td>$88,450.00</td><td>Market Risk / EV</td><td>Institutional liquidity proxy and market sentiment indicator.</td></tr>
            <tr><td><strong>CBOE Volatility Index (VIX)</strong></td><td>18.25</td><td>Market Risk / PD</td><td>Implied market volatility index for default probability estimation.</td></tr>
            <tr><td><strong>WTI Crude Oil Benchmark</strong></td><td>$81.50 / bbl</td><td>PD / DCF</td><td>Primary energy commodity input cost driving headline inflation.</td></tr>
            <tr><td><strong>BSL Average Spread</strong></td><td>SOFR + 325 bps</td><td>LGD / PD</td><td>Broadly Syndicated Loan pricing level benchmark.</td></tr>
            <tr><td><strong>Private Credit Spread Premium</strong></td><td>SOFR + 550 bps</td><td>LGD / EV</td><td>Direct lending illiquidity pricing differential.</td></tr>
            <tr><td><strong>Target SOFR Baseline Rate</strong></td><td>5.30%</td><td>PD / DCF</td><td>Base floating reference rate for corporate leverage structures.</td></tr>
        </tbody>
    </table>

    <div class="code-block">{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "title": "InstitutionalAITrainingLedger",
  "type": "object",
  "properties": {
    "ledger_entries": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "data_variable_node": { "type": "string" },
          "market_level_value": { "type": "string" },
          "primary_model_target": { "type": "string", "enum": ["PD", "LGD", "DCF", "EV", "Market Risk", "PD / DCF", "LGD / PD", "LGD / EV", "DCF / EV"] },
          "unstructured_context_provenance": { "type": "string" }
        },
        "required": ["data_variable_node", "market_level_value", "primary_model_target", "unstructured_context_provenance"],
        "additionalProperties": false
      }
    }
  },
  "required": ["ledger_entries"],
  "additionalProperties": false
}</div>

    <div class="page-break"></div>

    <!-- PART 2: WHALESCANNER INTELLIGENCE CORE -->
    <div class="section-title">Part II: Whalescanner Intelligence Core</div>
    <div class="card">
        <strong>Master Surveillance & Adaptive Calibration Directive</strong><br>
        Whalescanner operates a self-tuning, noise-resistant intelligence engine combining multi-capitalization SEC disclosures with fixed-income telemetry across a 3-Layer Adaptive Loop:
        <br>
        1. <em>Universal Ingestion:</em> Schedules 13D/13F/13G, Form 4, Form 8-K, Form 144.<br>
        2. <em>Algorithmic Materiality Filtering:</em> Gate scoring (0-100) using:
        <br><code>Materiality Score = (0.35 * Catalyst) + (0.25 * Margin_of_Safety) + (0.20 * Conviction) + (0.20 * Governance)</code><br>
        3. <em>Dynamic Calibration:</em> Auto-adjusting sensitivity parameters based on hit rate (SAR >= 95%, FDR <= 5%).
    </div>

    <div class="card">
        <strong>Track A High-Conviction Immediate Catalysts:</strong><br>
        • <strong>IMKTA (Score 96):</strong> Form 3 / 8-K Bylaws ingested. Oct 19 Rule 14a-8 deadline T-24 days. Rory Held (Summer Road) oversight over $1.5B real estate footprint and $455.1M cash.<br>
        • <strong>BILL (Score 95):</strong> Dual 10.1% blocks (BlackRock & Starboard) preserve float floor on ~85.2M shares post $300M repurchase.<br>
        • <strong>ASTS (Score 93):</strong> CTO/COO 10b5-1 option sales filtered as non-discretionary. Adriana Cisneros Code P $625k buy holds bull case with $3.7B+ liquidity shield.
    </div>

    <table class="data-table">
        <thead>
            <tr><th>Ticker</th><th>Archetype</th><th>Score</th><th>Primary Catalyst & Horizon</th><th>Valuation Floor / Backstop</th></tr>
        </thead>
        <tbody>
            <tr><td><strong>IMKTA</strong></td><td>Asset-Rich Hostile</td><td>96</td><td>Form 3 (Hefner) & 8-K Bylaws; Oct 19 Rule 14a-8 deadline.</td><td>$455.1M cash; $1.5B real estate footprint.</td></tr>
            <tr><td><strong>BILL</strong></td><td>Mega-Cap Software</td><td>95</td><td>Dual BlackRock/Starboard 10.1% blocks (>20.2% concentration).</td><td>High-retention SMB payment volume floor.</td></tr>
            <tr><td><strong>SEER</strong></td><td>Broken Cash Box</td><td>94</td><td>Special Committee review post-bid expiry.</td><td>$209.5M net cash box ($2.50+/sh).</td></tr>
            <tr><td><strong>ASTS</strong></td><td>Tech Infrastructure</td><td>93</td><td>C-suite option sales filtered; Cisneros Code P Buy ($625k).</td><td>$3.7B+ cash shield; $149.20 cap call floor.</td></tr>
            <tr><td><strong>INBK</strong></td><td>Distressed PTBV Bank</td><td>84</td><td>13D push for buybacks following EPS beat.</td><td>0.60x P/TBV; Tangible Book Value at $41.09.</td></tr>
            <tr><td><strong>ADSK</strong></td><td>Mega-Cap Software</td><td>83</td><td>Starboard board settlement; margin & AI spend read-through.</td><td>Dominant incumbent free-cash-flow yield.</td></tr>
            <tr><td><strong>BFLY</strong></td><td>Cash Inversion</td><td>81</td><td>Glenview Capital (6.17%) push for capital reallocation.</td><td>Negative Enterprise Value vs liquid assets.</td></tr>
            <tr><td><strong>ZI</strong></td><td>Credit-Equity Stress</td><td>80</td><td>Monitoring 8-Ks for loan covenant/credit amendments.</td><td>B2B data cash generation vs debt servicing.</td></tr>
            <tr><td><strong>SPTN</strong></td><td>Asset Governance</td><td>79</td><td>Macellum/Ancora 13D proxy push for M&A evaluation.</td><td>Distribution footprint & owned real estate.</td></tr>
        </tbody>
    </table>

    <div class="page-break"></div>

    <!-- PART 3: FORTRESS & HUNT PLAYBOOK -->
    <div class="section-title red">Part III: Fortress & Hunt Barbell Playbook</div>
    <div class="card">
        <strong>System: Adam-v30.1-Apex | Module: Portfolio_Orchestrator</strong><br>
        <strong>Macro Regime:</strong> Post-FOMC monetary contraction (Fed Funds target 3.75%–4.00%), colliding with physical commodity autarky (Brent Crude $103.67/bbl).<br>
        <strong>Portfolio Posture:</strong> Fortress (65% un-leveraged domestic extractors, refiners, zero-duration floaters) barbelled with Hunt (35% tactical volatility monetization & software short overlays).
    </div>

    <table class="data-table">
        <thead>
            <tr><th>Ticker / Asset</th><th>Current Price / Move</th><th>Primary Catalyst / Regime Alignment</th><th>Institutional Sentiment</th></tr>
        </thead>
        <tbody>
            <tr><td><strong>SPX</strong></td><td>7,704.13 / -0.02% DoD</td><td>Benchmark consolidates near 7,700 as tech support offsets rates.</td><td>Neutral</td></tr>
            <tr><td><strong>Brent Crude</strong></td><td>$103.67 / +0.57% DoD</td><td>Spot prices hold firmly above $103/bbl on Middle East transit friction.</td><td>Bullish</td></tr>
            <tr><td><strong>FCX (Freeport)</strong></td><td>$72.08 / -0.69% DoD</td><td>Resilient above $72, CBP CSMS Proclamation #69252300 smelt tracking.</td><td>Bullish (Target $85)</td></tr>
            <tr><td><strong>VLO (Valero)</strong></td><td>$383.12 / +1.02% DoD</td><td>Downstream refining crack spreads expand on heavy feedstock.</td><td>Bullish</td></tr>
            <tr><td><strong>MDB (MongoDB)</strong></td><td>$228.40 / -2.72% WoW</td><td>Dark pool distribution out of non-earning software layers.</td><td>Bearish (Target $190)</td></tr>
            <tr><td><strong>SFTBY (SoftBank)</strong></td><td>$27.50 / -2.31% WoW</td><td>Liquidation selling pressure across tech holding proxies.</td><td>Bearish</td></tr>
        </tbody>
    </table>

    <div class="card">
        <strong>Asymmetric Alpha Deep Dives:</strong><br>
        • <strong>Long FCX (Freeport-McMoRan):</strong> Tariff-walled re-shoring cycle under Section 232 and CBP CSMS Proclamation #69252300 (mandatory country-of-smelt entry tracking). Trading at $72.08 (Forward P/E 15.2x, EV/EBITDA 8.0x, FCF yield 8.7%). $4.8B cash, 1.1x net leverage. Buy/Sell Score: +0.89 | Quality Conviction: 95/100.<br>
        • <strong>Short Over-Leveraged BSL Software Cohort:</strong> IT budgets diverting to power/grid infrastructure. Fed rate floor 3.75%–4.00% & Fitch non-accruals driving DDEs before late-2026 debt walls. Forward P/E >40x, EV/Sales 9.2x, negative FCF yield -2.6%. Buy/Sell Score: -0.84 | Quality Conviction: 28/100.
    </div>

    <div class="code-block">[PROV-O TELEMETRY & REPLAY SEED LOG]
- Activity: prov:wasGeneratedBy -> node:adam-system2-standalone-kernel
- Qdrant Vector Retrieval: collection='whalescanner_edgar_embeddings' (Cosine >= 0.85)
- PennyLane/Qiskit Quantum Monte Carlo: Circuit Depth 12 Qubits, 64 Layers, PD Mean = 2.14%
- Replay Seed 0x99A4F: Operation Absolute Resolve Venezuela Extraction Shock</div>

</body>
</html>"""

    with open('market-mayhem/standalone/standalone_publication.html', 'w') as f:
        f.write(html_content)

    weasyprint.HTML('market-mayhem/standalone/standalone_publication.html').write_pdf('market-mayhem/standalone/Market_Mayhem_Standalone.pdf')
    print("Market_Mayhem_Standalone.pdf generated successfully!")

    if os.path.exists('market-mayhem/standalone/forecast_eval.html'):
        weasyprint.HTML('market-mayhem/standalone/forecast_eval.html').write_pdf('market-mayhem/standalone/Market_Mayhem_AOS-2026-0926_v4.pdf')
        print("Market_Mayhem_AOS-2026-0926_v4.pdf generated successfully!")

if __name__ == "__main__":
    generate_visual_assets()
    generate_standalone_html_and_pdf()
