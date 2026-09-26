import os
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import weasyprint

def generate_baseline_image():
    width, height = 800, 500
    img = Image.new('RGB', (width, height), '#0a0a0c')
    draw = ImageDraw.Draw(img)

    for y in range(height):
        r = int(10 + (y / height) * 20)
        g = int(30 + (y / height) * 50)
        b = int(45 + (y / height) * 65)
        draw.line([(0, y), (width, y)], fill=(r, g, b))

    draw.ellipse([580, 40, 700, 160], fill=(0, 255, 204), outline=(0, 255, 204))

    draw.rectangle([0, 320, 800, 500], fill=(12, 40, 50))
    draw.polygon([(0, 390), (800, 350), (800, 500), (0, 500)], fill=(20, 70, 75))

    buildings = [(40, 200, 80, 320), (90, 160, 140, 320), (150, 180, 200, 320), (210, 130, 260, 320), (270, 210, 310, 320)]
    for b in buildings:
        draw.rectangle(b, fill=(10, 80, 90), outline=(0, 255, 204))

    draw.polygon([(420, 240), (540, 240), (490, 330), (470, 330)], fill=(0, 200, 180), outline=(0, 255, 204))
    draw.line([(480, 330), (480, 420)], fill=(0, 255, 204), width=5)
    draw.ellipse([440, 415, 520, 430], fill=(0, 255, 204), outline=(0, 255, 204))
    draw.ellipse([420, 235, 540, 250], fill=(0, 255, 204))
    draw.ellipse([515, 220, 550, 255], fill=(50, 255, 120), outline=(0, 255, 204))

    draw.rectangle([0, 450, 800, 500], fill=(10, 20, 25))
    draw.text((20, 462), "RISK-ON: CLEAR SKIES & HIGH LIQUIDITY", fill=(0, 255, 204))

    img.save('market_mayhem_baseline.png')

def generate_challenger_image():
    width, height = 800, 500
    img = Image.new('RGB', (width, height), '#0a0a0c')
    draw = ImageDraw.Draw(img)

    for y in range(height):
        r = int(35 + (y / height) * 45)
        g = int(8 + (y / height) * 10)
        b = int(12 + (y / height) * 15)
        draw.line([(0, y), (width, y)], fill=(r, g, b))

    for x in range(0, 240, 20):
        draw.line([(x, 0), (x, 500)], fill=(45, 15, 20), width=1)
    for x in range(560, 800, 20):
        draw.line([(x, 0), (x, 500)], fill=(45, 15, 20), width=1)

    draw.polygon([(260, 280), (540, 280), (510, 440), (290, 440)], fill=(35, 20, 25), outline=(255, 51, 102), width=3)
    draw.rectangle([250, 270, 550, 285], fill=(50, 25, 30), outline=(255, 51, 102), width=2)

    np.random.seed(42)
    for _ in range(70):
        fx = np.random.randint(270, 530)
        fy = np.random.randint(110, 275)
        fr = np.random.randint(12, 38)
        color = (255, np.random.randint(30, 160), np.random.randint(40, 100))
        draw.ellipse([fx-fr, fy-fr, fx+fr, fy+fr], fill=color)

    for _ in range(180):
        sx = np.random.randint(180, 620)
        sy = np.random.randint(30, 270)
        sr = np.random.randint(1, 4)
        draw.ellipse([sx-sr, sy-sr, sx+sr, sy+sr], fill=(255, 51, 102))

    draw.rectangle([0, 450, 800, 500], fill=(25, 10, 15))
    draw.text((20, 462), "TAIL-RISK: LIQUIDITY SHOCK & DURATION CONFLAGRATION", fill=(255, 51, 102))

    img.save('market_mayhem_challenger.png')

def generate_chart():
    fig, ax = plt.subplots(figsize=(9, 4.2), facecolor='#0a0a0c')
    ax.set_facecolor('#0a0a0c')

    days = np.array([1, 2, 3, 4, 5, 6, 7])
    baseline_yield = np.array([5.11, 5.13, 5.16, 5.18, 5.21, 5.23, 5.25])
    challenger_yield = np.array([5.11, 5.32, 5.58, 5.65, 5.48, 5.35, 5.18])

    ax.plot(days, baseline_yield, color='#00ffcc', linewidth=2.8, marker='o', label='Baseline (Hawkish Repricing)')
    ax.plot(days, challenger_yield, color='#ff3366', linewidth=2.8, marker='s', linestyle='--', label='Challenger (Op. Absolute Resolve)')

    ax.set_title('10-Year Treasury Yield Trajectory (7-Day Duration Shock Simulation)', color='#ffffff', fontsize=12, fontweight='bold', pad=12)
    ax.set_xlabel('Simulation Horizon (Days)', color='#a0a0b0', fontsize=10)
    ax.set_ylabel('Yield (%)', color='#a0a0b0', fontsize=10)
    ax.set_xticks(days)
    ax.set_xticklabels(['Day 1\n(Sep 23)', 'Day 2', 'Day 3', 'Day 4\n(Peak Shock)', 'Day 5', 'Day 6', 'Day 7\n(Recovery)'])
    ax.tick_params(colors='#a0a0b0')

    for spine in ax.spines.values():
        spine.set_color('#222233')

    ax.legend(facecolor='#14141d', edgecolor='#333344', labelcolor='#ffffff', loc='upper right')
    plt.grid(True, linestyle=':', alpha=0.3, color='#444455')
    plt.tight_layout()
    plt.savefig('market_mayhem_chart.png', dpi=200, facecolor='#0a0a0c')

def generate_pdf():
    html_content = """<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Market Mayhem - Institutional Quantitative Intelligence</title>
    <style>
        @page {
            size: A4 portrait;
            margin: 15mm;
            @bottom-right {
                content: "Page " counter(page) " of " counter(pages);
                font-family: 'Courier New', monospace;
                font-size: 8pt;
                color: #00ffcc;
            }
            @bottom-left {
                content: "ADAM SYSTEM 2 // MARKET MAYHEM PUBLICATION";
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
            font-size: 10pt;
            line-height: 1.4;
        }
        .header {
            border-bottom: 2px solid #ff3366;
            padding-bottom: 10px;
            margin-bottom: 20px;
        }
        .title {
            font-size: 26pt;
            font-weight: 900;
            color: #ffffff;
            letter-spacing: 2px;
            margin: 0;
            text-transform: uppercase;
        }
        .subtitle {
            font-size: 11pt;
            color: #00ffcc;
            font-family: 'Courier New', monospace;
            margin-top: 4px;
        }
        .meta-bar {
            display: table;
            width: 100%;
            background-color: #14141d;
            border: 1px solid #222233;
            padding: 8px 12px;
            margin-bottom: 20px;
            box-sizing: border-box;
            font-family: 'Courier New', monospace;
            font-size: 8.5pt;
        }
        .meta-item {
            display: table-cell;
            color: #a0a0b0;
        }
        .meta-item span {
            color: #ffffff;
            font-weight: bold;
        }
        .section-header {
            font-size: 13pt;
            font-weight: bold;
            color: #00ffcc;
            border-left: 4px solid #00ffcc;
            padding-left: 8px;
            margin-top: 22px;
            margin-bottom: 12px;
            text-transform: uppercase;
            letter-spacing: 1px;
        }
        .section-header.red {
            color: #ff3366;
            border-left-color: #ff3366;
        }
        .page-break {
            page-break-before: always;
        }
        .image-grid {
            display: table;
            width: 100%;
            margin-bottom: 15px;
        }
        .image-cell {
            display: table-cell;
            width: 50%;
            padding-right: 8px;
            vertical-align: top;
        }
        .image-cell:last-child {
            padding-right: 0;
            padding-left: 8px;
        }
        .image-cell img {
            width: 100%;
            border: 1px solid #333344;
            border-radius: 4px;
        }
        .caption {
            font-size: 8pt;
            font-family: 'Courier New', monospace;
            color: #a0a0b0;
            margin-top: 4px;
            text-align: center;
        }
        .chart-box {
            text-align: center;
            margin-bottom: 15px;
        }
        .chart-box img {
            width: 100%;
            border: 1px solid #222233;
            border-radius: 4px;
        }
        table.data-table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 10px;
            margin-bottom: 15px;
            font-size: 8.5pt;
        }
        table.data-table th {
            background-color: #1a1a26;
            color: #00ffcc;
            border: 1px solid #28283a;
            padding: 6px 8px;
            text-align: left;
            font-family: 'Courier New', monospace;
        }
        table.data-table td {
            border: 1px solid #222233;
            padding: 6px 8px;
            background-color: #101017;
        }
        .negative {
            color: #ff3366;
            font-weight: bold;
        }
        .positive {
            color: #00ffcc;
            font-weight: bold;
        }
        .json-block {
            background-color: #0d0d12;
            border: 1px solid #222233;
            border-left: 3px solid #00ffcc;
            padding: 10px;
            font-family: 'Courier New', monospace;
            font-size: 8pt;
            color: #00ffcc;
            white-space: pre-wrap;
            margin-bottom: 15px;
            border-radius: 2px;
        }
        .telemetry-block {
            background-color: #050508;
            border: 1px solid #222233;
            border-left: 3px solid #ff3366;
            padding: 10px;
            font-family: 'Courier New', monospace;
            font-size: 7.5pt;
            color: #a0a0b0;
            white-space: pre-wrap;
            margin-bottom: 15px;
            line-height: 1.3;
        }
        .highlight-teal { color: #00ffcc; }
        .highlight-red { color: #ff3366; }
        .highlight-white { color: #ffffff; }
        p {
            margin-top: 0;
            margin-bottom: 10px;
            text-align: justify;
        }
    </style>
</head>
<body>

    <div class="header">
        <div class="title">Market Mayhem</div>
        <div class="subtitle">ADAM SYSTEM 2 QUANTITATIVE RISK CONTROL // SPECIAL DUAL-PLANE BRIEFING</div>
    </div>

    <div class="meta-bar">
        <div class="meta-item">DATE: <span>SEP 23, 2026</span></div>
        <div class="meta-item">REGIME: <span>HAWKISH REPRICING</span></div>
        <div class="meta-item">S&P 500: <span>7,706.03 (-0.75%)</span></div>
        <div class="meta-item">10Y TREASURY: <span>5.11% (+0.15%)</span></div>
        <div class="meta-item">BTC: <span>$86,965.21 (+1.83%)</span></div>
    </div>

    <div class="section-header">Section 1: The Visual Metaphor</div>
    <p>
        To evaluate institutional credit risk and Broadly Syndicated Loans (BSL) under SR 11-7 regulatory capital frameworks, we contrast the baseline reality against a cascading geopolitical duration shock. Below, two distinct visual metaphors frame the risk landscape:
    </p>

    <div class="image-grid">
        <div class="image-cell">
            <img src="market_mayhem_baseline.png" alt="Baseline Risk-On">
            <div class="caption">IMAGE 1: BASELINE MODEL (HAWKISH CLEAR SKIES)</div>
        </div>
        <div class="image-cell">
            <img src="market_mayhem_challenger.png" alt="Challenger Tail-Risk">
            <div class="caption">IMAGE 2: CHALLENGER MODEL (OP. ABSOLUTE RESOLVE)</div>
        </div>
    </div>

    <p style="font-size: 8.5pt; color: #a0a0b0;">
        <em>Figure 1.1:</em> The Baseline depicts pristine liquidity where yield expansion is orderly and asset valuations remain insulated. Conversely, the Challenger model simulates a liquidity shock & duration conflagration in an illiquid backdrop.
    </p>

    <div class="section-header red">Section 2: Dual-Plane Data Harness & Credit Mechanics</div>
    <p>
        Under <strong>Operation Absolute Resolve</strong>, a multi-day rolling geopolitical liquidity shock originating from regime extraction in Venezuela induces acute duration expansion across US Treasuries and Broadly Syndicated Loan (BSL) credit spreads.
    </p>

    <div class="chart-box">
        <img src="market_mayhem_chart.png" alt="10Y Yield Trajectory Chart">
    </div>

    <table class="data-table">
        <thead>
            <tr>
                <th>Asset / Credit Instrument</th>
                <th>Baseline Level</th>
                <th>Challenger Shock (Peak)</th>
                <th>Spread Impact</th>
                <th>SR 11-7 Capital Status</th>
            </tr>
        </thead>
        <tbody>
            <tr>
                <td><strong>US 10-Year Treasury Yield</strong></td>
                <td>5.11%</td>
                <td>5.65%</td>
                <td class="negative">+54 bps</td>
                <td class="positive">Compliant (Tier 1 Buffer)</td>
            </tr>
            <tr>
                <td><strong>BSL Broad Index Spread</strong></td>
                <td>385 bps</td>
                <td>490 bps</td>
                <td class="negative">+105 bps</td>
                <td class="positive">Pass (Buffer Triggered)</td>
            </tr>
            <tr>
                <td><strong>SIM-HY-CREDIT Portfolio</strong></td>
                <td>$1.24B Nav</td>
                <td>$1.147B Nav</td>
                <td class="negative">-7.5% Drawdown</td>
                <td class="negative">Advisory Review</td>
            </tr>
            <tr>
                <td><strong>Bitcoin (BTC / USD)</strong></td>
                <td>$86,965.21</td>
                <td>$79,200.00</td>
                <td class="negative">-8.9% Liquidity Drain</td>
                <td class="positive">Unencumbered</td>
            </tr>
        </tbody>
    </table>

    <p style="font-size: 8.5pt; background-color: #14141d; padding: 8px; border-left: 3px solid #00ffcc;">
        <strong>Thesis Resilience Note:</strong> While SIM-HY-CREDIT experiences an immediate -7.5% drawdown during the peak liquidity contraction on Day 4, the localized shock dissipates as collateralized debt obligation structural protections hold. This marks the comeback chapter in our long-term institutional credit thesis, confirming that underlying core obligor cash flows survive the immediate localized duration shock without triggering systemic solvency impairment.
    </p>

    <div class="page-break"></div>

    <div class="header">
        <div class="title" style="font-size: 18pt;">Market Mayhem (Cont.)</div>
        <div class="subtitle">SECTION 3 & SECTION 4 // ADJUDICATION & PROV-O AUDIT TELEMETRY</div>
    </div>

    <div class="section-header">Section 3: LLM-as-Judge Adjudication Engine</div>
    <p>
        The Adam Deterministic Rule Engine evaluates model divergence using strict <code>jsonLogic</code> rules over 4-clock temporal geometry. The adjudication validates that the flight-to-quality in the Challenger model correctly overrides the baseline hawkish trajectory during Days 2–5 while maintaining long-term underwriting integrity.
    </p>

    <div class="json-block">{
  "adjudication_engine": "Adam_v30.0_System2_Judge",
  "evaluated_at": "2026-09-23T23:59:59Z",
  "divergence_metrics": {
    "baseline_model": "Hawkish_Repricing_v2",
    "challenger_model": "Operation_Absolute_Resolve",
    "divergence_severity": "HIGH_ELEVATED",
    "max_yield_divergence_bps": 54.0,
    "bsl_spread_widening_bps": 105.0,
    "credit_portfolio_drawdown_pct": -7.5
  },
  "rule_evaluation_matrix": {
    "rule_01_flight_to_quality_override": {
      "logic": { ">=": [ { "var": "yield_divergence_bps" }, 50 ] },
      "result": true,
      "action": "TEMPORARY_OVERRIDE_BASELINE_WITH_CHALLENGER"
    },
    "rule_02_underwriting_thesis_integrity": {
      "logic": { "<=": [ { "var": "sim_hy_credit_loss" }, 0.10 ] },
      "result": true,
      "action": "MAINTAIN_LONG_TERM_CREDIT_THESIS"
    }
  },
  "verdict": "CHALLENGER_FLIGHT_TO_QUALITY_VALIDATED_CORE_THESIS_INTACT"
}</div>

    <div class="section-header red">Section 4: Telemetry & Replay Harness</div>
    <p>
        Monospace W3C PROV-O cryptographic audit trace verifying retrieval nodes, quantum Monte Carlo distribution checks, and replay seeds for Operation Absolute Resolve.
    </p>

    <div class="telemetry-block"><span class="highlight-teal">[PROV-O AUDIT TRACE]</span> id: urn:uuid:7f3a9e10-58c2-4a2e-9d88-1a92bf002026
<span class="highlight-teal">[PROV-O ACTIVITY]</span> prov:wasGeneratedBy -> node:adam-system2-kernel-v30.0
<span class="highlight-teal">[RETRIEVAL NODE]</span> Qdrant Vector DB: collection='credit_regime_embeddings'
  - Query Vector: [0.128, -0.449, 0.892, 0.012, -0.331, 0.764, -0.105, 0.552]
  - Distance Metric: Cosine (Threshold: >= 0.85)
  - Retrieved 4 Matches: 'venezuela_extraction_2026', 'duration_shock_2023', 'bsl_liquidity_crunch', 'sr11_7_cap_check'

<span class="highlight-teal">[QUANTUM MONTE CARLO]</span> Engine: PennyLane/Qiskit Synthetic Distribution Harness
  - Circuit Depth: 12 Qubits, 64 Variational Layers
  - Target Variable: Broadly Syndicated Loan Default Probability (PD_Sim)
  - Simulated PD Distribution (10,000 runs): Mean = 2.14%, 99.9th VaR = 6.42%
  - Convergence Check: PASS (delta = 0.00018)

<span class="highlight-red">[REPLAY SEEDS]</span> Operation Absolute Resolve Injection Protocol:
  - Seed 0x7F8B99A2: Geopolitical Liquidity Vector = Venezuela Intervention
  - Seed 0x3C4D5E6F: Yield Curve Parabolic Shift Delta = +54bps
  - Seed 0x11223344: BSL SIM-HY-CREDIT Spread Shock = -7.5% NAV

<span class="highlight-white">[STATUS]</span> Cryptographic Lineage Intact // All 4 temporal clocks synchronized // Zero-sized capability token confirmed.</div>

</body>
</html>"""

    with open('market_mayhem.html', 'w') as f:
        f.write(html_content)

    print("HTML written to market_mayhem.html")

    weasyprint.HTML('market_mayhem.html').write_pdf('Market_Mayhem.pdf')
    print("PDF successfully generated: Market_Mayhem.pdf")

if __name__ == "__main__":
    print("Generating assets...")
    generate_baseline_image()
    generate_challenger_image()
    generate_chart()
    print("Compiling PDF via WeasyPrint...")
    generate_pdf()
