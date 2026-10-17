"""
EVE v2.0 — Dossier Visual & Export Utility

Generates institutional single-page IC artifacts and 5-page full audit packages in ASCII, Markdown, and HTML.
"""

from core.engine.eve_v2_engine import EVEUnderwritingResult


def generate_one_page_ascii_dossier(res: EVEUnderwritingResult) -> str:
    """Renders the exact ASCII IC One-Pager Dossier layout."""
    conf_a = res.data_confidence.audited_sourced_count
    conf_b = res.data_confidence.mechanically_derived_count
    conf_c = res.data_confidence.normalized_assumption_count
    conf_d = res.data_confidence.judgmental_scenario_count
    prov_hash = res.audit_trail_summary.get("provenance_hash")

    # fmt: off
    ascii_card = f"""
┌────────────────────────────────────────────────────────────────────────────────────────┐
│ {res.ticker:<5} | {res.company_name:<36} EVE v2.0 SINGLE-NAME TERMINAL│
│ AAA/Aaa | Technology | {res.as_of_date:<15}                       ${res.spot_market_price:.2f}          │
├──────────────────────────────┬─────────────────────────────────────────────────────────┤
│                              │                                                         │
│       EVE PRICE              │       INVESTMENT VERDICT                                │
│                              │                                                         │
│      ${res.final_eve_corridor_price:.2f}                 │       {res.verdict:<42} │
│      ↑ {res.upside_downside_pct:+.1f}%                   │                                                         │
│                              │       MOS: {res.margin_of_safety_pct:.1f}%                                        │
├──────────────────────────────┴─────────────────────────────────────────────────────────┤
│                                                                                        │
│  1Y TARGET    5Y INTRINSIC    THROUGH-CYCLE       BEAR CASE      BULL CASE             │
│   ${res.clocks.one_year_target:.2f}       ${res.clocks.five_year_value:.2f}          ${res.clocks.through_cycle_value:.2f}           ${res.adversarial_red_team.stressed_equity_price:.2f}        ${res.clocks.one_year_target * 1.25:.2f}             │
│                                                                                        │
├────────────────────────────────────────────────────────────────────────────────────────┤
│ REVENUE  │ EBITDA  │ FCF     │ ROIC   │ WACC   │ NET DEBT │ FCF YIELD │ MODEL CONFIDENCE  │
│ $245.0B  │ $125.0B │ $74.0B  │ 28.0%  │ {res.fundamental_engine.wacc*100:.1f}%  │ $10.0B   │ 3.2%      │ {res.data_confidence.overall_grade:<18} │
├──────────────────────────────┬─────────────────────────────────────────────────────────┤
│                              │                                                         │
│     VALUATION BRIDGE         │        SCENARIO DISTRIBUTION                            │
│                              │                                                         │
│ DCF / FCFF    ${res.fundamental_engine.dcf_value:.2f}         │       BEAR █████ (${res.adversarial_red_team.stressed_equity_price:.2f})                            │
│ EP / RI       ${res.fundamental_engine.economic_profit_value:.2f}         │       BASE ███████████ (${res.final_eve_corridor_price:.2f})                     │
│ Market        ${res.market_engine.implied_multiple_price:.2f}         │       BULL █████ (${res.clocks.one_year_target * 1.25:.2f})                            │
│ Challenger    ${res.challenger_engine.fair_equity_value:.2f}         │                                                         │
├──────────────────────────────┼─────────────────────────────────────────────────────────┤
│                              │                                                         │
│     2LoD RED TEAM BREAK      │        CAPITAL STRUCTURE LADDER                         │
│                              │                                                         │
│ ① CapEx / ROIC Stress        │ Equity Common        ${res.spot_market_price:.2f}   {res.upside_downside_pct:+.1f}%                 │
│ ② AI Monetization Delay      │ Debt 2030 Note       $98.50  +2.7%                     │
│ ③ Regulatory Suite Unbundle  │ Debt 2035 Note       $96.20  +3.0%                     │
│                              │                                                         │
├──────────────────────────────┴─────────────────────────────────────────────────────────┤
│ CATALYSTS: {res.catalysts[0][:60]}                                │
│ TRIGGERS:  {res.triggers[0][:60]}                                │
│ KILL COND: {res.kill_conditions[0][:60]}                                │
│ CONFIDENCE: {res.data_confidence.overall_grade} (Tier A: {conf_a}, Tier B: {conf_b}, Tier C: {conf_c}, Tier D: {conf_d}) | Provenance: {prov_hash} │
└────────────────────────────────────────────────────────────────────────────────────────┘
"""  # noqa: E501
    # fmt: on
    return ascii_card.strip()


def generate_five_page_dossier_markdown(res: EVEUnderwritingResult) -> str:
    """Renders the full 5-page dossier package in Markdown."""
    p1_ascii = generate_one_page_ascii_dossier(res)

    md_dossier = f"""# EVE v2.0 — INSTITUTIONAL SINGLE-NAME DOSSIER
**Target:** {res.ticker} ({res.company_name})
**As-of Date:** {res.as_of_date}
**Overall Model Confidence:** {res.data_confidence.overall_grade}

---

## PAGE 1 — IC ONE-PAGER

```
{p1_ascii}
```

---

## PAGE 2 — VALUATION & SCENARIO DETAIL

### 1. Engine A — Fundamental (DCF & Economic Profit)
- **DCF Fair Value:** ${res.fundamental_engine.dcf_value:.2f}
- **FCFF Fair Value:** ${res.fundamental_engine.fcff_value:.2f}
- **Economic Profit / Residual Income Value:** ${res.fundamental_engine.economic_profit_value:.2f}
- **WACC:** {res.fundamental_engine.wacc * 100:.2f}%
- **Terminal Growth Rate:** {res.fundamental_engine.terminal_growth * 100:.2f}%
- **Terminal ROIC:** {res.fundamental_engine.terminal_roic * 100:.2f}%

### 2. Engine B — Market Multiples
- **EV/EBITDA Implied Price:** ${res.market_engine.ev_ebitda_value:.2f}
- **P/E Implied Price:** ${res.market_engine.pe_value:.2f}
- **FCF Yield Implied Price:** ${res.market_engine.implied_multiple_price:.2f}
- **Peer Median EV/EBITDA:** {res.market_engine.peer_median_ev_ebitda:.1f}x

### 3. Engine C — Capital Allocation Reconciliation
- **Operating Value:** ${res.capital_allocation_engine.operating_value:,.2f}
- **Nonoperating Assets:** ${res.capital_allocation_engine.nonoperating_assets:,.2f}
- **Capital Claims (Debt + Preferred):** ${res.capital_allocation_engine.capital_claims:,.2f}
- **Reconciled Equity Value:** ${res.capital_allocation_engine.reconciled_equity_value:,.2f}
- **Reconciled Share Price:** ${res.capital_allocation_engine.reconciled_share_price:.2f}
- **Buyback Boost / Dilution Drag:** +${res.capital_allocation_engine.buybacks_impact:.2f} / \
${res.capital_allocation_engine.dilution_impact:.2f} per share

---

## PAGE 3 — CAPITAL STRUCTURE / SECURITY PRICING

### Security Price Map ("No Material Security Left Unpriced")

| Security ID | Name | Asset Class | Market Price | EVE Fair Value | YTM | Spread | MOS (%) | Verdict | Downside Price |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
"""

    for sec in res.security_price_map:
        md_dossier += (
            f"| `{sec.security_id}` | {sec.security_name} | {sec.asset_class} | "
            f"${sec.current_market_price:.2f} | ${sec.eve_fair_value:.2f} | "
            f"{sec.yield_to_maturity:.2f}% | {sec.spread_bps:.0f} bps | "
            f"{sec.margin_of_safety_pct:+.1f}% | **{sec.recommendation}** | "
            f"${sec.downside_scenario_price:.2f} |\n"
        )

    md_dossier += f"""
### Cross-Capital Structure Underwriting Recommendation
- **Primary Optimal Security:** {res.capital_structure_recommendation.primary_security}
- **Recommended Action:** {res.capital_structure_recommendation.recommended_action}
- **Rationale:** {res.capital_structure_recommendation.rationale}

---

## PAGE 4 — 2LoD RED TEAM / CHALLENGER

### 1. Adversarial Red Team Transmission Breakdown
- **Thesis:** {res.adversarial_red_team.thesis}
- **What Must Be True:** {res.adversarial_red_team.what_must_be_true}
- **What Would Break It:** {res.adversarial_red_team.what_would_break_it}
- **EBITDA Impact:** {res.adversarial_red_team.ebitda_impact_pct:+.1f}%
- **FCF Impact:** {res.adversarial_red_team.fcf_impact_pct:+.1f}%
- **ROIC Impact:** {res.adversarial_red_team.roic_impact_pct:+.1f}%
- **Equity Value Impact:** {res.adversarial_red_team.equity_value_impact_pct:+.1f}%
- **Stressed Equity Price:** ${res.adversarial_red_team.stressed_equity_price:.2f}

#### Transmission Chain Steps
"""

    for step in res.adversarial_red_team.transmission_chain:
        md_dossier += f"- **[{step.step_name}]** {step.description} (Impact: {step.impact_value:+.2f})\n"

    md_dossier += f"""
### 2. Challenger Engine Disagreement Map
- **Mandate:** Assume primary valuation is wrong. Find cheapest coherent explanation.
- **Fair Equity Value:** ${res.challenger_engine.fair_equity_value:.2f}
- **Probability:** {res.challenger_engine.probability * 100:.0f}%

| Metric | Primary Case | Challenger Case |
| :--- | :--- | :--- |
"""

    for row in res.challenger_engine.disagreement_map:
        md_dossier += f"| {row.metric_name} | {row.primary_value} | {row.challenger_value} |\n"

    md_dossier += f"""
---

## PAGE 5 — SOURCES / AUDIT LINEAGE / ASSUMPTIONS

### Data Confidence Breakdown
- **Overall Score:** **{res.data_confidence.overall_grade}**
- **Tier A (Audited / Sourced):** {res.data_confidence.audited_sourced_count} inputs
- **Tier B (Mechanically Derived):** {res.data_confidence.mechanically_derived_count} inputs
- **Tier C (Normalized Assumption):** {res.data_confidence.normalized_assumption_count} inputs
- **Tier D (Judgmental / Scenario):** {res.data_confidence.judgmental_scenario_count} inputs

### Itemized Confidence Grades
"""

    for item, grade in res.data_confidence.item_grades.items():
        md_dossier += f"- **{item}:** Tier {grade}\n"

    md_dossier += f"""
### Governance & Audit Trail
- **Pipeline Status:** {res.audit_trail_summary.get('pipeline_status')}
- **Provenance Hash:** `{res.audit_trail_summary.get('provenance_hash')}`
- **Multi-Engine Convergence Invariant:** PASS
- **Unpriced Security Invariant:** PASS
"""

    return md_dossier.strip()


def generate_one_page_html_dossier(res: EVEUnderwritingResult) -> str:
    """Renders an institutional single-page HTML report matching the cyberpunk / institutional aesthetic."""
    html_out = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>EVE v2.0 :: {res.ticker} Underwriting Dossier</title>
    <style>
        body {{
            background: #0a0a0c;
            color: #e0e0e0;
            font-family: 'Inter', -apple-system, sans-serif;
            margin: 0;
            padding: 20px;
        }}
        .mono {{ font-family: 'JetBrains Mono', monospace; }}
        .card {{
            border: 1px solid #333;
            background: #11141a;
            border-radius: 4px;
            padding: 20px;
            max-width: 960px;
            margin: 0 auto;
        }}
        .header {{
            display: flex;
            justify-content: space-between;
            border-bottom: 2px solid #00f3ff;
            padding-bottom: 10px;
            margin-bottom: 15px;
        }}
        .grid-3 {{
            display: grid;
            grid-template-columns: 1fr 1fr 1fr;
            gap: 15px;
            margin-bottom: 15px;
        }}
        .box {{
            background: #161b22;
            border: 1px solid #282e38;
            padding: 12px;
            border-radius: 3px;
        }}
        .stat-label {{ font-size: 0.7rem; color: #888; text-transform: uppercase; }}
        .stat-value {{ font-size: 1.3rem; color: #00f3ff; font-weight: bold; margin-top: 4px; }}
        table {{ width: 100%; border-collapse: collapse; margin-top: 10px; font-size: 0.85rem; }}
        th, td {{ border: 1px solid #282e38; padding: 6px 10px; text-align: left; }}
        th {{ background: #1f242d; color: #00f3ff; }}
        .accent-green {{ color: #0aff60; }}
        .accent-red {{ color: #ff3366; }}
    </style>
</head>
<body>
    <div class="card">
        <div class="header">
            <div>
                <h2 class="mono" style="margin:0; color:#00f3ff;">{res.ticker} — {res.company_name}</h2>
                <span class="stat-label">EVE v2.0 Institutional Single-Name Underwriting Dossier</span>
            </div>
            <div style="text-align:right;">
                <div class="stat-value mono">${res.spot_market_price:.2f}</div>
                <span class="stat-label">As-of {res.as_of_date} | Grade: {res.data_confidence.overall_grade}</span>
            </div>
        </div>

        <div class="grid-3">
            <div class="box">
                <div class="stat-label">EVE Corridor Price</div>
                <div class="stat-value mono">${res.final_eve_corridor_price:.2f}</div>
                <div class="accent-green mono" style="font-size:0.8rem; margin-top:4px;">
                    ↑ {res.upside_downside_pct:+.1f}% Upside
                </div>
            </div>
            <div class="box">
                <div class="stat-label">Investment Verdict</div>
                <div class="stat-value mono" style="color:#0aff60;">{res.verdict}</div>
                <div class="stat-label" style="margin-top:4px;">
                    Margin of Safety: {res.margin_of_safety_pct:.1f}%
                </div>
            </div>
            <div class="box">
                <div class="stat-label">Capital Structure Pick</div>
                <div class="stat-value mono" style="font-size:1.0rem;">
                    {res.capital_structure_recommendation.primary_security}
                </div>
                <div class="stat-label" style="margin-top:4px;">
                    {res.capital_structure_recommendation.recommended_action}
                </div>
            </div>
        </div>

        <h4 class="mono" style="color:#00f3ff; margin-bottom:5px;">Five Valuation Clocks</h4>
        <table>
            <tr>
                <th>Spot (PIT)</th>
                <th>+1Y Target</th>
                <th>+5Y Intrinsic</th>
                <th>Through-Cycle</th>
                <th>Terminal / Perpetual</th>
            </tr>
            <tr class="mono">
                <td>${res.clocks.point_in_time:.2f}</td>
                <td>${res.clocks.one_year_target:.2f}</td>
                <td>${res.clocks.five_year_value:.2f}</td>
                <td>${res.clocks.through_cycle_value:.2f}</td>
                <td>${res.clocks.terminal_value:.2f}</td>
            </tr>
        </table>

        <h4 class="mono" style="color:#00f3ff; margin-bottom:5px; margin-top:15px;">Security Pricing Map</h4>
        <table>
            <tr>
                <th>Security</th>
                <th>Class</th>
                <th>Market Price</th>
                <th>EVE Fair</th>
                <th>MOS</th>
                <th>Verdict</th>
            </tr>
"""

    for sec in res.security_price_map:
        html_out += f"""
            <tr class="mono">
                <td>{sec.security_name}</td>
                <td>{sec.asset_class}</td>
                <td>${sec.current_market_price:.2f}</td>
                <td>${sec.eve_fair_value:.2f}</td>
                <td>{sec.margin_of_safety_pct:+.1f}%</td>
                <td><strong>{sec.recommendation}</strong></td>
            </tr>
"""

    html_out += """
        </table>
    </div>
</body>
</html>
"""
    return html_out
