# AFOS v30.1 Institutional Macro Pulse & Entity PD Sensitization Prompt

## Role & Context
Act as an institutional Chief Risk Officer, Quantitative Modeler, and Model Risk Auditor. Execute a complete macro pulse check, entity PD parameter sensitization, adversarial stress test, and audit log generation for: `[INSERT TARGET NAME/TICKER/ISIN]`.

## Execution Directives - Complete in 4 Consecutive Steps

### Step 1: Live Macro & Market Pulse Check
- Ingress live data (Current date: October 2026): Yield curve (2Y/10Y), VIX, MOVE, IG/HY spreads, central bank policy rate, and target entity live pricing (equity, bonds/CDS proxy, implied vol).
- Output a markdown **Macro & Market Pulse Table** with explicit URL/data source citations and Regime Classification (Expansion/Slowdown/Stagflation/Contraction/Liquidity Crunch).

### Step 2: Entity PD Parameter Sensitization Matrix
- Map current macro regime shifts into your Probability of Default (delta PD) model parameters (e.g., Merton distance-to-default, leverage, asset volatility, macro coefficients).
- Construct a **Parameter Sensitization Matrix** comparing:
  1. Baseline
  2. Live Macro State
  3. Moderate Stress (+1 SD spread, +20% vol)
  4. Severe Liquidity Shock (-35% equity, credit closure)
- Output resulting 1Y and 3Y cumulative PD values across all scenarios.

### Step 3: Adversarial Machine Challenge (Red Team)
- Critically stress-test the model: Identify uncaptured tail risks, non-linear feedback loops, upcoming debt walls, or covenant triggers.
- Provide a quantified **Challenged PD Range** adjusting the model outputs for red-team vulnerabilities.

### Step 4: Audit Trail & Expert Review Log
- Generate an immutable audit dossier containing:
  1. Data Lineage & Citations Registry.
  2. Full Parameter Change Log.
  3. Red-Team Vulnerability Log.
  4. Human Expert Review Sign-Off Template (Decision, Override Notes, Sign-off Status).

## Output Format
Deliver the complete analysis in a clean, professional, table-driven markdown report optimized for risk committee review and audit replay.
