# ADAM Financial Suite - Usage Guide

The ADAM Financial Suite is a modular, portable, and regulatory-compliant workstream architecture for financial modeling. It integrates Venture Capital (VC) Sponsor dynamics, Enterprise Value (EV) arbitration, and deep Credit Challenge sensitivity analysis.

## Core Concepts

*   **Workstream Context:** A JSON object encapsulating the entire state of the analysis (Inputs, Assumptions, Configuration).
*   **Service-Oriented Design:** Decoupled math engines (DCF, WACC), VC Logic, and Risk Logic.
*   **Interactivity:** A dependency-aware architecture allowing real-time recalculation.

## Directory Structure

*   `core/financial_suite/schemas`: Pydantic models for the Workstream Context.
*   `core/financial_suite/engines`: Core math engines (DCF, WACC, Solver).
*   `core/financial_suite/modules`: Business logic (VC, Risk, Reporting).
*   `core/financial_suite/interface`: Integration layer.

## Quick Start

### 1. Create a Context JSON

Create a file named `analysis_context.json`:

```json
{
  "meta": {
    "version": "2.1.0",
    "author": "Analyst",
    "timestamp": "2023-10-27T14:00:00Z"
  },
  "config": {
    "mode": "VC_SPONSOR",
    "calc_engine": "ITERATIVE",
    "regulatory_standard": "SNC_2025"
  },
  "valuation_context": {
    "wacc_method": "CAPM_STANDARD",
    "terminal_method": "DUAL_WEIGHTED",
    "growth_rate_perpetuity": 0.02,
    "exit_multiple": 12.0,
    "risk_free_rate": 0.04,
    "market_return": 0.10,
    "beta": 1.1,
    "pre_tax_cost_of_debt": 0.075,
    "tax_rate": 0.21
  },
  "capital_structure": {
    "securities": [
      {
        "name": "Senior Debt",
        "security_type": "TERM_LOAN",
        "priority": 1,
        "balance": 150.0,
        "interest_rate": 0.08
      },
      {
        "name": "Sponsor Equity",
        "security_type": "COMMON",
        "priority": 2,
        "balance": 0.0,
        "interest_rate": 0.0,
        "shares": 100.0,
        "investment": 100.0
      }
    ]
  },
  "credit_challenge": {
    "stress_test_active": false,
    "scenario": "BASE",
    "pd_method": "LOGISTIC_HYBRID"
  },
  "financials": {
    "current_year_revenue": 200.0,
    "historical_revenue": [150.0, 180.0, 200.0],
    "projected_revenue_growth": [0.10, 0.08, 0.06, 0.05, 0.04],
    "historical_ebitda_margin": [0.25, 0.25, 0.25],
    "projected_ebitda_margin": [0.25, 0.26, 0.27, 0.28, 0.28],
    "capex_percent_revenue": 0.04,
    "nwc_percent_revenue": 0.12,
    "depreciation_percent_revenue": 0.03
  },
  "collateral": {
      "cash_equivalents": 20.0,
      "accounts_receivable": 30.0,
      "inventory": 40.0,
      "ppe": 100.0,
      "intangibles": 50.0
  }
}
```

### 2. Run the Analysis (Python)

```python
from core.financial_suite.context_manager import ContextManager

# Initialize
manager = ContextManager(context_path="analysis_context.json")

# Run Workstream (Solver -> Waterfall -> Report)
results = manager.run_workstream()

# Access Results
ev = results['solver']['valuation']['enterprise_value']
irr = results['metrics']['irr']
rating = results['solver']['metrics']['rating']

print(f"EV: ${ev:,.2f}")
print(f"IRR: {irr*100:.1f}%")
print(f"Regulatory Rating: {rating}")

# Export Report
manager.export_report("final_report.md")
```

## Module Overview

### Interactive DCF Engine
Calculates Free Cash Flow to Firm (FCFF) and Enterprise Value. Supports dynamic WACC updates based on capital structure changes.

### VC Sponsor Module
Handles complex capital stacks, including Preferred Equity waterfall logic and IRR/MOIC calculations.

### Credit Challenge Module
Implements "Hybrid PD" modeling (Merton Structural & Logistic Regression). Generates regulatory ratings based on PD, LTV, and FCCR.

### Reporting
Automatically generates Markdown reports with Sensitivity Tables for:
1.  **Expected PD Matrix:** Sensitivity of PD to Margin and SOFR.
2.  **Downside PD Sensitivity:** Sensitivity of PD to Revenue Contraction and Asset Volatility.
