# Technical Specification: Distressed Credit Pricing & Restructuring Simulation

## 1. Overview
This simulation provides a comprehensive workflow for pricing distressed credit assets, specifically focusing on LBO structures with high leverage (6-8x). It models the full capital stack, simulates default probabilities (PD) and Loss Given Default (LGD), and projects recovery rates under various restructuring scenarios.

## 2. Core Components

### 2.1 Capital Structure Model
The simulation supports a multi-tranche capital structure:
*   **Seed Capital / Equity:** First loss piece.
*   **Preferred Equity:** Priority over common, often with PIK toggles.
*   **Mezzanine Debt:** Unsecured, high coupon, often with warrants.
*   **Junior Debt (Second Lien):** Subordinated secured debt.
*   **Senior Debt (First Lien):** Top of the capital stack, secured by assets.

### 2.2 Risk Metrics
*   **Leverage Ratio:** Total Debt / EBITDA.
*   **Interest Coverage:** EBITDA / Interest Expense.
*   **PD (Probability of Default):** Derived from leverage and industry risk factors using a logistic regression curve.
*   **LGD (Loss Given Default):** Modeled based on asset coverage and collateral quality.
*   **SNC Rating:** Shared National Credit rating derivation (Pass, Special Mention, Substandard, Doubtful, Loss) based on regulatory guidelines.

### 2.3 Restructuring Engine (Waterfall)
The engine calculates the "Distributable Value" (Enterprise Value at exit/restructuring) and flows it down the capital stack:
1.  Pay Senior Debt.
2.  Pay Junior Debt (if value remains).
3.  Pay Mezzanine (if value remains).
4.  Pay Preferred (if value remains).
5.  Residual to Equity.

### 2.4 Institutional Market Flows
Simulates the flow of capital from different investor types:
*   **Distressed Funds:** Buying Junior/Mez at deep discounts.
*   **CLOs:** Holding Senior paper, potentially forced sellers on downgrade.
*   **Private Credit:** Providing "Rescue Financing" or "DIP Financing".

## 3. Simulation Logic

1.  **Initialization:** Set up the company financials (EBITDA, Growth) and initial Capital Structure.
2.  **Stress Scenario:** Apply a "shock" (e.g., EBITDA impairment, Multiple compression).
3.  **Default Check:** If Leverage > Max Threshold or Coverage < 1.0, trigger default.
4.  **Valuation:** Calculate Enterprise Value (EV) based on distressed multiple.
5.  **Waterfall:** Allocate EV to tranches.
6.  **Pricing:** Calculate the "Fair Value" of each tranche based on expected recovery and required return.

## 4. API & Data Structures

### Input Data
```json
{
  "ebitda": 100000000,
  "leverage_multiple": 7.5,
  "capital_stack": [
    {"type": "Senior", "amount": 400000000, "rate": 0.08},
    {"type": "Junior", "amount": 200000000, "rate": 0.12},
    {"type": "Mezzanine", "amount": 100000000, "rate": 0.15}
  ],
  "industry": "Industrials",
  "scenario": "Recession"
}
```

### Output Data
```json
{
  "total_debt": 700000000,
  "leverage_ratio": 7.0,
  "status": "Distressed",
  "snc_rating": "Substandard",
  "tranche_pricing": [
    {"type": "Senior", "recovery": 1.0, "price": 98.5},
    {"type": "Junior", "recovery": 0.85, "price": 65.0},
    {"type": "Mezzanine", "recovery": 0.0, "price": 5.0}
  ]
}
```
