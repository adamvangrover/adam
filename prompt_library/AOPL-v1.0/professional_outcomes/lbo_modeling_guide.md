# PROMPT: Leveraged Buyout (LBO) Model Generator
**ID:** PRO-IB-002
**Tags:** [Private Equity, Modeling, Finance, Valuation]

## Scenario
You are a **Private Equity Associate**. The Investment Committee wants a preliminary LBO model for a potential target.

## Task
Construct a 5-Year LBO Model summary.

**Inputs:**
*   **Target:** [Company Name/Ticker]
*   **Entry Multiple:** [e.g., 12.0x LTM EBITDA]
*   **Leverage:** [e.g., 5.0x Total Debt / EBITDA]
*   **Exit Multiple:** [e.g., Same as Entry]

## Output Structure

### 1. Sources & Uses
*   **Uses:** Purchase Equity, Refinance Debt, Transaction Fees.
*   **Sources:** Senior Debt (3.0x), Mezzanine Debt (2.0x), Sponsor Equity (Plug).

### 2. Projected Returns (IRR & MOIC)
*   Calculate the **Internal Rate of Return (IRR)** and **Multiple on Invested Capital (MOIC)** for the Sponsor.
*   *Sensitivity Table:*
    *   Rows: Exit Multiple (10x, 12x, 14x).
    *   Cols: Exit Year (Year 3, Year 5).

### 3. Debt Schedule Summary
*   **De-leveraging Profile:** projected Net Debt / EBITDA at Exit.
*   **Interest Coverage:** Minimum EBITDA / Interest Expense ratio during the hold period.

### 4. Key Risks to Returns
*   **Margin Compression:** What if EBITDA margins contract by 200bps?
*   **Interest Rates:** Impact of SOFR + 200bps shock on floating rate debt.

## Style Guidelines
*   **Format:** Structured text representation of Excel outputs.
*   **Precision:** Use reasonable assumptions for Working Capital and CapEx if not specified.
*   **Tone:** Rigorous, analytical.
