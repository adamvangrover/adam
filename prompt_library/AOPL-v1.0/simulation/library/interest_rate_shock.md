# Crisis Simulation Library: Interest Rate Shock Scenarios

This library provides a set of user-ready scenarios focused on **Interest Rate Shocks**. These scenarios can be used as the `{{USER_SCENARIO_INPUT}}` in the main `crisis_simulation.md` prompt.

---

### Scenario IRS-001: Aggressive Central Bank Tightening

**Description:** The central bank, facing persistent high inflation, announces an unexpected 150 basis point increase in the federal funds rate. This is the largest single increase in over two decades. Financial markets react violently, with equity indices dropping sharply and bond yields soaring.

**Potential Primary Impact Nodes:**
*   **R-FIN-02 (Market Risk):** Immediate, severe repricing of assets.
*   **R-FIN-03 (Liquidity Risk):** Corporate and institutional borrowers face a sudden spike in short-term funding costs, leading to a scramble for cash.
*   **R-FIN-04 (Credit Risk):** Companies with high levels of variable-rate debt are immediately under financial stress.

---

### Scenario IRS-002: Sovereign Debt Crisis Contagion

**Description:** A major developed economy signals a potential default on its sovereign debt due to unsustainable interest payments. This triggers a global "flight to safety," causing borrowing costs for our organization to skyrocket as lenders demand higher risk premiums.

**Potential Primary Impact Nodes:**
*   **R-FIN-05 (Counterparty Risk):** Exposure to the defaulting sovereign's bonds becomes worthless.
*   **R-FIN-03 (Liquidity Risk):** Access to international credit markets is severely curtailed.
*   **R-STR-01 (Strategic Risk):** Long-term strategic projects reliant on external financing are now unviable.

---

### Scenario IRS-003: Inverted Yield Curve Recession Signal

**Description:** The yield curve inverts sharply, with short-term debt instruments yielding significantly more than long-term ones. This is widely interpreted by economists and market participants as a strong predictor of an imminent, deep recession. Business and consumer confidence plummets.

**Potential Primary Impact Nodes:**
*   **R-STR-02 (Market Position Risk):** Forecasted demand for products/services collapses, leading to inventory overhang and revised revenue projections.
*   **R-FIN-04 (Credit Risk):** The creditworthiness of our commercial and retail customers deteriorates rapidly.
*   **R-OPS-01 (Operational Risk):** Pressure to cut costs leads to budget freezes, impacting critical operational functions.

---

### Scenario IRS-004: Foreign Exchange Shock from Rate Divergence

**Description:** Our home country's central bank holds rates steady while a major trading partner's central bank aggressively hikes its rates. This divergence causes our domestic currency to depreciate by 20% in a single week, dramatically increasing the cost of imported raw materials and components.

**Potential Primary Impact Nodes:**
*   **R-SCM-01 (Supply Chain Risk):** Key suppliers who invoice in the foreign currency may refuse to ship goods without immediate price adjustments.
*   **R-FIN-02 (Market Risk):** Hedging instruments designed to protect against currency fluctuations may fail or prove insufficient.
*   **R-FIN-01 (Financial Reporting Risk):** Massive, unexpected foreign exchange losses must be reported, impacting earnings and shareholder confidence.
