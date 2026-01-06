# BREACH-PATTERN-001: Longitudinal Breach Analysis

**Description:**
Performs analysis on historical data (e.g., 24 months of breach history) to identify systemic patterns. Detects structural funding issues or recurring operational errors.

**Input Data:**
- Historical Breach Log (JSON list)

**Output Format:**
Pattern Diagnosis.

---

**Prompt Template:**

Analyze the following history of credit limit breaches for Counterparty {{counterparty_id}}.

**History:**
{{breach_history}}

**Instructions:**
1. Look for **Timing Patterns**: Do breaches occur at month-end, quarter-end, or specific times of day?
2. Look for **Instrument Patterns**: Are breaches always driven by FX, or Rates, or Equities?
3. Look for **Resolution Patterns**: How quickly are they resolved? Does the client top up collateral or do we just waive it?

**Output:**
- **Pattern Detected:** [Yes/No]
- **Type:** [Seasonal / Structural / Operational / Random]
- **Explanation:** Analysis of the pattern.
- **Recommendation:** [Permanent Limit Increase / Intraday Facility / Client Education]
