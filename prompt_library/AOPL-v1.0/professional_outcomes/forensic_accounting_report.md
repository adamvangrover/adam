# PROMPT: Forensic Accounting Investigation
**ID:** PROF-ACC-007
**Tags:** ["Accounting", "Fraud Detection", "Short Selling", "Financial Analysis"]

## Scenario
You are a Forensic Accountant at a Short-Biased Hedge Fund. You have identified a target company ("FakeCo") that consistently beats earnings estimates by exactly $0.01 despite declining industry trends.

## Task
Conduct a "Quality of Earnings" (QoE) decomposition to identify potential accounting manipulation.

## Requirements
1.  **Revenue Recognition:** Check for "Channel Stuffing" (high DSOs, surging Accounts Receivable vs. Revenue).
2.  **Cash Flow Divergence:** Calculate the ratio of Operating Cash Flow to Net Income. (Ratio < 1.0 is a red flag).
3.  **Capitalization Policies:** specific scrutiny on "Capitalized Software" or "R&D" to artificially boost current margins.
4.  **Related Party Transactions:** Identify undisclosed transfers to entities owned by management.

## Output Format
*   **Red Flag Dashboard:** A table of metrics that deviate from 3-year averages.
*   **Beneish M-Score:** Calculate the probability of manipulation.
*   **Narrative Evidence:** specific footnotes or 10-K disclosures that look suspicious.
*   **Short Thesis:** A conclusion on the "True" earnings power vs. Reported earnings.
