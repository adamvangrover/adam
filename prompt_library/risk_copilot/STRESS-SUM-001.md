# STRESS-SUM-001: Stress Test Narrative Generator

**Description:**
Generates narrative summaries of complex stress tests. Identifies top contributors to firm-wide risk and explains specific trading positions driving vulnerability.

**Input Data:**
- Stress Scenario Name (e.g. "Global Interest Rate Shock +200bps")
- Loss Distribution (by Desk/Asset Class)
- Top 5 Losers (Positions)

**Output Format:**
Executive Summary.

---

**Prompt Template:**

Write an Executive Risk Summary for the following Stress Test result.

**Scenario:** {{scenario_name}}
**Total Projected Loss:** ${{total_loss}}

**Breakdown:**
{{breakdown_table}}

**Top Drivers:**
{{top_drivers}}

**Instructions:**
1. Summarize the impact. Is this within our risk appetite?
2. Explain *why* the top drivers are losing money (e.g. "Long duration bonds suffered from rate hike").
3. Recommend hedging strategies.

**Output:**
Professional Risk Memo format.
