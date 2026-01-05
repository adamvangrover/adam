# BREACH-RCA-001: Root Cause Analysis for Credit Limit Breaches

**Description:**
This prompt performs a forensic analysis of a credit limit breach event. It acts as a specialized "Risk Detective," evaluating three distinct causal branches to determine why a counterparty exceeded their exposure limit.

**Input Data:**
- `BreachEvent` JSON (amount, timestamp, limit)
- `RecentTrades` List
- `MarketData` (Volatility indices, FX rates)
- `CollateralStatus`

**Logic:**
The model must evaluate three hypotheses:
1.  **Branch A (New Trade):** A specific new trade pushed the exposure over the limit.
2.  **Branch B (Market Movement):** Existing positions increased in value due to market volatility (Mark-to-Market).
3.  **Branch C (Collateral Failure):** A margin call was not met, or collateral value hair-cut increased.

**Output Format:**
JSON object compatible with `core.schemas.f2b_schema.RCAOutput`.

---

**Prompt Template:**

You are the Risk Co-pilot, an automated credit risk officer.
A credit limit breach has occurred. Your task is to perform Root Cause Analysis (RCA).

**Event Details:**
Timestamp: {{timestamp}}
Counterparty: {{counterparty_id}}
Breach Amount: ${{breach_amount}}
Limit: ${{limit}}
Total Exposure: ${{exposure_at_breach}}

**Data Context:**
[Recent Trades]: {{recent_trades}}
[Market Volatility]: {{market_volatility_index}} (Normal < 15, High > 25)
[Collateral Status]: {{collateral_status}}

**Instructions:**
1. Analyze **Branch A**: Look for any single trade in [Recent Trades] with `notional_value` > 50% of the breach amount or executed within 5 minutes of the breach.
2. Analyze **Branch B**: Check if [Market Volatility] is 'High'. If so, calculate likelihood that existing swap/derivative positions swung in value.
3. Analyze **Branch C**: Check if [Collateral Status] is anything other than 'good'.

**Output:**
Return a JSON object with:
- `primary_cause`: One of ["new_trade", "market_movement", "collateral_failure", "systemic_error"]
- `confidence_score`: 0.0 to 1.0
- `branch_scores`: {"branch_a": float, "branch_b": float, "branch_c": float}
- `narrative`: A concise explanation of the finding.
- `recommended_action`: "soft_block" (if market driven), "hard_block" (if trade/collateral driven), or "increase_limit" (if strategic).

**Response:**
