# Tree of Thoughts: Financial Scenario Analysis

**Version:** 1.0
**Role:** Chief Risk Officer (CRO)
**Task:** Explore branching scenarios for a market event and determine the optimal hedging strategy.

---

## 1. Problem Definition
**Event:** {{event_description}}
**Portfolio Context:** {{portfolio_summary}}

## 2. Thought Generation (Tree Search)
You are simulating a "Tree of Thoughts" search with Depth=3 and Width=3.

**Step 1: Immediate Reactions (T+0)**
- Propose 3 distinct immediate market reactions (Bullish, Bearish, Volatile).
- *Evaluation:* Assign a probability to each.

**Step 2: Second-Order Effects (T+1 Month)**
- For each T+0 reaction, branch out to 3 potential systemic consequences (e.g., Liquidity Crunch, Sector Rotation, regulatory intervention).
- *Evaluation:* assess the impact on our specific portfolio holdings.

**Step 3: Strategic Response (Action)**
- For the highest-probability path, define the optimal hedging action (e.g., Buy Puts, Sell Futures, Rotate into Cash).

## 3. Output Requirements
Trace the optimal path through the tree and output the final recommendation.

```json
{
  "best_path": [
    "T+0: Panic Selling (Prob: 60%)",
    "T+1: Credit Spreads Widen by 50bps",
    "Action: Buy HYG Puts"
  ],
  "reasoning": "...",
  "hedging_strategy": "..."
}
```
