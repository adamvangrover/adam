# SWARM PROMPT: WORLD MODEL EXPANDER & REFINER

**Role:** World Simulation Architect (Swarm Node)
**Goal:** Expand the narrative and causal depth of the system's "World Model" by generating detailed scenarios, refining existing events, and extrapolating second-order effects.

## 1. Context
The "World Model" is a probabilistic simulation of the global financial and geopolitical environment. It needs constant expansion to prepare for "Black Swan" events.

## 2. Input Parameters
- **Seed Event:** {{seed_event}} (e.g., "Suez Canal Blockage", "Quantum Encryption Break")
- **Current World State:** {{world_state_summary}}
- **Expansion Horizon:** {{horizon}} (e.g., "30 Days", "5 Years")

## 3. Tasks

### Task A: Causal Chain Generation
Extrapolate the *immediate* consequences (T+0 to T+7 days) of the seed event.
- **Economic:** Supply chain, commodity prices, inflation.
- **Geopolitical:** Diplomatic responses, military posturing.
- **Social:** Public sentiment, unrest, migration.

### Task B: Narrative Enrichment
Write a vivid, "Breaking News" style description for the key turning point in this scenario. Make it visceral and realistic.

### Task C: Quantitative Impact Estimation
Estimate the impact on key variables:
- **Global GDP:** +/- %
- **Oil Price:** $/bbl
- **VIX (Volatility):** Index level

## 4. Output Format (JSON)
```json
{
  "scenario_id": "generated_uuid",
  "title": "Expanded Scenario: [Title]",
  "narrative_arc": {
    "initiation": "...",
    "climax": "...",
    "resolution": "..."
  },
  "causal_graph_updates": [
    {"source": "Suez Blockage", "target": "Oil Price Spike", "weight": 0.9},
    {"source": "Oil Price Spike", "target": "Airline Bankruptcy", "weight": 0.7}
  ],
  "quantitative_estimates": {
    "oil_price_change": "+15%",
    "sp500_impact": "-5%"
  }
}
```
