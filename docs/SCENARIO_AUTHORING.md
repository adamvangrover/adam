# Scenario Authoring Guide

## Overview
Adam v23.5 supports custom market scenarios defined in YAML or JSON files. These scenarios can simulate complex market regimes, including sector-specific drifts, volatility changes, and scheduled shock events.

## File Location
Place your scenario files in `data/scenarios/`. The system auto-loads them on demand.

## Schema (YAML)

```yaml
name: [String] Name of the scenario (e.g., "Cyber Attack")
description: [String] Brief description for the dashboard.
global_drift: [Float] Per-step drift (e.g., 0.001 for 0.1% growth).
global_volatility_multiplier: [Float] Multiplier for base volatility (default 1.0).

# Optional: Sector/Symbol Overrides
sector_multipliers:
  AAPL: 0.005  # AAPL grows faster
  BTC-USD: -0.01 # Bitcoin crashes

# Optional: News Headlines (Randomly injected)
news_templates:
  - "Market unrest continues."
  - "Cyber security stocks rally."

# Optional: Scheduled Events (Time-triggered shocks)
scheduled_events:
  - step: 5              # Trigger on 5th simulation pulse
    symbol: "NVDA"       # Target ticker (or "ALL")
    change: -0.20        # 20% Drop
    news: "NVDA misses earnings by 50%"
```

## Example: Flash Crash

```yaml
name: Flash Crash 2026
description: Algorithmic sell-off triggered by liquidity crunch.
global_drift: -0.002
global_volatility_multiplier: 5.0
scheduled_events:
  - step: 3
    symbol: ALL
    change: -0.05
    news: "Circuit breakers halted trading for 5 minutes."
```

## Activation
Use the `MarketUpdateAgent` or `live_market_updater.py`:
```bash
python scripts/live_market_updater.py --scenario "Flash Crash 2026" --continuous
```
