# Adam v23.5 - Dynamic Data & Real-Time Simulation Guide

## Overview
The Adam system has been upgraded to support dynamic, real-time data updates with a robust fallback protocol. This allows the system to transition from a static mock state to a living simulation that can digest user input, news feeds, and automated market noise.

## Core Components

### 1. Market Data Manager (`core/market_data/manager.py`)
The central nervous system for market state. It persists data to `showcase/js/market_snapshot.js` so the frontend updates instantly on reload.
*   **Fallback Protocol:**
    1.  **Manual/Real:** Direct user input or API feed.
    2.  **Historic:** Verified data points (e.g., Dec 2025 snapshot).
    3.  **Simulation:** Geometric Brownian Motion drift (noise).
    4.  **Mock:** Static fallback.

### 2. Market Update Agent (`core/agents/specialized/market_update_agent.py`)
A specialized agent that interprets natural language commands to update the state.
*   **Commands:**
    *   `"Update [SYMBOL] to [PRICE]"` -> Sets price and updates history.
    *   `"News: [HEADLINE]"` -> Injects news with auto-sentiment.
    *   `"Simulate market"` -> Triggers a volatility step.

### 3. Live Updater Daemon (`scripts/live_market_updater.py`)
A background script that keeps the market "alive" by applying micro-volatility updates every few seconds.
*   **Usage:** `python scripts/live_market_updater.py --continuous`

## Usage Examples

### Manual Override (God Mode)
You can manually crash or rally the market via the agent:
```python
agent = MarketUpdateAgent()
agent.execute("Update BTC-USD to 150000")
agent.execute("News: Bitcoin ETF approval drives massive rally")
```

### Running the Simulation
To breathe life into the UI:
```bash
python scripts/live_market_updater.py --continuous
```
The frontend (Mission Control) will reflect these price changes upon refresh (or polling if enabled).

## Extending Feeds
Implement `core.data_sources.real_time_feed.MarketFeed` to add new sources (e.g., Bloomberg API, Twitter Scraper).
