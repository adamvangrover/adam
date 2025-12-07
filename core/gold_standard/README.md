# Adam Gold Standard Financial Toolkit

The Democratization of Institutional-Grade Data Engineering.

## Executive Summary

This toolkit transforms the Adam repository into a high-performance financial data engine. It implements a "Data Lakehouse" architecture using `yfinance` for ingestion and `Apache Parquet` for storage. The system is designed to serve two distinct mandates:
1.  **Intra-Day Trading Engine**: High-frequency data (1m), low latency, mean reversion strategies.
2.  **Robo-Advisor**: Long-term data, MPT/Black-Litterman optimization, risk simulation.

## Architecture

### 1. The Discovery Layer (`discovery.py`)
-   **Real Search Pull**: Dynamically discovers new assets using `yf.Search`.
-   **Universe Snapshots**: Mitigates survivorship bias by saving point-in-time universe compositions.
-   **Taxonomy**: Classifies assets using `yf.Sector` and `yf.Industry`.

### 2. The Ingestion Engine (`ingestion.py`)
-   **Reliability**: Implements exponential backoff and batching to respect API limits.
-   **Hybrid Mode**: Uses `download` for history and `fast_info` for real-time L1 data.
-   **Eager Ingestion**: Automatically fetches ephemeral 1m data (7-day window).

### 3. The Storage Layer (`storage.py`)
-   **Format**: Apache Parquet (Columnar Storage).
-   **Partitioning**:
    -   *Intra-day*: `ticker=AAPL/year=2024/` (Optimized for deep history lookup).
    -   *Daily*: `region=US/year=2024/` (Optimized for cross-sectional analysis).
-   **Immutability**: Appends new data; edits only via Reconciliation Agent.

### 4. Quality Assurance (`qa.py`)
-   **Schema Validation**: Uses `Pandera` to enforce data types and constraints.
-   **Calendar Integration**: Uses `pandas_market_calendars` to distinguish missing data from market holidays.

### 5. Trading Toolkit (`trading/`)
-   **Cleaning**: Handles zero-volume bars and imputes missing data.
-   **Strategy**: Implements Mean Reversion using Z-Score methodology.

### 6. Robo-Advisor Toolkit (`advisory/`)
-   **MPT**: Mean-Variance Optimization using `PyPortfolioOpt`.
-   **Black-Litterman**: Bayesian integration of market views and equilibrium returns.
-   **Risk**: Historical Simulation (VaR, CVaR).

## Quick Start

### Installation
Ensure the requirements are installed:
```bash
pip install yfinance pyarrow pandera pandas_market_calendars PyPortfolioOpt
```

### Example Usage
```python
from core.gold_standard.ingestion import IngestionEngine
from core.gold_standard.storage import StorageEngine
from core.gold_standard.discovery import DiscoveryAgent

# Initialize
storage = StorageEngine()
ingestor = IngestionEngine(storage)
discovery = DiscoveryAgent()

# 1. Discover Universe
tech_tickers = discovery.get_sector_universe("Technology")

# 2. Ingest Data
ingestor.ingest_daily_history(tech_tickers)
ingestor.ingest_intraday_eager(tech_tickers)

# 3. Load for Analysis
df = storage.load_daily()
```

## Maintenance
- **Weekly**: Run `ingest_intraday_eager` to capture 1m data.
- **Monthly**: Run `run_discovery_cycle` to update universes.
