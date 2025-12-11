# Adam Financial Framework: Gold Standard Open Source Toolkit

## Overview
This module implements the "Gold Standard" financial data architecture for the Adam system. It is designed to provide institutional-grade data engineering capabilities using open-source tools.

## Core Components

### 1. Discovery Layer (`discovery.py`)
Implements the "Real Search Pull" mechanism using `yfinance.Search`.
- **Dynamic Universe Discovery**: Moves beyond hard-coded watchlists.
- **Semantic Search**: Allows scanning for categories (e.g., "ESG ETF", "Emerging Markets").

### 2. Data Lakehouse (`lakehouse.py`)
Implements a storage layer using Apache Parquet.
- **Efficiency**: Columnar storage for fast retrieval.
- **Persistence**: Static repository of market data, updated incrementally.
- **Partitioning**: Organized by ticker and date.

### 3. Schema (`schema.py`)
Pydantic models ensuring strict type safety and data validation for all market artifacts (Path A compliance).

## Usage

```python
from core.financial_data import MarketDiscoveryAgent, DataLakehouse

# 1. Discover
agent = MarketDiscoveryAgent()
tech_tickers = agent.search_universe("Technology")

# 2. Store
lake = DataLakehouse(root_path="data/market_lakehouse")
lake.ingest(tech_tickers)
```

## Architecture
- **Ingestion**: `yfinance` (Yahoo Finance API)
- **Storage**: `pyarrow` (Apache Parquet)
- **Validation**: `pydantic`
