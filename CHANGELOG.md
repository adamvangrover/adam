# Changelog

## v26.4 - Protocol ARCHITECT_INFINITE (Day 5)

### Jules' Log
> "I noticed we lack a true real-time data ingestion layer in the V30 architecture. The `MarketScanner` was a placeholder emitting random data, which undermined the credibility of downstream agents like `QuantitativeAnalyst` and `RiskGuardian`. I researched efficient market data polling patterns and built `MarketScanner-V2` to bridge this gap, integrating `yfinance` to provide a live pulse of the market to the Neural Mesh."

### Added
- **New Organ**: `core/v30_architecture/python_intelligence/agents/market_scanner.py` - A specialized V30 agent that fetches real-time market data (Price, Volume, Change %) for a configurable list of tickers using `yfinance`.
- **Integration**: Updated `core/v30_architecture/python_intelligence/agents/swarm_runner.py` to replace the simulated scanner with the real-time implementation.
- **Tests**: `tests/test_v30_market_scanner.py` - Unit tests verifying data fetching logic, multi-ticker support, and correct event emission.

## v26.3 - Protocol ARCHITECT_INFINITE (Day 4)

### Jules' Log
> "I noticed we lack a dedicated risk monitoring capability in the V30 architecture. While `QuantitativeAnalyst` provides signals, there was no independent oversight of portfolio exposure. I researched quantitative risk metrics and built `RiskGuardian` to bridge this gap, implementing Value at Risk (VaR), Conditional VaR (CVaR), and Volatility tracking using `numpy` and `scipy`."

### Added
- **New Organ**: `core/v30_architecture/python_intelligence/agents/risk_guardian.py` - A specialized V30 agent that calculates portfolio risk metrics (VaR, CVaR, Sharpe Ratio) in real-time.
- **Integration**: Updated `core/v30_architecture/python_intelligence/agents/swarm_runner.py` to replace the dummy `RiskGuardian` with the fully implemented agent.
- **Tests**: `tests/test_risk_guardian.py` - Unit tests verifying the mathematical accuracy of risk metrics and correct event emission.

## v26.2 - Protocol ARCHITECT_INFINITE (Day 3)

### Jules' Log
> "I noticed we lack a fundamental analysis capability in the V30 architecture. The current system relies heavily on technicals (QuantitativeAnalyst) and sentiment (NewsBot). I researched 'RAG-Augmented Financial Analysis' and have built `FundamentalAnalyst` to bridge this gap, focusing on earnings call simulation and balance sheet parsing."

### Added
- **New Organ**: `core/v30_architecture/python_intelligence/agents/fundamental_analyst.py` - A specialized V30 agent that simulates 10-K data fetching and Earnings Call analysis to calculate Intrinsic Value (DCF), Distress (Altman Z), and Quality (Pietroski F).
- **Tests**: `tests/test_fundamental_analyst_v30.py` - Unit tests verifying the financial logic and Pydantic schemas.

## v26.1 - Protocol ARCHITECT_INFINITE (Day 2)

### Jules' Log
> "I noticed we lack a real-time quantitative analysis capability in the V30 architecture. The `NewsBot` and other agents were mocked or relied on static data. I researched real-time market data integration patterns and built `QuantitativeAnalyst` to bridge this gap. This agent now fetches live market data using `yfinance` and calculates RSI, SMA, and Bollinger Bands to emit actionable `technical_analysis` signals into the Neural Mesh."

### Added
- **New Organ**: `core/v30_architecture/python_intelligence/agents/quantitative_analyst.py` - A specialized V30 agent that performs real-time technical analysis on market data (SPY, QQQ, BTC-USD, etc.).
- **Refactor**: `core/v30_architecture/python_intelligence/agents/base_agent.py` - Extracted `BaseAgent` from `swarm_runner.py` to a shared module for reusability.
- **Integration**: Updated `core/v30_architecture/python_intelligence/agents/swarm_runner.py` to include `QuantitativeAnalyst` in the active swarm.
- **Tests**: `tests/test_quantitative_analyst.py` - Unit tests verifying data fetching, indicator calculation, and packet emission using mocked `yfinance` and `NeuralMesh`.

## v26.0 - Protocol ARCHITECT_INFINITE (Day 1)

### Jules' Log
> "I noticed we lack a unified understanding of market states across our agent fleet. Our `AlgoTradingAgent` was firing blindly regardless of whether the market was trending or chopping. I researched quantitative finance patterns and found that identifying the 'Market Regime' is a critical first step for any robust strategy. I have built `MarketRegimeAgent` to bridge this gap, using Hurst Exponent and ADX to classify the market environment."

### Added
- **New Organ**: `core/agents/specialized/market_regime_agent.py` - A specialized agent that classifies market conditions into `STRONG_TREND`, `MEAN_REVERSION`, or `HIGH_VOLATILITY_CRASH_RISK` using statistical metrics.
- **Tests**: `tests/test_market_regime_agent.py` - Unit tests verifying regime classification against synthetic data patterns (Sine Wave, Linear Trend, Random Walk Explosion).

## v23.5 - Autonomous Remediation & Enhancement

### Architecture
- **Dependency Management**: Installed critical missing packages (`fastapi`, `flask`, `celery`, `statsmodels`, `semantic-kernel`, `pandera`, `neo4j`, `flask-socketio`, `flask-jwt-extended`, `flask-sqlalchemy`, `flask-cors`, `tweepy`, `pycoingecko`, `feedparser`).
- **Core Schemas**: Updated `HNASP` schema to support `ExecutionTrace` list and `Optional` fields correctly.
- **Base Agent**: Refactored `AgentBase` to improve type safety, fix `jsonLogic` signature, and handle optional `fundamental_epa`.
- **Async Migration**: Refactored `NewsBot` to use `httpx` and `asyncio` for non-blocking I/O.

### Fixes
- **Syntax**: Fixed invalid escape sequences in `core/financial_suite/modules/reporting/generator.py` and `core/risk_engine/engine.py`.
- **Tests**:
    - Created missing `core/v23_graph_engine/data_pipeline/graph.py` to fix `test_adk_data_pipeline.py`.
    - Fixed `unittest.mock` recursion error in `tests/test_interaction_loop.py`.
    - Updated `tests/test_data_retrieval_agent.py` to be async-aware and fixed dependency injection.
    - Updated `tests/test_cyclical_agents.py` to match `ReflectorAgent`'s new output structure.
    - Skipped tests requiring `torch` if not available.
- **Type Safety**: Addressed critical `mypy` errors in core components.
- **Resilience**: Added timeouts to `CatalystAgent` requests. Improved error handling in `NewsBot` for optional ML dependencies.

### Security
- Addressed `bandit` warnings regarding requests without timeout.

## [v23.5-patch-security] - 2025-05-21 (Operation Green Light)

### Security
- **Hardening**: Replaced MD5 with SHA-256 for file hashing (`core/data_processing`).
- **Web**: Disabled Flask debug mode (`services/ui_backend.py`) and enabled Jinja2 autoescape (`core/newsletter_layout`).
- **Network**: Enforced 30s timeouts on external API requests (`government_stats_api`, `market_data_api`).
- **SQL**: Added input validation to `MCPRegistry` to prevent SQL injection.

### Reliability
- **Fallback**: Implemented graceful degradation for `langgraph` in all `core/engine/*_graph.py` modules. System now boots without graph dependencies.
- **Types**: Relaxed `TypedDict` strictness (`total=False`) in `core/engine/states.py` to support partial async updates.
- **Documentation**: Added Google-style docstrings to `core/agents/agent_base.py`.
- **Fixes**: Corrected indentation syntax error in `core/vertical_risk_agent/tools/mcp_server/server2.py`.

## [Unreleased] - 2025-05-20 (Simulated)

### Fixed
- **Critical:** Resolved `ModuleNotFoundError` for `core.v23_graph_engine.data_pipeline.graph` by implementing the missing graph definition.
- **Critical:** Fixed `tests/test_interaction_loop.py` mocking logic to support `AsyncMock` and correct `AgentOrchestrator` patching.
- **Core:** Refactored `core/agents/agent_base.py` to safely handle `asyncio` loops in threaded contexts and robustify `update_persona` against None values.
- **Core:** Refactored `core/system/interaction_loop.py` to correctly inject `config` into `Echo` and `check_token_limit`, and wrap async agent calls with `asyncio.run`.
- **Core:** Fixed `core/agents/query_understanding_agent.py` to call synchronous `LLMPlugin.generate_text` instead of non-existent `get_completion`.
- **Ops:** Installed missing dependencies: `pydantic`, `flask`, `torch` (CPU), `textblob`, `langchain-community`, `json-logic`.

### Added
- `reproduce_api_error.py` script to debug API endpoint failures.
- `RemediationPlan.json` outlining future steps for 100% system integrity.

### Known Issues
- `tests/api/test_service_state.py` fails with 500 error on Adam-mini endpoint (likely optimizer logic).
- Multiple agent tests failing due to strict mocking or logic divergence in v23 transition.
