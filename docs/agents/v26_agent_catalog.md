# Adam v26.0 Agent Catalog

A comprehensive reference for the core agents in the system.

> **Note:** Agents are strictly bifurcated into **System 1 (Swarm)** for fast perception and **System 2 (Graph)** for deep reasoning.

## 🧠 Meta-Agents (Orchestration)

These agents manage the flow of information, routing, and high-level planning.

| Agent Name | File Path | Role |
| :--- | :--- | :--- |
| **Meta-Orchestrator** | `core/engine/meta_orchestrator.py` | The Central Nervous System. Routes queries to Swarm or Graph. |
| **Neuro-Symbolic Planner** | `core/engine/neuro_symbolic_planner.py` | The Architect. Decomposes complex goals into a DAG of tasks. |
| **MetaCognitiveAgent** | `core/agents/meta_cognitive_agent.py` | Reflects on the system's own reasoning process. |
| **DiscussionChairAgent** | `core/agents/discussion_chair_agent.py` | Moderates multi-agent debates (e.g., Risk vs. Growth). |
| **AgentForge** | `core/agents/agent_forge.py` | Dynamically spawns new agent instances based on requirements. |

## 🛡️ Risk & Compliance Agents

Specialized in quantitative and qualitative risk assessment.

| Agent Name | File Path | Role |
| :--- | :--- | :--- |
| **RiskAssessmentAgent** | `core/agents/risk_assessment_agent.py` | General credit risk scoring (PD, LGD, Recovery Rate). |
| **RiskGuardian** | `core/agents/risk_guardian.py` | Real-time portfolio risk monitoring (VaR, CVaR). |
| **CreditRiskAgent** | `core/agents/credit_risk_agent.py` | Specialized credit modeling (Merton, Z-Score). |
| **LiquidityRiskAgent** | `core/agents/liquidity_risk_agent.py` | Liquidity coverage ratio (LCR) and impact cost analysis. |
| **MarketRiskAgent** | `core/agents/market_risk_agent.py` | Market volatility and drawdown analysis. |
| **OperationalRiskAgent** | `core/agents/operational_risk_agent.py` | LDA Monte Carlo for operational loss events. |
| **GeopoliticalRiskAgent** | `core/agents/geopolitical_risk_agent.py` | Contagion analysis and supply chain disruption. |
| **RegulatoryComplianceAgent** | `core/agents/regulatory_compliance_agent.py` | Checks against regulatory frameworks (Basel III, Dodd-Frank). |
| **PolicyEnforcerAgent** | `core/agents/governance/policy_enforcer_agent.py` | Enforces "Logic as Data" rules using `jsonLogic`. |

## 📊 Financial Analysis Agents

Focused on fundamental, technical, and quantitative analysis.

| Agent Name | File Path | Role |
| :--- | :--- | :--- |
| **FundamentalAnalyst** | `core/agents/fundamental_analyst_agent.py` | DCF, 3-Statement Modeling, Ratio Analysis. |
| **FinancialModelingAgent** | `core/agents/financial_modeling_agent.py` | LBO models, M&A accretion/dilution. |
| **TechnicalAnalystAgent** | `core/agents/technical_analyst_agent.py` | Chart patterns, RSI, MACD, Bollinger Bands. |
| **QuantitativeAnalyst** | `core/agents/quantitative_risk_agent.py` | Statistical arbitrage signals and factor analysis. |
| **MarketSentimentAgent** | `core/agents/market_sentiment_agent.py` | NLP-based sentiment scoring from news and social media. |
| **MarketRegimeAgent** | `core/agents/specialized/market_regime_agent.py` | Classifies market state (Trending vs. Mean Reverting). |
| **MacroLiquidityAgent** | `core/agents/specialized/macro_liquidity_agent.py` | Analyzes global liquidity conditions. |

## ⛓️ Crypto & DeFi Agents

Specialized in blockchain and digital asset markets.

| Agent Name | File Path | Role |
| :--- | :--- | :--- |
| **CryptoAgent** | `core/agents/crypto_agent.py` | General crypto market analysis. |
| **DeFiLiquidityAgent** | `core/agents/specialized/defi_liquidity_agent.py` | Analyzes liquidity pools, IL, and yield farming. |
| **CryptoArbitrageAgent** | `core/agents/specialized/crypto_arbitrage_agent.py` | Identifies cross-exchange price spreads. |

## ⚙️ Execution & Strategy Agents

Agents that propose or simulate trades and strategies.

| Agent Name | File Path | Role |
| :--- | :--- | :--- |
| **AlgoTradingAgent** | `core/agents/algo_trading_agent.py` | Executes algorithmic trading strategies. |
| **StrategyBacktestAgent** | `core/agents/strategy_backtest_agent.py` | Backtests strategies against historical data. |
| **PortfolioOptimizationAgent**| `core/agents/portfolio_optimization_agent.py` | Mean-Variance Optimization, Efficient Frontier. |
| **QuantumPortfolioManager** | `core/agents/quantum_portfolio_manager_agent.py` | QAOA-based portfolio optimization. |

## 🔬 Specialized & Research Agents

| Agent Name | File Path | Role |
| :--- | :--- | :--- |
| **LegalAgent** | `core/agents/legal_agent.py` | Contract review, covenant extraction. |
| **NewsBot** | `core/agents/news_bot.py` | Real-time news ingestion and filtering. |
| **SNCAnalystAgent** | `core/agents/snc_analyst_agent.py` | Shared National Credit rating automation. |
| **SupplyChainRiskAgent** | `core/agents/supply_chain_risk_agent.py` | Analyzes supplier dependencies and disruption risks. |
| **ManagementAssessmentAgent**| `core/agents/specialized/management_assessment_agent.py`| Qualitatively scores management teams. |
| **CodeAlchemist** | `core/agents/code_alchemist.py` | Self-improving code generation. |

## 🧪 System 1 Swarm Workers

Fast, asynchronous agents for monitoring and perception.

*   **SentinelWorker:** Anomaly detection in numerical streams.
*   **NewsScanner:** Continuous RSS/API polling.
*   **DataFetcher:** Low-latency market data retrieval.
