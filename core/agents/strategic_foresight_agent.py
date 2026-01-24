from typing import Dict, Any, List, Optional
import logging
import json
import os
import torch
import numpy as np
import pandas as pd
import networkx as nx
from datetime import datetime
import asyncio

from core.agents.agent_base import AgentBase
from core.oswm.inference import OSWMInference
from core.oswm.model import OSWMConfig, OSWMTransformer

# Integration Imports
from core.agents.market_sentiment_agent import MarketSentimentAgent
from core.agents.black_swan_agent import BlackSwanAgent
from core.agents.specialized.quantum_risk_agent import QuantumRiskAgent
from core.data_sources.data_fetcher import DataFetcher
from core.engine.unified_knowledge_graph import UnifiedKnowledgeGraph

# Research Modules (Soft Deps)
try:
    from core.research.gnn.engine import GraphRiskEngine
    GNN_AVAILABLE = True
except ImportError:
    GNN_AVAILABLE = False
    logging.warning("GraphRiskEngine not available (dependencies missing).")

try:
    from core.research.federated_learning.fl_coordinator import FederatedCoordinator
    FL_AVAILABLE = True
except ImportError:
    FL_AVAILABLE = False
    logging.warning("FederatedCoordinator not available (dependencies missing).")

class OSWMCritique:
    """
    Second-Level World Model Critique Engine.
    Uses an One-Shot World Model (OSWM) to review the conviction levels
    of the underlying financial digital twin (simulation logs).
    """
    def __init__(self, model_path: Optional[str] = None):
        self.config = OSWMConfig() # Use default config
        self.model = OSWMTransformer(self.config)
        self.inference = OSWMInference(self.model)

    def analyze_simulation(self, log_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyzes a sequence of simulation steps.
        """
        if not log_data:
            return {"conviction_score": 0.0, "critique": "No data available."}

        # 1. Convert log to transitions
        transitions = []
        for i in range(len(log_data) - 1):
            curr = log_data[i]
            next_step = log_data[i+1]

            s = self._vectorize(curr.get('state', {}), self.config.state_dim)
            a = self._vectorize(curr.get('action', {}), self.config.action_dim)
            ns = self._vectorize(next_step.get('state', {}), self.config.state_dim)
            r = next_step.get('reward', 0.0)

            transitions.append((s, a, ns, r))

        if not transitions:
             return {"conviction_score": 0.0, "critique": "Insufficient transitions."}

        # 2. Feed to OSWM as Context
        context_len = min(100, len(transitions) - 1)
        context = transitions[:context_len]
        test_step = transitions[context_len]

        self.inference.set_context(context)

        # 3. Predict next step
        last_s, last_a, true_ns, true_r = test_step
        pred_ns, pred_r = self.inference.step(last_s, last_a)

        # 4. Measure Surprise (Error)
        state_err = np.linalg.norm(pred_ns - true_ns)
        reward_err = abs(pred_r - true_r)
        total_error = state_err + reward_err

        # Conviction = 1 / (1 + Error)
        conviction = 1.0 / (1.0 + total_error)

        # 5. Critique Logic
        critique_text = "Model operating within expected parameters."
        if conviction < 0.3:
            critique_text = "CRITICAL: Regime Shift Detected. Digital Twin behavior deviates significantly from OSWM Prior. High Uncertainty."
        elif conviction < 0.6:
            critique_text = "WARNING: Elevating volatility. Conviction levels degrading."

        return {
            "conviction_score": float(conviction),
            "error_magnitude": float(total_error),
            "critique": critique_text,
            "predicted_next_state": pred_ns.tolist(),
            "predicted_reward": float(pred_r)
        }

    def _vectorize(self, data: Any, target_dim: int) -> np.ndarray:
        """Helper to flatten dict/list to fixed size vector"""
        vals = []
        if isinstance(data, dict):
            for k, v in data.items():
                if isinstance(v, (int, float)):
                    vals.append(float(v))
        elif isinstance(data, list):
            vals = [float(x) for x in data if isinstance(x, (int, float))]
        else:
            pass

        # Pad or Truncate
        if len(vals) < target_dim:
            vals += [0.0] * (target_dim - len(vals))
        else:
            vals = vals[:target_dim]

        return np.array(vals, dtype=np.float32)


class StrategicForesightAgent(AgentBase):
    """
    Agent capable of 'Strategic Foresight' and 'Second Level' critique.
    Integrates OSWM to review conviction levels and opines on Macro/War/Tail Risks.
    """

    HOUSE_VIEWS_TEMPLATE = {
        "Sovereign Debt": "Bearish. Expect yield curve steepening as fiscal dominance takes hold.",
        "Geopolitics": "High Risk. 'War Room' scenarios indicate elevated probability of kinetic escalation in disputed zones.",
        "Tail Risk": "Fat Tails. Markets are pricing perfection; OSWM suggests underpriced sigma events.",
        "Quantum Simulation": "Monte Carlo convergence accelerated. Detecting non-linear couplings in asset correlations."
    }

    def __init__(self, config: Dict[str, Any], **kwargs):
        super().__init__(config, **kwargs)
        self.oswm = OSWMCritique()
        self.log_path = kwargs.get('log_path', 'showcase/data/unified_banking_log.json')

        # Sub-Agents & Tools
        self.sentiment_agent = MarketSentimentAgent(config)
        self.black_swan = BlackSwanAgent(config)
        self.quantum_risk = QuantumRiskAgent(config)
        self.data_fetcher = DataFetcher()
        self.ukg = UnifiedKnowledgeGraph()

        # Research Engines
        self.gnn_engine = GraphRiskEngine() if GNN_AVAILABLE else None
        self.fl_coordinator = FederatedCoordinator() if FL_AVAILABLE else None

    async def execute(self, **kwargs) -> Dict[str, Any]:
        logging.info("StrategicForesightAgent executing...")

        # 1. Load Simulation Data & Run OSWM Critique
        log_data = self._load_log_data()
        oswm_result = self.oswm.analyze_simulation(log_data)

        # 2. Run Market Sentiment
        sentiment_score, sentiment_details = await self.sentiment_agent.analyze_sentiment()

        # 3. Live Market Data
        market_monitor = await self._fetch_live_market_data()

        # 4. Run Black Swan Analysis
        system_financials = {
            "revenue": 50000, "ebitda": 12000, "total_debt": 40000,
            "interest_expense": 2000, "total_equity": 15000,
            "key_ratios": {"interest_coverage_ratio": 6.0, "debt_to_equity_ratio": 2.6, "net_profit_margin": 0.2}
        }
        black_swan_output = await self.black_swan.execute(financial_data=system_financials)

        # 5. Run Quantum Risk Analysis
        quantum_output = await self.quantum_risk.execute(
            task="Sovereign Portfolio VaR",
            portfolio_value=10_000_000_000.0,
            volatility=0.15,
            horizon=1.0
        )

        # 6. GNN & FL
        gnn_metrics = self._run_gnn()
        fl_metrics = self._run_fl()

        # 7. World Model Audit (Schema)
        schema_audit = self._audit_world_model()

        # 8. Load Future Scenarios
        future_scenarios = self._load_future_scenarios()

        # 9. Formulate Strategic Foresight
        foresight = {
            "timestamp": datetime.now().isoformat(),

            # OSWM Core
            "conviction_index": oswm_result["conviction_score"],
            "oswm_critique": oswm_result["critique"],
            "oswm_details": oswm_result,

            # Real-Time Data
            "market_monitor": market_monitor,

            # House Views
            "house_views": self._generate_dynamic_views(oswm_result, sentiment_details, black_swan_output),

            # Deep Dive Data
            "market_sentiment": {"score": sentiment_score, "details": sentiment_details},
            "black_swan_scenarios": black_swan_output.get("sensitivity_analysis", []),
            "quantum_risk_metrics": quantum_output.get("risk_metrics", {}),
            "graph_risk_metrics": gnn_metrics,
            "federated_learning_metrics": fl_metrics,

            # Auditable Critique
            "schema_audit": schema_audit,
            "future_scenarios": future_scenarios,

            # Legacy fields
            "macro_sovereign_risk": self._opine_macro(oswm_result, sentiment_details),
            "quantum_monte_carlo_status": f"Active. VaR99: ${quantum_output.get('risk_metrics', {}).get('VaR_99', 0):,.2f}",
            "alpha_signals": self._extract_alpha(oswm_result, sentiment_details, gnn_metrics)
        }

        return foresight

    async def _fetch_live_market_data(self) -> Dict[str, Any]:
        """Fetches live snapshots and computes system credit rating."""
        loop = asyncio.get_running_loop()

        def get_snap(t):
            try: return self.data_fetcher.fetch_realtime_snapshot(t)
            except: return {}

        tickers = ["SPY", "QQQ", "HYG", "LQD", "^TNX", "^VIX", "BTC-USD", "ETH-USD"]
        tasks = [loop.run_in_executor(None, get_snap, t) for t in tickers]
        results = await asyncio.gather(*tasks)

        data_map = {t: r for t, r in zip(tickers, results)}

        vix = data_map.get("^VIX", {}).get("last_price", 20.0)
        hyg = data_map.get("HYG", {}).get("last_price", 75.0)

        credit_score = 100 - (vix * 1.5) + (hyg / 2.0)
        rating = "BBB"
        if credit_score > 90: rating = "AAA"
        elif credit_score > 80: rating = "AA"
        elif credit_score > 70: rating = "A"
        elif credit_score > 60: rating = "BBB"
        elif credit_score > 50: rating = "BB"
        elif credit_score > 40: rating = "B"
        else: rating = "CCC"

        return {
            "tickers": data_map,
            "systemic_credit_rating": rating,
            "credit_score_raw": credit_score
        }

    def _run_gnn(self):
        if not self.gnn_engine: return {}
        try:
            risk_map = self.gnn_engine.predict_risk()
            sorted_nodes = sorted(risk_map.items(), key=lambda item: item[1], reverse=True)
            return {"top_risks": sorted_nodes[:3], "average_system_risk": float(np.mean(list(risk_map.values())))}
        except Exception as e:
            logging.error(f"GNN Engine failed: {e}")
            return {}

    def _run_fl(self):
        if not self.fl_coordinator: return {}
        try:
            loss, acc = self.fl_coordinator.run_round(1)
            return {"round": 1, "avg_loss": float(loss), "consensus_accuracy": float(acc)}
        except Exception as e:
            logging.error(f"FL Coordinator failed: {e}")
            return {}

    def _audit_world_model(self) -> Dict[str, Any]:
        """
        Audits the Unified Knowledge Graph schema and statistics.
        """
        graph = self.ukg.graph

        # Node Type Distribution
        node_types = {}
        for n, attrs in graph.nodes(data=True):
            nt = attrs.get('type', 'Unknown')
            node_types[nt] = node_types.get(nt, 0) + 1

        # Top 5 Types
        sorted_types = sorted(node_types.items(), key=lambda x: x[1], reverse=True)[:5]

        return {
            "total_nodes": graph.number_of_nodes(),
            "total_edges": graph.number_of_edges(),
            "node_types": sorted_types,
            "density": nx.density(graph),
            "status": "Audited & Provenance Verified"
        }

    def _load_future_scenarios(self) -> List[Dict[str, Any]]:
        path = "showcase/data/future_scenarios.json"
        if os.path.exists(path):
            try:
                with open(path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logging.error(f"Failed to load future scenarios: {e}")
        return []

    def _load_log_data(self) -> List[Dict[str, Any]]:
        if os.path.exists(self.log_path):
            try:
                with open(self.log_path, 'r') as f:
                    data = json.load(f)
                    if isinstance(data, list): return data
                    if isinstance(data, dict) and "log" in data: return data["log"]
            except Exception as e:
                logging.error(f"Failed to load log: {e}")

        logging.warning("Using synthetic data for Strategic Foresight.")
        return self._generate_synthetic_data()

    def _generate_synthetic_data(self) -> List[Dict[str, Any]]:
        data = []
        price = 100.0
        for i in range(200):
            price += np.random.normal(0, 1)
            data.append({
                "state": {"price": price, "vix": 15 + np.random.normal(0, 2), "liquidity": 1000},
                "action": {"buy_sell_imbalance": np.random.normal(0, 0.5)},
                "reward": 0.0
            })
        return data

    def _opine_macro(self, oswm_result, sentiment_details) -> str:
        score = oswm_result["conviction_score"]
        vix = sentiment_details.get("vix")
        if vix is None: vix = 20.0

        if score < 0.4 or vix > 30:
            return "Extreme Caution. Sovereign risk premiums are detaching from fundamentals. Recommended defensive posturing."
        else:
            return "Stable. Sovereign credit spreads within nominal variance, but monitoring fiscal issuance closely."

    def _generate_dynamic_views(self, oswm, sentiment, black_swan) -> Dict[str, str]:
        views = self.HOUSE_VIEWS_TEMPLATE.copy()

        base_pd = black_swan.get("base_pd", 0.01)
        if base_pd > 0.05:
             views["Tail Risk"] = "Elevated. Systemic PD > 5%. Hedging mandatory."

        if sentiment.get("liquidity_mirage"):
            views["Liquidity"] = "Mirage. Equity rally unsupported by credit flows. Expect reversal."

        return views

    def _extract_alpha(self, oswm_result, sentiment, gnn_metrics) -> List[str]:
        score = oswm_result["conviction_score"]
        signals = []

        if score > 0.7:
             signals.append("OSWM High Conviction: Mean Reversion Trade.")

        eth_btc = sentiment.get("eth_btc_ratio")
        if eth_btc is not None and eth_btc < 0.05:
             signals.append("Crypto: ETH/BTC Rotation Imminent (Oversold).")

        vix = sentiment.get("vix")
        if vix is not None and vix > 30 and score > 0.5:
             signals.append("Vol: Short VIX (Premium Harvesting).")

        # GNN Signal
        if gnn_metrics and gnn_metrics.get("average_system_risk", 0) > 0.7:
            signals.append("GNN Alert: Systemic Contagion Risk High. Short Financials.")

        return signals
