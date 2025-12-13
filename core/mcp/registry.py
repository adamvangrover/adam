# core/mcp/registry.py

from typing import Dict, Any, Callable
from core.pricing_engine import PricingEngine
from core.market_data import MarketDataService
from core.execution_router import ExecutionRouter
from core.risk_engine import RiskEngine
from core.strategy import StrategyManager
from core.memory.engine import MemoryEngine
from core.family_office import FamilyOfficeService

# New Tools
from src.core_valuation import ValuationEngine as CoreValuationEngine
from src.config import DEFAULT_ASSUMPTIONS
from core.memory.provo_graph import ProvoGraph
from core.data_processing.universal_ingestor_v2 import UniversalIngestor
from core.engine.neuro_symbolic_planner import NeuroSymbolicPlanner
import requests

class MCPRegistry:
    """
    Registry for FO Super-App MCP tools.
    Instantiates core services and exposes them as callable tools.
    """

    def __init__(self):
        self.pricing = PricingEngine()
        self.market = MarketDataService()
        self.execution = ExecutionRouter(mode="simulated")
        self.risk = RiskEngine()
        self.strategy = StrategyManager()
        self.memory = MemoryEngine() # Persists to data/personal_memory.db
        self.fo_service = FamilyOfficeService()

        # New Neuro-Symbolic Tools
        self.provo = ProvoGraph()
        self.ingestor = UniversalIngestor()
        self.planner = NeuroSymbolicPlanner()

        self.tools = self._register_tools()

    def _register_tools(self) -> Dict[str, Callable]:
        tools = {
            "price_asset": self.pricing.get_price,
            "retrieve_market_data": self.market.get_latest_quote,
            "execute_order": self.execution.execute_order,
            "calculate_risk": self.risk.calculate_portfolio_risk,
            "generate_strategy": self.strategy.generate_strategy_draft,
            "load_memory": self.memory.query_memory,
            "write_memory": self.memory.store_memory,
            "get_context": self.memory.get_context,
            "generate_ips": self.fo_service.generate_ips,
            "plan_wealth_goal": self.fo_service.plan_wealth_goal,
            "screen_deal": self.fo_service.screen_deal,
            "aggregate_family_risk": self.fo_service.aggregate_family_risk,

            # Optimized AWO Tools
            "universal_ingestor_scrub": self.ingest_url,
            "financial_engineering_dcf": self.run_dcf,
            "query_universal_memory": self.provo.get_ips,
            "log_provenance": self.provo.log_activity,
            "submit_workflow_plan": self.submit_plan
        }
        return tools

    def invoke(self, tool_name: str, **kwargs) -> Any:
        """Invoke a tool by name."""
        if tool_name not in self.tools:
            raise ValueError(f"Tool {tool_name} not found.")

        tool_func = self.tools[tool_name]
        try:
            return tool_func(**kwargs)
        except Exception as e:
            return {"error": str(e)}

    # --- Tool Wrappers ---

    def ingest_url(self, url: str) -> Dict[str, Any]:
        """Fetches a URL and scrubs it using Universal Ingestor logic."""
        try:
            # Simple fetch
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            text = response.text

            # Use Ingestor's scrubber
            from core.data_processing.universal_ingestor_v2 import GoldStandardScrubber
            clean_text = GoldStandardScrubber.clean_text(text)
            metadata = GoldStandardScrubber.extract_metadata(clean_text)
            conviction = GoldStandardScrubber.assess_conviction(clean_text, "html")

            return {
                "url": url,
                "content": clean_text[:10000], # Truncate for now
                "metadata": metadata,
                "conviction": conviction,
                "status": "Verified"
            }
        except Exception as e:
            return {"error": f"Ingestion failed: {e}"}

    def run_dcf(self, ebitda: float, capex_pct: float = 0.05, growth_rates: list = None) -> Dict[str, Any]:
        """Runs the Rust/Python Valuation Engine."""
        if growth_rates is None:
            growth_rates = [0.05, 0.04, 0.03, 0.02, 0.02]

        try:
            engine = CoreValuationEngine(
                ebitda_base=ebitda,
                capex_percent=capex_pct,
                nwc_percent=0.02,
                debt_cost=0.06,
                equity_percent=0.7
            )
            df, ev, wacc = engine.run_dcf(growth_rates)
            return {
                "enterprise_value": ev,
                "wacc": wacc,
                "projections": df.to_dict(orient="records")
            }
        except Exception as e:
            return {"error": f"Valuation failed: {e}"}

    def submit_plan(self, plan_text: str) -> Dict[str, Any]:
        """Parses and logs a natural language plan."""
        return self.planner.parse_natural_language_plan(plan_text)

    def get_tool_definitions(self):
        """Return schema definitions (placeholder for loading tools.json)."""
        import json
        import os
        base_dir = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(base_dir, "tools.json")
        if os.path.exists(path):
            with open(path, "r") as f:
                return json.load(f)
        return {}
