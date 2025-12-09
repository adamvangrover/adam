from typing import Dict, Any, Callable
from core.pricing_engine import PricingEngine
from core.market_data import MarketDataService
from core.execution_router import ExecutionRouter
from core.risk_engine import RiskEngine
from core.strategy import StrategyManager
from core.memory import MemoryEngine
from core.family_office import FamilyOfficeService

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

        self.tools = self._register_tools()

    def _register_tools(self) -> Dict[str, Callable]:
        return {
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
            "aggregate_family_risk": self.fo_service.aggregate_family_risk
        }

    def invoke(self, tool_name: str, **kwargs) -> Any:
        """Invoke a tool by name."""
        if tool_name not in self.tools:
            raise ValueError(f"Tool {tool_name} not found.")

        tool_func = self.tools[tool_name]
        try:
            return tool_func(**kwargs)
        except Exception as e:
            return {"error": str(e)}

    def get_tool_definitions(self):
        """Return schema definitions (placeholder for loading tools.json)."""
        import json
        import os
        base_dir = os.path.dirname(os.path.abspath(__file__))
        with open(os.path.join(base_dir, "tools.json"), "r") as f:
            return json.load(f)
