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
from core.product.core_valuation import FinancialEngineeringEngine
from core.memory.provo_graph import ProvoGraph
from core.data_processing.universal_ingestor import UniversalIngestor
from core.engine.neuro_symbolic_planner import NeuroSymbolicPlanner
from core.tools.universal_ingestor_mcp import UniversalIngestorMCP
from core.data_access.lakehouse_connector import LakehouseConnector
import requests
import json
import os


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
        self.memory = MemoryEngine()  # Persists to data/personal_memory.db
        self.fo_service = FamilyOfficeService()

        # New Neuro-Symbolic Tools
        self.provo = ProvoGraph()
        self.ingestor = UniversalIngestor()
        self.planner = NeuroSymbolicPlanner()
        self.ingestor_mcp = UniversalIngestorMCP()
        self.lakehouse = LakehouseConnector()

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
            "azure_ai_search": self.ingestor_mcp.execute,
            "financial_engineering_dcf": self.run_dcf,
            "financial_engineering_wacc": self.run_wacc,
            "financial_engineering_beta": self.run_beta,
            "financial_engineering_sharpe": self.run_sharpe,
            "microsoft_fabric_run_sql": self.lakehouse.execute,
            "get_asset_history": self.get_historical_data,
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
            from core.data_processing.universal_ingestor import GoldStandardScrubber
            clean_text = GoldStandardScrubber.clean_text(text)
            metadata = GoldStandardScrubber.extract_metadata(clean_text, "html")
            conviction = GoldStandardScrubber.assess_conviction(clean_text, "html")

            return {
                "url": url,
                "content": clean_text[:10000],  # Truncate for now
                "metadata": metadata,
                "conviction": conviction,
                "status": "Verified"
            }
        except Exception as e:
            return {"error": f"Ingestion failed: {e}"}

    def run_dcf(self, free_cash_flows: list, discount_rate: float, terminal_value: float) -> Dict[str, Any]:
        """Runs the Financial Engineering Engine DCF."""
        try:
            val = FinancialEngineeringEngine.calculate_dcf(free_cash_flows, discount_rate, terminal_value)
            return {"present_value": val}
        except Exception as e:
            return {"error": f"DCF Calculation failed: {e}"}

    def run_wacc(self, market_cap: float, total_debt: float, cost_of_equity: float, cost_of_debt: float, tax_rate: float) -> Dict[str, Any]:
        """Runs the Financial Engineering Engine WACC."""
        try:
            val = FinancialEngineeringEngine.calculate_wacc(
                market_cap, total_debt, cost_of_equity, cost_of_debt, tax_rate)
            return {"wacc": val}
        except Exception as e:
            return {"error": f"WACC Calculation failed: {e}"}

    def run_beta(self, asset_returns: list, market_returns: list) -> Dict[str, Any]:
        """Runs the Financial Engineering Engine Beta."""
        try:
            val = FinancialEngineeringEngine.calculate_beta(asset_returns, market_returns)
            return {"beta": val}
        except Exception as e:
            return {"error": f"Beta failed: {e}"}

    def run_sharpe(self, returns: list, risk_free_rate: float) -> Dict[str, Any]:
        """Runs the Financial Engineering Engine Sharpe Ratio."""
        try:
            val = FinancialEngineeringEngine.calculate_sharpe_ratio(returns, risk_free_rate)
            return {"sharpe_ratio": val}
        except Exception as e:
            return {"error": f"Sharpe failed: {e}"}

    def get_historical_data(self, ticker: str, start_year: int = 2020) -> str:
        """Convenience wrapper for Lakehouse history."""
        # Note: lakehouse.execute returns JSON string
        if not ticker.isalnum():
            raise ValueError("Ticker must be alphanumeric to prevent SQL injection.")
        try:
            start_year = int(start_year)
        except ValueError:
            raise ValueError("start_year must be an integer.")

        query = f"SELECT * FROM financials WHERE ticker = '{ticker}' AND year >= {start_year}"
        return self.lakehouse.execute(query)

    def submit_plan(self, plan_text: str) -> Dict[str, Any]:
        """Parses and logs a natural language plan."""
        return self.planner.parse_natural_language_plan(plan_text)

    def get_tool_definitions(self):
        """Return schema definitions (placeholder for loading tools.json)."""
        # Return schemas for new tools
        return {
            "azure_ai_search": self.ingestor_mcp.get_schema(),
            "microsoft_fabric_run_sql": self.lakehouse.get_schema(),
            "financial_engineering_dcf": {
                "name": "financial_engineering_dcf",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "free_cash_flows": {"type": "array"},
                        "discount_rate": {"type": "number"},
                        "terminal_value": {"type": "number"}
                    }
                }
            }
        }
