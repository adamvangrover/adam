from __future__ import annotations
import sqlite3
import pandas as pd
import sys
import os
from typing import List, Optional, Dict, Any, Union
from pydantic import BaseModel, Field

# Ensure core is importable
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../..")))

try:
    from core.vertical_risk_agent.generative_risk import GenerativeRiskEngine
    from core.v22_quantum_pipeline.qmc_engine import QuantumMonteCarloEngine
except ImportError:
    GenerativeRiskEngine = None
    QuantumMonteCarloEngine = None

try:
    from core.engine.meta_orchestrator import MetaOrchestrator
except ImportError:
    MetaOrchestrator = None


class FinancialRatio(BaseModel):
    name: str
    value: str


class SimulationResult(BaseModel):
    pd: float = Field(..., description="Probability of Default")
    asset_value_stats: Dict[str, float] = Field(..., description="Statistics of asset value distribution")


class AgentTools:
    """
    Centralized tool definitions for the Vertical Risk Agent.
    Implements strict typing and error handling.
    """

    def __init__(self, db_path: str = "finance_data.db"):
        self.db_path = db_path
        self._orchestrator = None

    def _get_orchestrator(self):
        if self._orchestrator is None and MetaOrchestrator:
            self._orchestrator = MetaOrchestrator()
        return self._orchestrator

    def get_10k_filing(self, ticker: str, year: str) -> str:
        """
        Returns the full text of the 10-K filing for the given ticker and year.
        """
        # Placeholder for actual implementation
        return f"Full 10-K text content for {ticker} in {year}..."

    def get_financial_ratios(self, ticker: str) -> List[FinancialRatio]:
        """
        Returns key financial ratios for a given ticker.
        """
        return [
            FinancialRatio(name="Debt/EBITDA", value="2.5x"),
            FinancialRatio(name="Interest Coverage", value="5.0x")
        ]

    def query_sql(self, query: str) -> str:
        """
        Executes a read-only SQL query against the local financial database.
        Restricted to specific tables for security.
        """
        if not query.strip().upper().startswith("SELECT"):
            raise ValueError("Only SELECT queries are allowed.")

        # Whitelist of allowed tables to prevent accessing secrets or system tables
        ALLOWED_TABLES = {'financials', 'sqlite_sequence'}

        def authorizer(action, arg1, arg2, dbname, source):
            # Allow SELECT statements
            if action == sqlite3.SQLITE_SELECT:
                return sqlite3.SQLITE_OK
            # Allow READ on whitelisted tables
            if action == sqlite3.SQLITE_READ:
                if arg1 in ALLOWED_TABLES:
                    return sqlite3.SQLITE_OK
                return sqlite3.SQLITE_DENY
            # Allow standard SQL functions
            if action == sqlite3.SQLITE_FUNCTION:
                return sqlite3.SQLITE_OK
            # Deny everything else (INSERT, UPDATE, DELETE, PRAGMA, etc.)
            return sqlite3.SQLITE_DENY

        try:
            # Use read-only URI mode as an extra layer of defense
            uri_path = f"file:{os.path.abspath(self.db_path)}?mode=ro"
            conn = sqlite3.connect(uri_path, uri=True)
            conn.set_authorizer(authorizer)

            # Use chunks or limit to prevent OOM on large datasets in a real scenario
            df = pd.read_sql_query(query, conn)
            conn.close()
            return df.to_markdown()
        except Exception as e:
            return f"SQL Error: {str(e)}"

    def get_covenant_definitions(self, doc_id: str) -> str:
        """
        Retrieves the legal definitions of financial covenants from a specific credit agreement.
        """
        return """
        Section 7.1. Financial Covenants.
        (a) Consolidated Leverage Ratio. The Borrower shall not permit the Consolidated Leverage Ratio
            as of the end of any Fiscal Quarter to be greater than 4.50 to 1.00.
        """

    def simulate_quantum_merton_model(self, asset_value: float, debt: float, volatility: float, horizon: float) -> Union[str, Dict[str, Any]]:
        """
        Runs an End-to-End Quantum Monte Carlo simulation for credit risk (Merton Model).
        """
        if not QuantumMonteCarloEngine:
            return "Error: QMC Engine not available."

        try:
            qmc = QuantumMonteCarloEngine()
            # Assuming simulate_merton_model returns a dict or object convertible to dict
            result = qmc.simulate_merton_model(asset_value, debt, volatility, 0.05, horizon)
            return result
        except Exception as e:
            return f"Simulation Error: {str(e)}"

    def generate_stress_scenarios(self, regime: str = "stress", n_samples: int = 5) -> Union[str, List[Any]]:
        """
        Generates synthetic market scenarios using a Generative Risk Engine.
        """
        if not GenerativeRiskEngine:
            return "Error: Generative Risk Engine not available."

        try:
            engine = GenerativeRiskEngine()
            scenarios = engine.generate_scenarios(n_samples=n_samples, regime=regime)
            return scenarios
        except Exception as e:
            return f"Generation Error: {str(e)}"

    async def run_deep_dive_analysis(self, query: str) -> str:
        """
        Triggers a full v23.5 Deep Dive analysis using the MetaOrchestrator.
        """
        orchestrator = self._get_orchestrator()
        if not orchestrator:
            return "Error: MetaOrchestrator not available."

        try:
            # Assuming route_request is an async method
            result = await orchestrator.route_request(query)
            return str(result)
        except Exception as e:
            return f"Error running Deep Dive: {str(e)}"
