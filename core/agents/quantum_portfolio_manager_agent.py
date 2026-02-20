from __future__ import annotations
from typing import Any, Dict, List, Optional, Union
import logging
import asyncio
import numpy as np
import pandas as pd

from core.agents.agent_base import AgentBase
from core.schemas.agent_schema import AgentInput, AgentOutput
from core.data_sources.data_fetcher import DataFetcher
from core.simulations.quantum_monte_carlo import QuantumMonteCarloBridge

class QuantumPortfolioManagerAgent(AgentBase):
    """
    Agent responsible for Quantum-Accelerated Portfolio Optimization.
    It fetches historical data, calculates risk metrics, and uses a quantum bridge (QAOA)
    to determine optimal asset allocation.
    """

    def __init__(self, config: Dict[str, Any], **kwargs):
        """
        Initializes the QuantumPortfolioManagerAgent.
        """
        super().__init__(config, **kwargs)
        self.data_fetcher = DataFetcher()
        self.bridge = QuantumMonteCarloBridge()
        self.lookback_period = config.get("lookback_period", "1y")
        self.interval = config.get("interval", "1d")

    async def execute(self, input_data: Union[str, AgentInput, Dict[str, Any]] = None, **kwargs) -> Union[Dict[str, Any], AgentOutput]:
        """
        Executes the portfolio optimization logic.
        Expects a list of tickers in input_data (or kwargs).
        """
        logging.info("QuantumPortfolioManagerAgent execution started.")

        # 1. Input Normalization
        tickers = []
        is_standard_mode = False
        query = "Portfolio Optimization"

        if input_data is not None:
            if isinstance(input_data, AgentInput):
                # Extract tickers from context or query if possible
                # Simple heuristic: Split query by comma if it looks like a list
                query = input_data.query
                if "," in query:
                    tickers = [t.strip() for t in query.split(",")]
                is_standard_mode = True
            elif isinstance(input_data, list):
                tickers = input_data
            elif isinstance(input_data, dict):
                tickers = input_data.get("tickers", [])
                if "tickers" in input_data:
                    del input_data["tickers"]
                kwargs.update(input_data)

        # Fallback to kwargs
        if not tickers:
            tickers = kwargs.get("tickers", [])

        if not tickers:
            return {
                "error": "No tickers provided for optimization.",
                "status": "failed"
            }

        logging.info(f"Optimizing portfolio for: {tickers}")

        # 2. Fetch Historical Data (Async wrapper)
        loop = asyncio.get_running_loop()

        try:
            # Parallel fetch
            tasks = [
                loop.run_in_executor(None, self.data_fetcher.fetch_historical_data, ticker, self.lookback_period, self.interval)
                for ticker in tickers
            ]
            results = await asyncio.gather(*tasks)

            # 3. Process Data into Returns Matrix
            price_data = {}
            for ticker, history in zip(tickers, results):
                if not history:
                    logging.warning(f"No history for {ticker}, skipping.")
                    continue

                # Convert list of dicts to DataFrame
                df = pd.DataFrame(history)
                # Ensure we use 'close' price
                if 'close' in df.columns:
                    # Set index to date
                    df['date'] = pd.to_datetime(df['date'])
                    df.set_index('date', inplace=True)
                    price_data[ticker] = df['close']

            if len(price_data) < 2:
                return {
                    "error": "Insufficient data for optimization (need at least 2 assets with history).",
                    "status": "failed"
                }

            prices_df = pd.DataFrame(price_data)

            # Forward fill then drop na
            prices_df.ffill(inplace=True)
            prices_df.dropna(inplace=True)

            # Calculate daily returns
            returns_df = prices_df.pct_change().dropna()

            # Expected returns (mean) and Covariance
            mu = returns_df.mean().values
            sigma = returns_df.cov().values
            valid_tickers = returns_df.columns.tolist()

            # 4. Run Quantum Optimization
            # Bridge expects numpy arrays
            optimization_result = self.bridge.optimize_portfolio(valid_tickers, mu, sigma)

            # 5. Format Output
            result = {
                "agent": "QuantumPortfolioManagerAgent",
                "status": "success",
                "optimized_allocation": optimization_result["allocation"],
                "quantum_energy": optimization_result["energy"],
                "method": optimization_result["method"],
                "tickers_processed": valid_tickers
            }

            if is_standard_mode:
                return self._format_output(result, query)

            return result

        except Exception as e:
            logging.error(f"Error in QuantumPortfolioManagerAgent: {e}")
            return {
                "error": str(e),
                "status": "error"
            }

    def _format_output(self, result: Dict[str, Any], query: str) -> AgentOutput:
        allocation = result.get("optimized_allocation", {})
        energy = result.get("quantum_energy")
        method = result.get("method")

        answer = f"Quantum Portfolio Optimization Results ({method}):\n"
        answer += f"Query: {query}\n\n"
        answer += "Optimal Allocation:\n"
        for asset, weight in allocation.items():
            answer += f"- {asset}: {weight*100:.2f}%\n"

        answer += f"\nQuantum State Energy: {energy:.4f} (Lower is more stable)"

        return AgentOutput(
            answer=answer,
            sources=["SimulatedQuantumBridge", "YahooFinance"],
            confidence=0.9,
            metadata=result
        )

    def get_skill_schema(self) -> Dict[str, Any]:
        return {
            "name": "QuantumPortfolioManagerAgent",
            "description": "Optimizes asset allocation using Quantum Approximate Optimization Algorithm (QAOA).",
            "skills": [
                {
                    "name": "optimize_portfolio",
                    "description": "Calculates optimal portfolio weights.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "tickers": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "List of ticker symbols (e.g. ['AAPL', 'MSFT'])"
                            },
                            "lookback_period": {
                                "type": "string",
                                "description": "Historical data period (default '1y')"
                            }
                        },
                        "required": ["tickers"]
                    }
                }
            ]
        }
