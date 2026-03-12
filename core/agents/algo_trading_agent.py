from __future__ import annotations
from typing import Any, Dict, List, Optional, Union
import logging
import numpy as np
import pandas as pd
import random
import asyncio

from core.agents.agent_base import AgentBase
from core.schemas.agent_schema import AgentInput, AgentOutput

# Configure logging
logger = logging.getLogger(__name__)

class AlgoTradingAgent(AgentBase):
    """
    Agent responsible for executing algorithmic trading strategies.
    Simulates trading strategies based on historical market data.
    """

    def __init__(self, config: Dict[str, Any], **kwargs):
        """
        Initializes the trading agent.

        Args:
            config: A dictionary containing configuration parameters.
                    Expected keys: 'strategies', 'initial_balance', 'data' (optional dataframe)
        """
        super().__init__(config, **kwargs)
        self.strategies = self.config.get('strategies', ['momentum', 'mean_reversion', 'arbitrage'])
        self.initial_balance = self.config.get('initial_balance', 10000)

        # Data can be passed in config or injected later
        self.data = self.config.get('data')
        self.results = {}

    async def execute(self, input_data: Union[str, AgentInput, Dict[str, Any]] = None, **kwargs) -> Union[Dict[str, Any], AgentOutput]:
        """
        Executes trading simulations or strategy evaluations.
        Supports both legacy string/dict input and new AgentInput schema.

        Returns:
            Dict containing the performance metrics, or AgentOutput.
        """
        # 1. Input Normalization
        query = ""
        is_standard_mode = False
        target_strategy = None

        # Handle input
        if input_data is not None:
            if isinstance(input_data, AgentInput):
                query = input_data.query
                is_standard_mode = True
                if input_data.context:
                    # Allow passing data via context
                    if 'data' in input_data.context:
                        self.data = pd.DataFrame(input_data.context['data'])
            elif isinstance(input_data, str):
                query = input_data
            elif isinstance(input_data, dict):
                # Legacy support
                query = input_data.get("query", "")
                target_strategy = input_data.get("strategy")
                if "data" in input_data:
                    self.data = pd.DataFrame(input_data["data"])
                kwargs.update(input_data)

        if not isinstance(self.data, pd.DataFrame) or self.data.empty:
            # Fallback for testing/demo if no data provided
            logger.warning("No market data provided. Generating synthetic data.")
            self._generate_synthetic_data()

        logger.info(f"AlgoTradingAgent execution started. Strategy: {target_strategy or 'All'}")

        # Execute
        if target_strategy:
            results = self.run_simulation(target_strategy)
            combined_results = {target_strategy: results}
        else:
            combined_results = self.evaluate_strategies()

        # Format Result
        result_package = {
            "strategies_evaluated": list(combined_results.keys()),
            "results": combined_results,
            "best_strategy": self._find_best_strategy(combined_results)
        }

        if is_standard_mode:
            return self._format_output(result_package, query)

        return result_package

    def _generate_synthetic_data(self):
        """Generates synthetic market data for demonstration."""
        dates = pd.date_range('2023-01-01', '2023-12-31', freq='D')
        prices = np.random.normal(100, 5, len(dates))  # Simulating daily closing prices
        self.data = pd.DataFrame({'Date': dates, 'Close': prices})

    def run_simulation(self, strategy: str) -> Dict[str, Any]:
        """
        Runs the simulation for a given strategy.
        """
        if strategy == 'momentum':
            return self.momentum_trading()
        elif strategy == 'mean_reversion':
            return self.mean_reversion_trading()
        elif strategy == 'arbitrage':
            return self.arbitrage_trading()
        else:
            logger.warning(f"Unknown strategy: {strategy}")
            return {"error": "Unknown strategy"}

    def momentum_trading(self) -> Dict[str, Any]:
        """
        Simulates momentum trading strategy.
        Buy when the price is above a short-term moving average, sell when below.
        """
        balance = self.initial_balance
        position = 0  # No position initially (cash)
        trade_log = []

        short_window = 20  # short-term moving average
        long_window = 50  # long-term moving average

        if len(self.data) < long_window:
             return {"error": "Not enough data for momentum strategy"}

        for i in range(long_window, len(self.data)):
            short_ma = self.data['Close'][i - short_window:i].mean()
            long_ma = self.data['Close'][i - long_window:i].mean()
            price = self.data['Close'].iloc[i]

            if short_ma > long_ma and position == 0:  # Buy signal
                position = balance / price
                balance = 0  # Spend all balance

            elif short_ma < long_ma and position > 0:  # Sell signal
                balance = position * price
                position = 0  # Exit position

            trade_log.append({'Balance': balance, 'Position': position})

        final_balance = balance if position == 0 else position * self.data['Close'].iloc[-1]
        return self.calculate_performance_metrics(trade_log, final_balance)

    def mean_reversion_trading(self) -> Dict[str, Any]:
        """
        Simulates mean reversion trading strategy.
        Buy when the price is below the mean (moving average), sell when above.
        """
        balance = self.initial_balance
        position = 0
        trade_log = []

        window_size = 20  # Lookback period for moving average

        if len(self.data) < window_size:
             return {"error": "Not enough data for mean reversion strategy"}

        for i in range(window_size, len(self.data)):
            mean = self.data['Close'][i - window_size:i].mean()
            price = self.data['Close'].iloc[i]

            if price < mean and position == 0:  # Buy signal
                position = balance / price
                balance = 0

            elif price > mean and position > 0:  # Sell signal
                balance = position * price
                position = 0

            trade_log.append({'Balance': balance, 'Position': position})

        final_balance = balance if position == 0 else position * self.data['Close'].iloc[-1]
        return self.calculate_performance_metrics(trade_log, final_balance)

    def arbitrage_trading(self) -> Dict[str, Any]:
        """
        Simulates an arbitrage strategy (simplified version).
        Exploit price differences between two assets or exchanges.
        """
        balance = self.initial_balance
        position = 0
        trade_log = []

        for i in range(1, len(self.data)):
            price_1 = self.data['Close'].iloc[i]
            # Simulate price difference
            price_2 = price_1 * (1 + random.uniform(-0.01, 0.01))

            if price_1 < price_2 and position == 0:  # Arbitrage opportunity
                position = balance / price_1
                balance = 0

            elif price_1 > price_2 and position > 0:  # Exit arbitrage opportunity
                balance = position * price_1
                position = 0

            trade_log.append({'Balance': balance, 'Position': position})

        final_balance = balance if position == 0 else position * self.data['Close'].iloc[-1]
        return self.calculate_performance_metrics(trade_log, final_balance)

    def calculate_performance_metrics(self, trade_log: List[Dict], final_balance: float) -> Dict[str, float]:
        """
        Calculates performance metrics such as cumulative returns and drawdown.
        """
        if not trade_log:
            return {
                'Final Balance': self.initial_balance,
                'Total Return': 0.0,
                'Win Rate': 0.0,
                'Max Drawdown': 0.0,
                'Sharpe Ratio': 0.0
            }

        trade_df = pd.DataFrame(trade_log)
        # Using .values for broadcasting safety if indices mismatch, though list construction prevents it mostly
        trade_df['Cumulative Balance'] = trade_df['Balance'] + trade_df['Position'] * self.data['Close'].iloc[-len(trade_df):].values

        total_return = (trade_df['Cumulative Balance'].iloc[-1] - self.initial_balance) / self.initial_balance

        # Win rate approximation based on daily balance increases?
        # Or simpler: just % of days positive?
        # Original logic: np.mean(trade_df['Cumulative Balance'] > 0) checks if balance is positive, which is always true.
        # Let's check daily returns > 0
        daily_returns = trade_df['Cumulative Balance'].pct_change().dropna()
        win_rate = np.mean(daily_returns > 0) * 100 if not daily_returns.empty else 0.0

        max_drawdown = self.calculate_max_drawdown(trade_df['Cumulative Balance'])

        sharpe_ratio = 0.0
        if not daily_returns.empty and daily_returns.std() != 0:
            sharpe_ratio = daily_returns.mean() / daily_returns.std()

        return {
            'Final Balance': final_balance,
            'Total Return': total_return,
            'Win Rate': win_rate,
            'Max Drawdown': max_drawdown,
            'Sharpe Ratio': sharpe_ratio
        }

    def calculate_max_drawdown(self, cumulative_balance: pd.Series) -> float:
        """
        Calculates the maximum drawdown of the portfolio.
        """
        rolling_max = cumulative_balance.cummax()
        drawdown = (cumulative_balance - rolling_max) / rolling_max
        return drawdown.min()

    def evaluate_strategies(self) -> Dict[str, Dict[str, Any]]:
        """
        Evaluates all strategies.
        """
        for strategy in self.strategies:
            self.results[strategy] = self.run_simulation(strategy)
        return self.results

    def _find_best_strategy(self, results: Dict[str, Dict[str, Any]]) -> str:
        """Identifies the strategy with highest total return."""
        best_strat = "None"
        best_return = -float('inf')

        for strat, metrics in results.items():
            if "error" in metrics:
                continue
            ret = metrics.get('Total Return', 0)
            if ret > best_return:
                best_return = ret
                best_strat = strat
        return best_strat

    def _format_output(self, result: Dict[str, Any], query: str) -> AgentOutput:
        """Helper to format output to AgentOutput."""
        best_strat = result.get("best_strategy", "None")
        results = result.get("results", {})

        answer = f"Algorithmic Trading Simulation Results for '{query or 'General'}':\n"
        answer += f"Best Performing Strategy: {best_strat}\n\n"

        for strat, metrics in results.items():
            if "error" in metrics:
                answer += f"- {strat}: Failed ({metrics['error']})\n"
            else:
                answer += f"- {strat}:\n"
                answer += f"  Return: {metrics.get('Total Return', 0)*100:.2f}%\n"
                answer += f"  Sharpe: {metrics.get('Sharpe Ratio', 0):.2f}\n"
                answer += f"  Max DD: {metrics.get('Max Drawdown', 0)*100:.2f}%\n"

        return AgentOutput(
            answer=answer,
            sources=["Historical Market Data Simulation"],
            confidence=0.9,
            metadata=result
        )

# Example usage
if __name__ == "__main__":
    # Simulate market data
    logging.basicConfig(level=logging.INFO)

    async def main():
        config = {'strategies': ['momentum', 'mean_reversion', 'arbitrage']}
        agent = AlgoTradingAgent(config)

        # Test Standard Mode
        print("\n--- Standard Mode ---")
        input_obj = AgentInput(query="Evaluate All Strategies")
        result = await agent.execute(input_obj)
        print(result.answer)

    asyncio.run(main())
