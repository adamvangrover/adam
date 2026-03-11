# core/agents/algo_trading_agent.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import logging
from typing import Dict, Any, Optional, List
from core.agents.agent_base import AgentBase

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class AlgoTradingAgent(AgentBase):
    """
    Agent responsible for executing algorithmic trading strategies.
    """

    def __init__(self, config: Dict[str, Any], kernel: Optional[Any] = None):
        """
        Initializes the trading agent.

        Args:
            config (dict): Configuration dictionary.
            kernel (Optional[Any]): Semantic Kernel instance.
        """
        super().__init__(config, kernel=kernel)
        self.strategies = self.config.get('strategies', ['momentum', 'mean_reversion', 'arbitrage'])
        self.initial_balance = self.config.get('initial_balance', 10000)
        self.results = {}
        # Data is not stored in __init__ anymore, but passed during execution

    async def execute(self, *args, **kwargs):
        """
        Evaluates strategies by running simulations.

        Args:
            data (pd.DataFrame): Historical market data (e.g., price history). Passed via kwargs.

        Returns:
            dict: A dictionary of performance metrics for each strategy.
        """
        data = kwargs.get('data')

        if data is None:
            logging.error("No data provided for execution.")
            return {"error": "No data provided."}

        if not isinstance(data, pd.DataFrame):
             logging.error("Data must be a pandas DataFrame.")
             return {"error": "Data must be a pandas DataFrame."}

        self.data = data
        logging.info("Starting evaluation of trading strategies...")

        results = self.evaluate_strategies()
        return results

    def run_simulation(self, strategy):
        """
        Runs the simulation for a given strategy.

        Args:
            strategy (str): The name of the strategy to simulate.

        Returns:
            dict: The performance metrics for the strategy.
        """
        if strategy == 'momentum':
            return self.momentum_trading()
        elif strategy == 'mean_reversion':
            return self.mean_reversion_trading()
        elif strategy == 'arbitrage':
            return self.arbitrage_trading()
        else:
            logging.warning(f"Unknown strategy: {strategy}")
            return {}

    def momentum_trading(self):
        """
        Simulates momentum trading strategy.

        Buy when the price is above a short-term moving average, sell when below.

        Returns:
            dict: Performance metrics for momentum strategy.
        """
        balance = self.initial_balance
        position = 0  # No position initially (cash)
        trade_log = []

        short_window = 20  # short-term moving average
        long_window = 50  # long-term moving average

        if len(self.data) < long_window:
             logging.warning("Not enough data for momentum strategy.")
             return {}

        for i in range(long_window, len(self.data)):
            short_ma = self.data['Close'][i - short_window:i].mean()
            long_ma = self.data['Close'][i - long_window:i].mean()

            if short_ma > long_ma and position == 0:  # Buy signal
                position = balance / self.data['Close'][i]
                balance = 0  # Spend all balance

            elif short_ma < long_ma and position > 0:  # Sell signal
                balance = position * self.data['Close'][i]
                position = 0  # Exit position

            trade_log.append({'Balance': balance, 'Position': position})

        final_balance = balance if position == 0 else position * self.data['Close'].iloc[-1]
        return self.calculate_performance_metrics(trade_log, final_balance)

    def mean_reversion_trading(self):
        """
        Simulates mean reversion trading strategy.

        Buy when the price is below the mean (moving average), sell when above.

        Returns:
            dict: Performance metrics for mean reversion strategy.
        """
        balance = self.initial_balance
        position = 0
        trade_log = []

        window_size = 20  # Lookback period for moving average

        if len(self.data) < window_size:
             logging.warning("Not enough data for mean reversion strategy.")
             return {}

        for i in range(window_size, len(self.data)):
            mean = self.data['Close'][i - window_size:i].mean()
            price = self.data['Close'][i]

            if price < mean and position == 0:  # Buy signal
                position = balance / price
                balance = 0

            elif price > mean and position > 0:  # Sell signal
                balance = position * price
                position = 0

            trade_log.append({'Balance': balance, 'Position': position})

        final_balance = balance if position == 0 else position * self.data['Close'].iloc[-1]
        return self.calculate_performance_metrics(trade_log, final_balance)

    def arbitrage_trading(self):
        """
        Simulates an arbitrage strategy (simplified version).

        Exploit price differences between two assets or exchanges.

        Returns:
            dict: Performance metrics for arbitrage strategy.
        """
        balance = self.initial_balance
        position = 0
        trade_log = []

        for i in range(1, len(self.data)):
            # Simplified arbitrage logic: If price difference between two exchanges
            # exceeds a certain threshold, take advantage of it.
            price_1 = self.data['Close'][i]
            price_2 = self.data['Close'][i] * (1 + random.uniform(-0.01, 0.01))  # Simulate price difference

            if price_1 < price_2 and position == 0:  # Arbitrage opportunity
                position = balance / price_1
                balance = 0

            elif price_1 > price_2 and position > 0:  # Exit arbitrage opportunity
                balance = position * price_1
                position = 0

            trade_log.append({'Balance': balance, 'Position': position})

        final_balance = balance if position == 0 else position * self.data['Close'].iloc[-1]
        return self.calculate_performance_metrics(trade_log, final_balance)

    def calculate_performance_metrics(self, trade_log, final_balance):
        """
        Calculates performance metrics such as cumulative returns and drawdown.

        Args:
            trade_log (list): List of trade actions.
            final_balance (float): The final balance after the simulation.

        Returns:
            dict: The performance metrics.
        """
        trade_df = pd.DataFrame(trade_log)
        if trade_df.empty:
             return {
                'Final Balance': self.initial_balance,
                'Total Return': 0.0,
                'Win Rate': 0.0,
                'Max Drawdown': 0.0,
                'Sharpe Ratio': 0.0
            }

        trade_df['Cumulative Balance'] = trade_df['Balance'] + trade_df['Position'] * self.data['Close'].reset_index(drop=True).iloc[len(self.data)-len(trade_df):].values

        total_return = (trade_df['Cumulative Balance'].iloc[-1] - self.initial_balance) / self.initial_balance
        win_rate = np.mean(trade_df['Cumulative Balance'] > 0) * 100
        max_drawdown = self.calculate_max_drawdown(trade_df['Cumulative Balance'])

        # Sharpe Ratio
        daily_returns = trade_df['Cumulative Balance'].pct_change().dropna()
        sharpe_ratio = daily_returns.mean() / daily_returns.std() if daily_returns.std() != 0 else 0

        return {
            'Final Balance': final_balance,
            'Total Return': total_return,
            'Win Rate': win_rate,
            'Max Drawdown': max_drawdown,
            'Sharpe Ratio': sharpe_ratio
        }

    def calculate_max_drawdown(self, cumulative_balance):
        """
        Calculates the maximum drawdown of the portfolio.

        Args:
            cumulative_balance (pd.Series): Series of cumulative balance values.

        Returns:
            float: The maximum drawdown percentage.
        """
        rolling_max = cumulative_balance.cummax()
        drawdown = (cumulative_balance - rolling_max) / rolling_max
        return drawdown.min()

    def evaluate_strategies(self):
        """
        Evaluates all strategies by running simulations and comparing their performance.

        Returns:
            dict: A dictionary of performance metrics for each strategy.
        """
        for strategy in self.strategies:
            logging.info(f"Running simulation for strategy: {strategy}")
            self.results[strategy] = self.run_simulation(strategy)

        return self.results

    def plot_performance(self):
        """
        Plots the performance of all strategies.
        """
        plt.figure(figsize=(10, 6))

        for strategy, result in self.results.items():
            if 'Final Balance' in result:
                 # Note: Ideally we would plot the equity curve, but result currently only has summary metrics
                 # To plot equity curve, run_simulation needs to return the time series.
                 # For now, we keep this method but it might not plot a line graph as intended in original code
                 # unless we change run_simulation return type.
                 # Given refactor scope, let's just log that plotting needs time series data.
                 logging.info(f"Strategy {strategy}: Final Balance {result['Final Balance']}")
            else:
                 logging.warning(f"No results for {strategy}")

        # Original code plotted result['Final Balance'] which suggests result was a time series?
        # Looking at original code:
        # result = self.run_simulation(strategy) -> returns dict with 'Final Balance' (scalar).
        # plt.plot(result['Final Balance']) -> plotting a scalar? That doesn't make sense for a line plot.
        # It seems the original code might have been pseudocode or broken regarding plotting.
        # I will leave this method as is but comment out the plotting to avoid errors.
        logging.info("Plotting not implemented in this version (requires equity curve data).")
        # plt.show()


# Example usage
if __name__ == "__main__":
    import asyncio
    # Simulate market data (replace with actual market data)
    dates = pd.date_range('2023-01-01', '2023-12-31', freq='D')
    prices = np.random.normal(100, 5, len(dates))  # Simulating daily closing prices
    market_data = pd.DataFrame({'Date': dates, 'Close': prices})

    # Initialize the agent
    config = {'strategies': ['momentum', 'mean_reversion'], 'initial_balance': 10000}
    agent = AlgoTradingAgent(config=config)

    # Run and evaluate the strategies
    async def main():
        results = await agent.execute(data=market_data)
        print("Results:", results)

    asyncio.run(main())
