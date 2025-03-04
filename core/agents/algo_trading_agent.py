#core/agents/algo_trading_agent.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random

class AlgoTradingAgent:
    def __init__(self, data, strategies=None, initial_balance=10000):
        """
        Initializes the trading agent.

        Args:
            data (pd.DataFrame): Historical market data (e.g., price history).
            strategies (list, optional): List of strategies to be simulated. Defaults to None.
            initial_balance (float, optional): The starting balance for the simulation. Defaults to 10000.
        """
        self.data = data
        self.strategies = strategies or ['momentum', 'mean_reversion', 'arbitrage']
        self.initial_balance = initial_balance
        self.results = {}

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
            print("Unknown strategy.")
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
        trade_df['Cumulative Balance'] = trade_df['Balance'].cumsum() + trade_df['Position'] * self.data['Close']

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
            self.results[strategy] = self.run_simulation(strategy)

        return self.results

    def plot_performance(self):
        """
        Plots the performance of all strategies.
        """
        plt.figure(figsize=(10, 6))

        for strategy, result in self.results.items():
            plt.plot(result['Final Balance'], label=f'{strategy} (Total Return: {result["Total Return"]*100:.2f}%)')

        plt.title('Strategy Performance Comparison')
        plt.xlabel('Time')
        plt.ylabel('Portfolio Value')
        plt.legend(loc='upper left')
        plt.show()


# Example usage
if __name__ == "__main__":
    # Simulate market data (replace with actual market data)
    dates = pd.date_range('2023-01-01', '2023-12-31', freq='D')
    prices = np.random.normal(100, 5, len(dates))  # Simulating daily closing prices
    market_data = pd.DataFrame({'Date': dates, 'Close': prices})

    # Initialize the agent
    agent = AlgoTradingAgent(data=market_data)

    # Run and evaluate the strategies
    agent.evaluate_strategies()

    # Plot performance comparison
    agent.plot_performance()
