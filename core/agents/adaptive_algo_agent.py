from core.agents.algo_trading_agent import AlgoTradingAgent
from core.learning.adaptive_learning import QLearningAgent, LearningMemory
import numpy as np
import pandas as pd
import random

class AdaptiveAlgoTradingAgent(AlgoTradingAgent):
    """
    An extension of AlgoTradingAgent that uses Reinforcement Learning (Q-Learning)
    to dynamically select the best trading strategy based on market conditions.
    """
    def __init__(self, data: pd.DataFrame, strategies=None, initial_balance=10000,
                 learning_rate=0.1, discount_factor=0.95, epsilon=1.0):
        super().__init__(data, strategies, initial_balance)

        # Define actions: 0=Momentum, 1=Mean Reversion, 2=Arbitrage (simulated)
        self.action_space = len(self.strategies)

        # Define state space size (e.g., 3 regimes: Low/Med/High Volatility)
        # Simplified: We map market volatility to an integer state.
        self.state_space = 3

        self.learner = QLearningAgent(
            state_size=self.state_space,
            action_size=self.action_space,
            learning_rate=learning_rate,
            discount_factor=discount_factor,
            exploration_rate=epsilon
        )
        self.memory = LearningMemory()
        self.episode_rewards = []

    def get_market_state(self, current_idx: int, lookback: int = 20) -> int:
        """
        Discretize market state based on recent volatility.
        Returns:
            0: Low Volatility (Stable)
            1: Medium Volatility (Normal)
            2: High Volatility (Choppy)
        """
        if current_idx < lookback:
            return 1 # Default to Normal

        recent_prices = self.data['Close'][current_idx-lookback:current_idx]
        volatility = recent_prices.std()

        # Simple thresholds (these should be tuned or dynamic in a real system)
        if volatility < 2.0:
            return 0
        elif volatility < 5.0:
            return 1
        else:
            return 2

    def run_adaptive_simulation(self, episode_length: int = 20, episodes: int = 50):
        """
        Train the agent over multiple episodes.
        Each step is a period (e.g., 20 days) where one strategy is applied.
        """
        # We simulate training by sliding a window over the data
        # If data is short, we loop or use random starting points.

        total_steps = len(self.data) - episode_length
        if total_steps <= 0:
            print("Not enough data for simulation.")
            return

        current_idx = 0
        balance = self.initial_balance

        for episode in range(episodes):
            # Reset environment (balance) occasionally or just continue?
            # Let's continue to simulate long-term adaptation.
            if current_idx + episode_length >= len(self.data):
                current_idx = 0 # Loop data
                balance = self.initial_balance # Reset balance on loop

            state = self.get_market_state(current_idx)
            action = self.learner.act(state)

            # Map action index to strategy name
            strategy_name = self.strategies[action]

            # Execute strategy for the episode duration
            # Note: The base class methods (momentum_trading etc) run on FULL data.
            # We need a way to run them on a SUBSET or modify them.
            # For this additive implementation, we will simulate the return based on
            # calling the full strategy method but extracting only the relevant period's performance.

            # This is inefficient but reuses existing logic without modification (Additive).
            # A better approach would refactor the base class to accept start/end indices.

            full_results = self.run_simulation(strategy_name)
            # Calculate return for this specific window approximately
            # In a real scenario, we'd pass start_idx/end_idx to run_simulation.
            # Here we mock the step reward based on the strategy's average performance
            # combined with current market volatility (simulated environment response).

            # Mock reward function for demonstration:
            # If High Volatility (State 2) and Mean Reversion (Action 1), high reward.
            # If Low Volatility (State 0) and Momentum (Action 0), high reward.
            # Otherwise random/low.

            reward = 0
            if state == 2 and strategy_name == 'mean_reversion':
                reward = 100 + random.uniform(-10, 20)
            elif state == 0 and strategy_name == 'momentum':
                reward = 100 + random.uniform(-10, 20)
            elif state == 1:
                reward = 50 + random.uniform(-20, 20) # Random walk
            else:
                reward = -50 + random.uniform(-10, 10) # Punishment for wrong strategy

            next_idx = current_idx + episode_length
            next_state = self.get_market_state(next_idx)

            # Store experience
            self.memory.push(state, action, reward, next_state, False)

            # Learn
            self.learner.learn(state, action, reward, next_state, False)

            self.episode_rewards.append(reward)
            current_idx = next_idx

        return self.episode_rewards

    def get_optimal_strategy(self, current_volatility_state: int) -> str:
        """Returns the learned best strategy for a given volatility state."""
        action = np.argmax(self.learner.get_q_values(current_volatility_state))
        return self.strategies[action]
