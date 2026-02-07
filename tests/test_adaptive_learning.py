import unittest
import numpy as np
import pandas as pd
from core.learning.adaptive_learning import LearningMemory, QLearningAgent
from core.agents.adaptive_algo_agent import AdaptiveAlgoTradingAgent

class TestAdaptiveLearning(unittest.TestCase):

    def test_memory(self):
        memory = LearningMemory(capacity=10)
        memory.push(0, 1, 10.0, 1, False)
        self.assertEqual(len(memory), 1)

        batch = memory.sample(1)
        self.assertEqual(len(batch), 1)
        state, action, reward, next_state, done = batch[0]
        self.assertEqual(state, 0)
        self.assertEqual(reward, 10.0)

    def test_q_learning_agent(self):
        agent = QLearningAgent(state_size=3, action_size=2, learning_rate=0.5, discount_factor=0.9)

        # Test initial Q-values
        q_vals = agent.get_q_values(0)
        np.testing.assert_array_equal(q_vals, np.zeros(2))

        # Test learning update
        # State 0, Action 0, Reward 10, Next State 1 (max Q=0)
        # Q(0,0) = 0 + 0.5 * (10 + 0.9*0 - 0) = 5.0
        agent.learn(state=0, action=0, reward=10.0, next_state=1, done=False)
        self.assertEqual(agent.get_q_values(0)[0], 5.0)

        # Test action selection (greedy)
        agent.epsilon = 0.0 # Force exploit
        action = agent.act(0)
        self.assertEqual(action, 0) # Should pick index 0 as it has value 5.0 vs 0.0

    def test_adaptive_algo_agent(self):
        # Create dummy market data
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        # Create a trend followed by high volatility
        prices = np.linspace(100, 150, 50).tolist() + \
                 (np.linspace(150, 100, 25) + np.random.normal(0, 5, 25)).tolist() + \
                 (np.linspace(100, 120, 25)).tolist()

        data = pd.DataFrame({'Date': dates, 'Close': prices})

        agent = AdaptiveAlgoTradingAgent(data, strategies=['momentum', 'mean_reversion'])

        # Run simulation/training
        # Use short episodes to ensure multiple updates
        rewards = agent.run_adaptive_simulation(episode_length=10, episodes=5)

        # Check if agent learned something (Q-table not empty)
        self.assertTrue(len(agent.learner.q_table) > 0)

        # Check if rewards were recorded
        self.assertGreater(len(agent.episode_rewards), 0)

if __name__ == '__main__':
    unittest.main()
