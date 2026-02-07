import random
import numpy as np
from collections import deque
from typing import List, Tuple, Dict, Any

class LearningMemory:
    """
    Stores agent experiences as (state, action, reward, next_state, done) tuples.
    Used for experience replay in reinforcement learning.
    """
    def __init__(self, capacity: int = 10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state: Any, action: int, reward: float, next_state: Any, done: bool):
        """Add an experience to memory."""
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> List[Tuple]:
        """Sample a batch of experiences."""
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)


class QLearningAgent:
    """
    Implements Q-Learning algorithm for discrete state and action spaces.
    """
    def __init__(
        self,
        state_size: int,
        action_size: int,
        learning_rate: float = 0.1,
        discount_factor: float = 0.95,
        exploration_rate: float = 1.0,
        exploration_decay: float = 0.995,
        min_exploration_rate: float = 0.01
    ):
        self.state_size = state_size
        self.action_size = action_size
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = exploration_rate
        self.epsilon_decay = exploration_decay
        self.epsilon_min = min_exploration_rate

        # Q-Table: state_index -> [Q(s, a0), Q(s, a1), ...]
        # For simplicity, we assume state is discretizable to an integer index.
        # If state space is large/continuous, we'd need function approximation (e.g., neural net).
        # Here we use a dictionary to sparsely store visited states.
        self.q_table: Dict[int, np.ndarray] = {}

    def get_q_values(self, state: int) -> np.ndarray:
        """Return Q-values for a given state. Initialize if not present."""
        if state not in self.q_table:
            self.q_table[state] = np.zeros(self.action_size)
        return self.q_table[state]

    def act(self, state: int) -> int:
        """Select an action using epsilon-greedy policy."""
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)

        q_values = self.get_q_values(state)
        return np.argmax(q_values)

    def learn(self, state: int, action: int, reward: float, next_state: int, done: bool):
        """Update Q-value based on experience."""
        q_values = self.get_q_values(state)
        next_q_values = self.get_q_values(next_state)

        target = reward
        if not done:
            target += self.gamma * np.max(next_q_values)

        # Q(s,a) = Q(s,a) + lr * (target - Q(s,a))
        q_values[action] += self.lr * (target - q_values[action])

        # Update epsilon
        if done:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save_model(self, filepath: str):
        """Save Q-table to file (placeholder)."""
        # Implementation depends on file format preference (pickle, json, etc.)
        pass

    def load_model(self, filepath: str):
        """Load Q-table from file (placeholder)."""
        pass
