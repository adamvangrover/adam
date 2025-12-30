# AlphaFinance Simulation Environment

## Overview
`AlphaFinance` is a Reinforcement Learning (RL) environment designed to train autonomous agents in portfolio management and trading execution. It draws inspiration from DeepMind's **AlphaZero** and **MuZero** architectures, framing financial decision-making as a sequential game.

## Architecture

### Environment (`AlphaFinanceEnv`)
*   **State Space:** A window of market data (Price, Volume, Technical Indicators, Macro Signals).
*   **Action Space:** Continuous action space representing the target portfolio weight allocation (e.g., 0.0 to 1.0 for a single asset, or a vector for multi-asset).
*   **Reward Function:** Log Returns or Sharpe Ratio (risk-adjusted return).

### Agent (`AlphaAgent`)
*   **Actor-Critic:** The agent is designed to use an Actor-Critic architecture (or MCTS in future versions).
*   **Policy Network:** Predicts the optimal action (weights).
*   **Value Network:** Predicts the expected future reward (Value at Risk / Expected Return).

## Usage

```python
from core.simulations.alpha_finance import AlphaFinanceEnv, AlphaAgent

# Initialize Env
data = load_market_data("AAPL")
env = AlphaFinanceEnv(data_feed=data)

# Run Loop
state = env.reset()
done = False
while not done:
    action = agent.act(state)
    next_state, reward, done, info = env.step(action)
    agent.learn(state, action, reward, next_state)
```

## Future Work
*   **Multi-Agent:** Simulate market dynamics with multiple competing AlphaAgents.
*   **MuZero Integration:** Implement the model-based planning of MuZero to "imagine" future market scenarios.
*   **Real Data:** Connect to `BigQueryConnector` to train on massive historical datasets.
