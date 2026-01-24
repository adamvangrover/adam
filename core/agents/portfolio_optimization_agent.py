# core/agents/portfolio_optimization_agent.py

import numpy as np
import pandas as pd
import logging
from typing import Dict, Any, List, Optional, Tuple
from core.agents.agent_base import AgentBase

# Optional imports for AI/Math
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    from scipy.optimize import minimize
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

if HAS_TORCH:
    class LSTMModel(nn.Module):
        def __init__(self, input_size=1, hidden_size=50):
            super(LSTMModel, self).__init__()
            self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=True)
            self.fc = nn.Linear(hidden_size, input_size)

        def forward(self, x):
            # x shape: (batch, seq, feature)
            out, _ = self.lstm(x)
            # Take last time step
            out = self.fc(out[:, -1, :])
            return out

class PortfolioOptimizationAgent(AgentBase):
    """
    Agent for portfolio optimization using both Classical (Mean-Variance) and AI (LSTM) approaches.
    """

    def __init__(self, config: Dict[str, Any] = None, constitution: Dict[str, Any] = None, kernel: Any = None):
        super().__init__(config, constitution, kernel)
        self.risk_free_rate = self.config.get('risk_free_rate', 0.02)

        if HAS_TORCH:
            self.model = LSTMModel()
            self.criterion = nn.MSELoss()
            self.optimizer = optim.Adam(self.model.parameters(), lr=0.01)
        else:
            logging.warning("PyTorch not found. AI optimization disabled.")

    async def execute(self, *args, **kwargs):
        """
        Main execution entry point.
        Expected kwargs:
            - historical_prices: pd.DataFrame or dict (assets as columns, date index)
            - method: 'mean_variance' (default) or 'ai_forecast'
        """
        data = kwargs.get('historical_prices')
        method = kwargs.get('method', 'mean_variance')

        if data is None:
            return {"status": "error", "message": "No historical_prices provided"}

        # Convert to DataFrame if dict
        if isinstance(data, dict):
            try:
                data = pd.DataFrame(data)
            except Exception as e:
                return {"status": "error", "message": f"Failed to convert data to DataFrame: {e}"}

        if method == 'mean_variance':
            return self.optimize_mean_variance(data)
        elif method == 'ai_forecast':
            return self.optimize_ai(data)
        else:
            return {"status": "error", "message": f"Unknown method: {method}"}

    def optimize_mean_variance(self, prices: pd.DataFrame) -> Dict[str, Any]:
        """
        Performs Classical Mean-Variance Optimization (Markowitz).
        Minimizes volatility for a given target return (or maximizes Sharpe).
        Here we Maximize Sharpe Ratio.
        """
        if not HAS_SCIPY:
            return {"status": "error", "message": "Scipy not installed."}

        # Calculate returns
        returns = prices.pct_change().dropna()
        mean_returns = returns.mean() * 252 # Annualized
        cov_matrix = returns.cov() * 252 # Annualized

        num_assets = len(prices.columns)
        assets = prices.columns.tolist()

        # Define functions for optimization
        def portfolio_performance(weights):
            weights = np.array(weights)
            returns = np.sum(mean_returns * weights)
            std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            return returns, std

        def negative_sharpe(weights):
            p_ret, p_std = portfolio_performance(weights)
            return -(p_ret - self.risk_free_rate) / p_std

        # Constraints: sum(weights) = 1
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        # Bounds: 0 <= weight <= 1 (no shorting)
        bounds = tuple((0, 1) for _ in range(num_assets))
        # Initial guess: equal weights
        init_guess = num_assets * [1. / num_assets,]

        result = minimize(negative_sharpe, init_guess, method='SLSQP', bounds=bounds, constraints=constraints)

        optimized_weights = result.x
        opt_ret, opt_std = portfolio_performance(optimized_weights)

        # Format results
        allocation = {assets[i]: round(optimized_weights[i], 4) for i in range(num_assets)}

        return {
            "status": "success",
            "method": "Mean-Variance (Max Sharpe)",
            "allocation": allocation,
            "metrics": {
                "expected_return": round(opt_ret, 4),
                "volatility": round(opt_std, 4),
                "sharpe_ratio": round((opt_ret - self.risk_free_rate) / opt_std, 4)
            }
        }

    def optimize_ai(self, prices: pd.DataFrame) -> Dict[str, Any]:
        """
        Uses LSTM to forecast next day returns and allocates based on positive predictions.
        (Simplified strategy: Proportional to forecasted positive returns).
        """
        if not HAS_TORCH:
            return {"status": "error", "message": "PyTorch not available."}

        returns = prices.pct_change().dropna()
        assets = prices.columns.tolist()

        forecasts = {}

        # Train a model for each asset (Simplified)
        # Ideally, use a multivariate LSTM

        for asset in assets:
            asset_returns = returns[asset].values
            if len(asset_returns) < 30:
                forecasts[asset] = 0.0
                continue

            # Prepare data: use last 10 days to predict next
            seq_len = 10
            X, y = [], []
            for i in range(len(asset_returns) - seq_len):
                X.append(asset_returns[i:i+seq_len])
                y.append(asset_returns[i+seq_len])

            X_train = torch.tensor(np.array(X), dtype=torch.float32).unsqueeze(-1) # (N, seq, 1)
            y_train = torch.tensor(np.array(y), dtype=torch.float32).unsqueeze(-1)

            # Reset model for this asset (or use shared model)
            # For simplicity, re-init
            model = LSTMModel()
            optimizer = optim.Adam(model.parameters(), lr=0.01)
            loss_fn = nn.MSELoss()

            # Train
            model.train()
            for _ in range(50):
                optimizer.zero_grad()
                out = model(X_train)
                loss = loss_fn(out, y_train)
                loss.backward()
                optimizer.step()

            # Predict next
            model.eval()
            last_seq = torch.tensor(asset_returns[-seq_len:], dtype=torch.float32).view(1, seq_len, 1)
            with torch.no_grad():
                pred = model(last_seq).item()

            forecasts[asset] = pred

        # Alloc logic: softmax of positive returns, or 0 if negative
        # 1. Filter positive
        pos_forecasts = {k: v for k, v in forecasts.items() if v > 0}

        if not pos_forecasts:
            # All negative? Go to cash (empty allocation or equal weight defensive)
            allocation = {a: 0.0 for a in assets}
        else:
            total_score = sum(pos_forecasts.values())
            allocation = {k: round(v / total_score, 4) for k, v in pos_forecasts.items()}
            # Fill zeros
            for a in assets:
                if a not in allocation:
                    allocation[a] = 0.0

        return {
            "status": "success",
            "method": "AI Forecast (LSTM)",
            "allocation": allocation,
            "forecasts": {k: round(v, 6) for k, v in forecasts.items()}
        }

# Example usage
if __name__ == "__main__":
    # Mock data
    dates = pd.date_range(start='2023-01-01', periods=100)
    data = pd.DataFrame({
        'AAPL': np.random.normal(150, 5, 100),
        'GOOG': np.random.normal(120, 4, 100),
        'MSFT': np.random.normal(250, 8, 100)
    }, index=dates)

    agent = PortfolioOptimizationAgent()

    import asyncio
    print("--- Mean Variance ---")
    res_mv = asyncio.run(agent.execute(historical_prices=data, method='mean_variance'))
    print(res_mv)

    print("\n--- AI Forecast ---")
    res_ai = asyncio.run(agent.execute(historical_prices=data, method='ai_forecast'))
    print(res_ai)

class AIPoweredPortfolioOptimizationAgent(PortfolioOptimizationAgent):
    """
    Legacy wrapper for PortfolioOptimizationAgent to maintain backward compatibility.
    """
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config=config)
