from core.agents.agent_base import AgentBase
from pydantic import BaseModel, Field
from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np
import logging

class BacktestInput(BaseModel):
    """
    Input schema for the StrategyBacktestAgent.
    """
    strategy_name: str = Field(..., description="Name of the strategy to backtest (e.g., 'SMA_CROSSOVER', 'MEAN_REVERSION').")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Parameters specific to the strategy (e.g., {'short_window': 20, 'long_window': 50}).")
    ticker: Optional[str] = Field(None, description="Ticker symbol to backtest (e.g., 'AAPL').")
    start_date: Optional[str] = Field(None, description="Start date for the backtest (YYYY-MM-DD).")
    end_date: Optional[str] = Field(None, description="End date for the backtest (YYYY-MM-DD).")
    initial_capital: float = Field(10000.0, description="Starting capital for the backtest.")
    data: Optional[List[Dict[str, Any]]] = Field(None, description="Optional raw data (list of dicts with 'Date', 'Close', etc.). If provided, skips fetching.")

class BacktestOutput(BaseModel):
    """
    Output schema for the StrategyBacktestAgent.
    """
    strategy_name: str = Field(..., description="Name of the strategy executed.")
    metrics: Dict[str, float] = Field(..., description="Key performance metrics (Sharpe, Drawdown, Return).")
    equity_curve: List[float] = Field(..., description="Time series of portfolio value.")
    trades: List[Dict[str, Any]] = Field(default_factory=list, description="List of executed trades.")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata.")

class StrategyBacktestAgent(AgentBase):
    """
    Agent responsible for backtesting trading strategies against historical data.
    """

    async def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Executes a backtest based on the provided input.

        Args:
            input_data (Dict[str, Any]): Dictionary matching BacktestInput schema.

        Returns:
            Dict[str, Any]: Dictionary matching BacktestOutput schema.
        """
        logging.info(f"StrategyBacktestAgent: Received input: {input_data}")

        # Validate input
        try:
            validated_input = BacktestInput(**input_data)
        except Exception as e:
            logging.error(f"StrategyBacktestAgent: Input validation failed: {e}")
            raise ValueError(f"Invalid input for StrategyBacktestAgent: {e}")

        # Fetch Data
        df = self._fetch_data(validated_input)

        if df is None or df.empty:
            logging.warning("StrategyBacktestAgent: No data available for backtest.")
            return BacktestOutput(
                strategy_name=validated_input.strategy_name,
                metrics={},
                equity_curve=[],
                trades=[],
                metadata={"error": "No data found"}
            ).model_dump()

        # Run Backtest
        results = self._run_backtest(df, validated_input)

        # Calculate Metrics
        metrics = self._calculate_metrics(results['equity_curve'], results['trades'], validated_input.initial_capital)

        output = BacktestOutput(
            strategy_name=validated_input.strategy_name,
            metrics=metrics,
            equity_curve=results['equity_curve'],
            trades=results['trades'],
            metadata={"ticker": validated_input.ticker}
        )

        logging.info(f"StrategyBacktestAgent: Backtest complete. metrics={metrics}")
        return output.model_dump()

    def _fetch_data(self, input_data: BacktestInput) -> pd.DataFrame:
        """
        Fetches historical data. If 'data' is provided in input, uses that.
        Otherwise, generates synthetic data (mock).
        """
        if input_data.data:
            df = pd.DataFrame(input_data.data)
            if 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date'])
                df.set_index('Date', inplace=True)
            return df

        # Mock Data Generation (Synthetic)
        logging.info("StrategyBacktestAgent: Generating synthetic data.")
        dates = pd.date_range(start=input_data.start_date or '2023-01-01',
                              end=input_data.end_date or '2023-12-31', freq='D')
        n = len(dates)

        # Random Walk with Drift
        np.random.seed(42)
        returns = np.random.normal(0.0005, 0.02, n) # Mean return 0.05%, Vol 2%
        price_path = 100 * np.cumprod(1 + returns)

        df = pd.DataFrame({'Close': price_path}, index=dates)
        return df

    def _run_backtest(self, df: pd.DataFrame, input_data: BacktestInput) -> Dict[str, Any]:
        """
        Runs the core backtesting logic.
        """
        strategy = input_data.strategy_name
        params = input_data.parameters
        initial_capital = input_data.initial_capital

        capital = initial_capital
        position = 0 # 0 = Flat, >0 = Long
        equity_curve = []
        trades = []

        # Strategy Indicators
        if strategy == 'SMA_CROSSOVER':
            short_window = params.get('short_window', 20)
            long_window = params.get('long_window', 50)
            df['Short_MA'] = df['Close'].rolling(window=short_window).mean()
            df['Long_MA'] = df['Close'].rolling(window=long_window).mean()

        elif strategy == 'MEAN_REVERSION':
             window = params.get('window', 20)
             std_dev_mult = params.get('std_dev_mult', 2.0)
             df['MA'] = df['Close'].rolling(window=window).mean()
             df['Std'] = df['Close'].rolling(window=window).std()
             df['Upper'] = df['MA'] + (df['Std'] * std_dev_mult)
             df['Lower'] = df['MA'] - (df['Std'] * std_dev_mult)

        # Simulation Loop
        for i in range(len(df)):
            date = df.index[i]
            price = df['Close'].iloc[i]

            # Skip if indicators are NaN (warmup period)
            if strategy == 'SMA_CROSSOVER':
                 if pd.isna(df['Short_MA'].iloc[i]) or pd.isna(df['Long_MA'].iloc[i]):
                     equity_curve.append(capital)
                     continue

            signal = 0 # -1 Sell, 0 Hold, 1 Buy

            # Logic
            if strategy == 'SMA_CROSSOVER':
                short_ma = df['Short_MA'].iloc[i]
                long_ma = df['Long_MA'].iloc[i]
                prev_short = df['Short_MA'].iloc[i-1]
                prev_long = df['Long_MA'].iloc[i-1]

                # Crossover
                if short_ma > long_ma and prev_short <= prev_long:
                    signal = 1
                elif short_ma < long_ma and prev_short >= prev_long:
                    signal = -1

            elif strategy == 'MEAN_REVERSION':
                if pd.isna(df['Lower'].iloc[i]): # Check only one logic band
                    equity_curve.append(capital)
                    continue

                if price < df['Lower'].iloc[i]:
                    signal = 1 # Buy Dip
                elif price > df['Upper'].iloc[i]:
                    signal = -1 # Sell Rip

            # Execution
            if signal == 1 and position == 0:
                # Buy
                shares = capital / price
                position = shares
                capital = 0
                trades.append({'date': str(date), 'type': 'BUY', 'price': price, 'shares': shares})

            elif signal == -1 and position > 0:
                # Sell
                capital = position * price
                trades.append({'date': str(date), 'type': 'SELL', 'price': price, 'shares': position, 'pnl': capital - (trades[-1]['shares'] * trades[-1]['price'])})
                position = 0

            # Update Equity
            current_equity = capital + (position * price)
            equity_curve.append(current_equity)

        return {
            "equity_curve": equity_curve,
            "trades": trades
        }

    def _calculate_metrics(self, equity_curve: List[float], trades: List[Dict[str, Any]], initial_capital: float) -> Dict[str, float]:
        """
        Calculates performance metrics.
        """
        if not equity_curve:
            return {"total_return": 0.0, "sharpe_ratio": 0.0, "max_drawdown": 0.0}

        final_equity = equity_curve[-1]
        total_return = (final_equity - initial_capital) / initial_capital

        # Max Drawdown
        equity_series = pd.Series(equity_curve)
        rolling_max = equity_series.cummax()
        drawdown = (equity_series - rolling_max) / rolling_max
        max_drawdown = drawdown.min()

        # Sharpe Ratio (Daily)
        returns = equity_series.pct_change().dropna()
        if returns.std() == 0:
            sharpe = 0.0
        else:
            sharpe = (returns.mean() / returns.std()) * np.sqrt(252) # Annualized

        return {
            "total_return": round(total_return, 4),
            "max_drawdown": round(max_drawdown, 4),
            "sharpe_ratio": round(sharpe, 4),
            "trade_count": len(trades)
        }

    def get_skill_schema(self) -> Dict[str, Any]:
        return {
            "name": "StrategyBacktestAgent",
            "description": "Runs historical backtests for trading strategies.",
            "skills": [
                {
                    "name": "run_backtest",
                    "description": "Executes a backtest.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "strategy_name": {"type": "string"},
                            "parameters": {"type": "object"},
                            "ticker": {"type": "string"}
                        },
                        "required": ["strategy_name"]
                    }
                }
            ]
        }
