import asyncio
import logging
import numpy as np
import scipy.stats as stats
import yfinance as yf
import pandas as pd
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

# Import BaseAgent
try:
    from core.v30_architecture.python_intelligence.agents.base_agent import BaseAgent
except ImportError:
    # If run as script
    import sys
    import os
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../')))
    from core.v30_architecture.python_intelligence.agents.base_agent import BaseAgent

logger = logging.getLogger("RiskGuardian")

class RiskMetrics(BaseModel):
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    portfolio_var_95: float
    portfolio_cvar_95: float
    volatility: float
    sharpe_ratio: float
    max_drawdown: float
    correlation_matrix: Dict[str, Dict[str, float]]
    alerts: List[str]

class RiskGuardian(BaseAgent):
    """
    A V30 agent that monitors portfolio risk in real-time.
    Calculates VaR, CVaR, and correlation breakdowns.
    """
    def __init__(self):
        super().__init__("RiskGuardian-V30", "risk_management")
        # Default portfolio for monitoring (can be dynamic in future)
        self.portfolio = ["SPY", "QQQ", "IWM", "BTC-USD", "ETH-USD", "VIX"]
        # Simplified equal-ish weight (approximate)
        self.weights = np.array([0.2, 0.2, 0.1, 0.2, 0.2, 0.1])

    async def run(self):
        logger.info(f"{self.name} started. Monitoring Risk for: {self.portfolio}")
        while True:
            try:
                # 1. Fetch Data
                data = await self._fetch_historical_data()

                if data is not None and not data.empty:
                    # 2. Calculate Risk
                    metrics = self._calculate_risk_metrics(data)

                    # 3. Emit Assessment
                    if metrics:
                        await self.emit("risk_assessment", metrics.model_dump())

                        # Emit specific alerts if high risk
                        if metrics.portfolio_var_95 > 0.05: # >5% daily VaR is high
                            await self.emit("risk_alert", {
                                "level": "HIGH",
                                "message": f"Portfolio VaR spiked to {metrics.portfolio_var_95:.2%}",
                                "timestamp": datetime.now(timezone.utc).isoformat()
                            })

            except Exception as e:
                logger.error(f"Error in RiskGuardian loop: {e}")

            # Sleep for 5 minutes
            await asyncio.sleep(300)

    async def _fetch_historical_data(self) -> Optional[pd.DataFrame]:
        try:
            loop = asyncio.get_event_loop()
            # Download 1 year of data
            # yfinance returns a MultiIndex columns if group_by='ticker'
            df = await loop.run_in_executor(None,
                lambda: yf.download(self.portfolio, period="1y", interval="1d", group_by='ticker', progress=False)
            )

            if df is None or df.empty:
                return None

            # Extract 'Close' prices
            close_prices = pd.DataFrame()
            if isinstance(df.columns, pd.MultiIndex):
                for ticker in self.portfolio:
                    try:
                        # yfinance structure: (Ticker, PriceType) e.g. ('SPY', 'Close')
                        # or (PriceType, Ticker) depending on version/args, but group_by='ticker' usually gives Ticker first
                        if ticker in df.columns:
                            close_prices[ticker] = df[ticker]['Close']
                    except KeyError:
                        continue
            else:
                 # Single ticker case (unlikely given init) or flat columns
                 # If flat, columns might be 'Close' (if 1 ticker) or 'Close', 'Open' etc.
                 if 'Close' in df.columns:
                    # Ensure DataFrame and proper column name
                    temp = df[['Close']]
                    if len(self.portfolio) == 1:
                        temp.columns = [self.portfolio[0]]
                        close_prices = temp
                    else:
                        # Fallback: If we can't map to ticker, ignore to prevent crashes
                        pass

            return close_prices.dropna()
        except Exception as e:
            logger.error(f"Failed to fetch data: {e}")
            return None

    def _calculate_risk_metrics(self, prices: pd.DataFrame) -> Optional[RiskMetrics]:
        try:
            # Calculate daily returns
            returns = prices.pct_change().dropna()

            if returns.empty:
                return None

            # Portfolio Returns (assuming daily rebalancing to fixed weights for simplicity)
            # Normalize weights if some tickers failed
            valid_tickers = returns.columns.tolist()
            weights = []
            for t in valid_tickers:
                # Find index in self.portfolio
                if t in self.portfolio:
                    idx = self.portfolio.index(t)
                    weights.append(self.weights[idx])
                else:
                    weights.append(0)

            weights = np.array(weights)
            if weights.sum() == 0:
                return None
            weights = weights / weights.sum()

            portfolio_returns = returns.dot(weights)

            # 1. Value at Risk (VaR) 95%
            # Historical Method: 5th percentile
            var_95 = np.percentile(portfolio_returns, 5) * -1

            # 2. Conditional VaR (CVaR) 95% (Expected Shortfall)
            # Average of returns worse than VaR
            loss_distribution = portfolio_returns[portfolio_returns <= -var_95]
            if loss_distribution.empty:
                 cvar_95 = var_95
            else:
                 cvar_95 = loss_distribution.mean() * -1

            # 3. Volatility (Annualized)
            volatility = portfolio_returns.std() * np.sqrt(252)

            # 4. Sharpe Ratio (Assume Risk Free Rate = 4%)
            rf = 0.04
            sharpe = (portfolio_returns.mean() * 252 - rf) / volatility if volatility > 0 else 0

            # 5. Max Drawdown
            cumulative = (1 + portfolio_returns).cumprod()
            peak = cumulative.cummax()
            drawdown = (cumulative - peak) / peak
            max_drawdown = drawdown.min()

            # 6. Correlation Matrix
            # Convert keys to string just in case
            corr_matrix = returns.corr().to_dict()
            # Ensure JSON serializable (sometimes nan/inf can sneak in, but pydantic handles some)

            # Alerts
            alerts = []
            if var_95 > 0.03:
                alerts.append("High VaR Detected")
            if max_drawdown < -0.20:
                alerts.append("Deep Drawdown State")
            if volatility > 0.25:
                alerts.append("High Volatility Regime")

            return RiskMetrics(
                portfolio_var_95=round(float(var_95), 4),
                portfolio_cvar_95=round(float(cvar_95), 4) if not np.isnan(cvar_95) else 0.0,
                volatility=round(float(volatility), 4),
                sharpe_ratio=round(float(sharpe), 4),
                max_drawdown=round(float(max_drawdown), 4),
                correlation_matrix=corr_matrix,
                alerts=alerts
            )

        except Exception as e:
            logger.error(f"Calculation error: {e}")
            return None
