from typing import Any, Dict, List, Optional, Tuple
import logging
import numpy as np
import pandas as pd
from core.agents.agent_base import AgentBase
import yfinance as yf

# Configure logging
logger = logging.getLogger("MarketRegimeAgent")

class MarketRegimeAgent(AgentBase):
    """
    Agent responsible for classifying the current market regime (e.g., Bull, Bear, Choppy, Volatile)
    using statistical metrics such as Hurst Exponent, ADX, and Volatility ratios.
    This acts as a 'Force Multiplier' for other agents by providing context on *how* to trade.
    """

    def __init__(self, config: Dict[str, Any], **kwargs):
        super().__init__(config, **kwargs)
        self.symbol = config.get("symbol", "SPY")
        self.lookback_period = config.get("lookback_period", 100)
        self.adx_period = config.get("adx_period", 14)

        # Thresholds
        self.hurst_threshold = config.get("hurst_threshold", 0.5) # > 0.5 Trending, < 0.5 Mean Reverting
        self.adx_threshold = config.get("adx_threshold", 25) # > 25 Strong Trend
        self.volatility_percentile = config.get("volatility_percentile", 0.8) # Top 20% vol = High Vol regime

    async def execute(self, **kwargs) -> Dict[str, Any]:
        """
        Executes the regime analysis.

        Args:
            symbol (str, optional): The ticker to analyze. Overrides config.

        Returns:
            Dict containing the identified regime and metrics.
        """
        symbol = kwargs.get("symbol", self.symbol)
        logger.info(f"MarketRegimeAgent analyzing {symbol}...")

        # 1. Fetch Data
        df = self._fetch_data(symbol)
        if df is None or df.empty:
            logger.error(f"No data found for {symbol}")
            return {"status": "error", "message": "No data found"}

        # 2. Calculate Metrics
        hurst = self._calculate_hurst(df['Close'])
        adx = self._calculate_adx(df)
        vol_regime = self._calculate_volatility_regime(df['Close'])

        # 3. Classify Regime
        regime = self._classify_regime(hurst, adx, vol_regime)

        result = {
            "symbol": symbol,
            "regime": regime,
            "metrics": {
                "hurst_exponent": float(hurst),
                "adx": float(adx),
                "volatility_state": vol_regime
            },
            "status": "success",
            "timestamp": pd.Timestamp.now().isoformat()
        }

        logger.info(f"Regime identified: {regime} (Hurst={hurst:.2f}, ADX={adx:.2f})")
        return result

    def _fetch_data(self, symbol: str) -> Optional[pd.DataFrame]:
        """Fetches historical data using yfinance."""
        try:
            # We need enough data for Hurst (requires reasonable sample size, e.g. 100+)
            # and ADX (requires lookback + smoothing)
            period = "1y"
            ticker = yf.Ticker(symbol)
            df = ticker.history(period=period)
            if df.empty:
                return None
            return df
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {e}")
            return None

    def _calculate_hurst(self, series: pd.Series) -> float:
        """
        Calculates the Hurst Exponent using the Rescaled Range (R/S) analysis.
        H < 0.5: Mean Reverting
        H ~ 0.5: Random Walk
        H > 0.5: Trending
        """
        try:
            ts = series.values
            if len(ts) < 20:
                return 0.5 # Not enough data

            # Alternative: simplified R/S
            # 1. Calculate logarithmic returns
            returns = np.diff(np.log(ts))

            # We will use a standard library approach if available, but since we are manual:
            # Let's use the variance method which is stable.
            # Var(t) proportional to t^(2H)

            lags = range(2, 20)
            variances = []
            for lag in lags:
                # Price difference at lag
                diffs = ts[lag:] - ts[:-lag]
                variances.append(np.var(diffs))

            # Fit line to log(var) vs log(lag)
            # Slope = 2H
            poly = np.polyfit(np.log(lags), np.log(variances), 1)
            hurst = poly[0] / 2.0

            return min(max(hurst, 0.0), 1.0) # Clamp

        except Exception as e:
            logger.error(f"Error calculating Hurst: {e}")
            return 0.5

    def _calculate_adx(self, df: pd.DataFrame) -> float:
        """Calculates Average Directional Index (ADX)."""
        try:
            n = self.adx_period
            high = df['High']
            low = df['Low']
            close = df['Close']

            # True Range
            tr1 = high - low
            tr2 = abs(high - close.shift(1))
            tr3 = abs(low - close.shift(1))
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

            # Directional Movement
            up_move = high - high.shift(1)
            down_move = low.shift(1) - low

            plus_dm = pd.Series(np.where((up_move > down_move) & (up_move > 0), up_move, 0.0), index=df.index)
            minus_dm = pd.Series(np.where((down_move > up_move) & (down_move > 0), down_move, 0.0), index=df.index)

            # Smooth
            # Wilder's Smoothing: Prev + (Curr - Prev)/n
            # We can use pandas ewm with alpha=1/n
            atr = tr.ewm(alpha=1/n, min_periods=n).mean()
            plus_di = 100 * (plus_dm.ewm(alpha=1/n, min_periods=n).mean() / atr)
            minus_di = 100 * (minus_dm.ewm(alpha=1/n, min_periods=n).mean() / atr)

            dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
            adx = dx.ewm(alpha=1/n, min_periods=n).mean()

            return adx.iloc[-1]

        except Exception as e:
            logger.error(f"Error calculating ADX: {e}")
            return 0.0

    def _calculate_volatility_regime(self, series: pd.Series) -> str:
        """
        Determines if volatility is 'LOW', 'NORMAL', or 'HIGH' based on history.
        """
        try:
            returns = series.pct_change().dropna()
            current_vol = returns.tail(20).std() # Short term
            hist_vol = returns.std() # Long term

            ratio = current_vol / hist_vol if hist_vol > 0 else 1.0

            if ratio > 1.5:
                return "HIGH"
            elif ratio < 0.5:
                return "LOW"
            else:
                return "NORMAL"
        except Exception:
            return "NORMAL"

    def _classify_regime(self, hurst: float, adx: float, vol: str) -> str:
        """Logic gate for Regime Classification."""

        # 1. Volatility Override
        if vol == "HIGH":
            return "HIGH_VOLATILITY_CRASH_RISK"

        # 2. Trend Strength (ADX)
        if adx > self.adx_threshold:
            # Strong Trend. Check Hurst to confirm persistence or mean reversion within trend?
            # Usually ADX > 25 implies Trend.
            # Use Hurst to confirm 'Clean' trend vs 'Noisy' trend?
            return "STRONG_TREND"

        # 3. Hurst Differentiation (for low ADX)
        if hurst < 0.4:
            return "MEAN_REVERSION" # Choppy, range-bound
        elif hurst > 0.6:
            # Trending but ADX is low? Could be nascent trend.
            return "NASCENT_TREND"

        return "UNDEFINED_NOISE"

if __name__ == "__main__":
    # Test harness
    logging.basicConfig(level=logging.INFO)
    agent = MarketRegimeAgent({})
    # This requires internet and yfinance to work
    import asyncio
    asyncio.run(agent.execute(symbol="BTC-USD"))
