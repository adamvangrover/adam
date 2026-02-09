import asyncio
import logging
import yfinance as yf
import pandas as pd
import ta
import random
from typing import List

# Import BaseAgent
try:
    from core.v30_architecture.python_intelligence.agents.base_agent import BaseAgent
except ImportError:
    # If run as script
    import sys
    import os
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../')))
    from core.v30_architecture.python_intelligence.agents.base_agent import BaseAgent

logger = logging.getLogger("QuantitativeAnalyst")

class QuantitativeAnalyst(BaseAgent):
    """
    A V30 agent that performs real-time technical analysis on market data.
    """
    def __init__(self):
        super().__init__("Quant-V30", "market_analysis")
        self.tickers = ["SPY", "QQQ", "IWM", "BTC-USD", "ETH-USD"]

    async def run(self):
        logger.info(f"{self.name} started. Monitoring: {self.tickers}")
        while True:
            try:
                # Pick a random ticker to analyze to distribute load/API calls
                ticker_symbol = random.choice(self.tickers)
                await self.analyze_ticker(ticker_symbol)
            except Exception as e:
                logger.error(f"Error in QuantitativeAnalyst loop: {e}")

            # Wait a bit before next analysis
            await asyncio.sleep(random.uniform(5.0, 10.0))

    async def analyze_ticker(self, symbol: str):
        try:
            # Run blocking yfinance call in a thread to avoid blocking the async loop
            loop = asyncio.get_event_loop()
            df = await loop.run_in_executor(None, self._fetch_data, symbol)

            if df is None or df.empty:
                logger.warning(f"No data for {symbol}")
                return

            # Calculate Indicators
            # RSI
            rsi = ta.momentum.RSIIndicator(close=df["Close"], window=14)
            current_rsi = rsi.rsi().iloc[-1]

            # SMA
            sma_20 = ta.trend.SMAIndicator(close=df["Close"], window=20)
            current_sma = sma_20.sma_indicator().iloc[-1]

            # Bollinger Bands
            bb = ta.volatility.BollingerBands(close=df["Close"], window=20, window_dev=2)
            bb_high = bb.bollinger_hband().iloc[-1]
            bb_low = bb.bollinger_lband().iloc[-1]
            current_price = df["Close"].iloc[-1]

            # Construct Payload
            payload = {
                "symbol": symbol,
                "price": round(float(current_price), 2),
                "rsi": round(float(current_rsi), 2) if not pd.isna(current_rsi) else None,
                "sma_20": round(float(current_sma), 2) if not pd.isna(current_sma) else None,
                "bb_position": self._get_bb_position(current_price, bb_high, bb_low),
                "signal": self._generate_signal(current_rsi)
            }

            await self.emit("technical_analysis", payload)
            logger.info(f"Analyzed {symbol}: RSI={payload['rsi']}")

        except Exception as e:
            logger.error(f"Failed to analyze {symbol}: {e}")

    def _fetch_data(self, symbol):
        try:
            # Download data
            # Use 'max' or '1d' to get enough data for indicators
            ticker = yf.Ticker(symbol)
            # Fetch minimal necessary history
            df = ticker.history(period="5d", interval="15m")
            return df
        except Exception as e:
            logger.error(f"yfinance error for {symbol}: {e}")
            return None

    def _get_bb_position(self, price, high, low):
        if pd.isna(high) or pd.isna(low):
            return "unknown"
        if price > high:
            return "above_upper"
        elif price < low:
            return "below_lower"
        else:
            return "within_bands"

    def _generate_signal(self, rsi):
        if pd.isna(rsi):
            return "NEUTRAL"
        if rsi > 70:
            return "OVERBOUGHT"
        elif rsi < 30:
            return "OVERSOLD"
        return "NEUTRAL"
