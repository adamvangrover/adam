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
                # Analyze ALL tickers in parallel/batch
                await self.analyze_all_tickers()
            except Exception as e:
                logger.error(f"Error in QuantitativeAnalyst loop: {e}")

            # Wait a bit before next analysis (Batch processing allows longer intervals)
            await asyncio.sleep(random.uniform(15.0, 30.0))

    async def analyze_all_tickers(self):
        try:
            # Run blocking yfinance call in a thread
            loop = asyncio.get_event_loop()
            data = await loop.run_in_executor(None, self._fetch_batch_data, self.tickers)

            if data is None or data.empty:
                logger.warning("No batch data received")
                return

            # Iterate over tickers
            for symbol in self.tickers:
                try:
                    # Extract DF for symbol
                    if isinstance(data.columns, pd.MultiIndex):
                        # yf.download(group_by='ticker') returns columns like (Ticker, Open), (Ticker, Close)...
                        try:
                            # Using xs to get cross section for the ticker
                            # Or simple indexing if group_by='ticker'
                            df = data[symbol]
                        except KeyError:
                            # Ticker might not be in the result if download failed for it
                            continue
                    else:
                        # If only 1 ticker was downloaded (unlikely given self.tickers list)
                        df = data

                    # Drop NaNs (crucial for mixed asset classes alignment)
                    df = df.dropna(subset=['Close'])

                    if df.empty:
                        continue

                    # Calculate and Emit
                    await self._process_ticker_data(symbol, df)

                except Exception as e:
                    logger.error(f"Failed to process {symbol} in batch: {e}")

        except Exception as e:
            logger.error(f"Batch analysis failed: {e}")

    def _fetch_batch_data(self, symbols: List[str]):
        try:
            # Download data for all tickers at once
            # group_by='ticker' structures the result by ticker
            # threads=True is default but explicit is good
            df = yf.download(symbols, period="5d", interval="15m", group_by='ticker', threads=True, progress=False)
            return df
        except Exception as e:
            logger.error(f"yfinance batch error: {e}")
            return None

    async def _process_ticker_data(self, symbol: str, df: pd.DataFrame):
        try:
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
            logger.error(f"Failed to calculate indicators for {symbol}: {e}")

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
