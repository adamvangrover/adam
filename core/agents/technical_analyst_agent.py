from __future__ import annotations
from typing import Any, Dict, Union, Optional, List
import logging
import asyncio
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pickle
import os

from core.agents.agent_base import AgentBase
from core.schemas.agent_schema import AgentInput, AgentOutput
from core.data_sources.data_fetcher import DataFetcher

class TechnicalAnalystAgent(AgentBase):
    """
    Agent responsible for technical analysis of financial assets.
    """
    def __init__(self, config: Dict[str, Any], constitution: Optional[Dict[str, Any]] = None, kernel: Optional[Any] = None):
        super().__init__(config, constitution, kernel)
        self.data_fetcher = DataFetcher()
        self.model_path = config.get('model_path', 'models/technical_model.pkl')

        # Try loading a pre-trained model
        try:
            self.model = self.load_model(self.model_path)
        except Exception as e:
            logging.warning(f"Could not load technical model: {e}")
            self.model = None

    async def execute(self, input_data: Union[str, AgentInput, Dict[str, Any]] = None, **kwargs) -> Union[Dict[str, Any], AgentOutput]:
        """
        Executes the technical analysis.
        Supports both legacy string/dict input and new AgentInput schema.
        """
        # 1. Input Normalization
        query = ""
        is_standard_mode = False
        price_data = None
        train_model = False

        if input_data is not None:
            if isinstance(input_data, AgentInput):
                query = input_data.query
                is_standard_mode = True
                if input_data.context:
                    self.set_context(input_data.context)
                    price_data = input_data.context.get("price_data")
            elif isinstance(input_data, str):
                query = input_data
            elif isinstance(input_data, dict):
                query = input_data.get("query", "")
                price_data = input_data.get("price_data")
                train_model = input_data.get("train_model", False)
                if "context" in input_data:
                    self.set_context(input_data["context"])
                kwargs.update(input_data)

        # Fallback to kwargs
        if not price_data:
            price_data = kwargs.get("price_data")
        if not train_model:
            train_model = kwargs.get("train_model", False)

        # If no price_data provided, try to fetch using query (assuming it's a ticker)
        if not price_data and query:
            # Simple heuristic: if query looks like a ticker (short, no spaces)
            if " " not in query and len(query) < 10:
                logging.info(f"Fetching historical data for {query}...")
                loop = asyncio.get_running_loop()
                price_data = await loop.run_in_executor(None, self.data_fetcher.fetch_historical_data, query)
            else:
                 logging.warning(f"Query '{query}' does not look like a ticker, and no price_data provided.")

        logging.info(f"TechnicalAnalystAgent execution started. Mode: {'Standard' if is_standard_mode else 'Legacy'}")

        if not price_data:
            error_msg = "No price data available for analysis."
            logging.error(error_msg)
            if is_standard_mode:
                return AgentOutput(
                    answer="Analysis failed: No price data provided or found.",
                    confidence=0.0,
                    metadata={"error": error_msg}
                )
            return {"error": error_msg}

        # Analyze
        try:
             # analyze_price_data is sync, run it directly or in executor if heavy
             signal = self.analyze_price_data(price_data, train_model)

             # Enrich result
             result = {
                 "signal": signal,
                 "ticker": query if query else "Unknown",
                 "data_points": len(price_data) if isinstance(price_data, list) else len(price_data) if hasattr(price_data, '__len__') else 0
             }

             if is_standard_mode:
                 return self._format_output(result, query)

             return result

        except Exception as e:
            logging.exception(f"Error during technical analysis: {e}")
            if is_standard_mode:
                return AgentOutput(
                    answer=f"Analysis failed: {str(e)}",
                    confidence=0.0,
                    metadata={"error": str(e)}
                )
            return {"error": str(e)}

    def _format_output(self, result: Dict[str, Any], query: str) -> AgentOutput:
        signal = result.get("signal", "hold")
        answer = f"Technical Analysis for {query}:\n"
        answer += f"Signal: {signal.upper()}\n"

        return AgentOutput(
            answer=answer,
            sources=["Historical Price Data"],
            confidence=0.8 if signal != "hold" else 0.5,
            metadata=result
        )

    def analyze_price_data(self, price_data, train_model=False):
        print("Analyzing price data...")

        # 1. Feature Engineering
        df = pd.DataFrame(price_data)

        # Ensure we have numeric data
        cols = ['close', 'open', 'high', 'low', 'volume']
        for col in cols:
             if col in df.columns:
                 df[col] = pd.to_numeric(df[col], errors='coerce')

        if 'close' not in df.columns:
             # Try fallback to capitalized
             if 'Close' in df.columns:
                 df['close'] = df['Close']
             else:
                 raise ValueError("Price data must contain 'close' column.")

        df['SMA_50'] = df['close'].rolling(window=50).mean()
        df['SMA_200'] = df['close'].rolling(window=200).mean()
        df['RSI'] = self.calculate_rsi(df['close'])
        # ... (calculate other technical indicators and features)

        # 2. ML Model Training (if requested)
        if train_model:
            try:
                features, labels = self.prepare_training_data(df)
                if not features.empty and not labels.empty:
                    self.model = RandomForestClassifier()  # Or another suitable model
                    self.model.fit(features, labels)
                    self.save_model(self.model, self.model_path)
            except Exception as e:
                logging.error(f"Training failed: {e}")

        # 3. Signal Generation
        if self.model:
            try:
                features = df.dropna().drop(['signal'], axis=1, errors='ignore')  # Drop 'signal' if present
                # Ensure features match model input?
                # For simplicity, we just try/except
                # In real scenario, we need robust feature alignment.
                if not features.empty:
                    signal = self.model.predict(features.iloc[[-1]])[0]  # Predict on the latest data point
                    print(f"ML-Based Trading Signal: {signal}")
                else:
                    signal = "hold"
            except Exception as e:
                logging.warning(f"Model prediction failed: {e}. Defaulting to hold.")
                signal = "hold"
        else:
            # Fallback simple logic
            last_close = df['close'].iloc[-1]
            last_sma50 = df['SMA_50'].iloc[-1]

            if not np.isnan(last_sma50):
                if last_close > last_sma50:
                    signal = "buy"
                else:
                    signal = "sell"
            else:
                signal = "hold"

            print(f"Simple Logic Signal: {signal}")

        # 4. Technical Indicator Analysis
        # ... (analyze technical indicators and patterns)

        return signal

    def calculate_rsi(self, prices, period=14):
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def prepare_training_data(self, df):
        # Placeholder for real feature engineering
        # Create a dummy target 'signal' based on future price direction
        df['target'] = np.where(df['close'].shift(-1) > df['close'], 'buy', 'sell')

        features = df[['SMA_50', 'SMA_200', 'RSI']].dropna()
        # Align labels
        labels = df['target'].loc[features.index]

        return features, labels

    def load_model(self, model_path):
        if os.path.exists(model_path):
            with open(model_path, 'rb') as f:
                return pickle.load(f)
        return None

    def save_model(self, model, model_path):
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
