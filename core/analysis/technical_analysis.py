# core/analysis/technical_analysis.py

from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.trend import MACD, ADXIndicator, SMAIndicator
from ta.volatility import BollingerBands
from ta.volume import OnBalanceVolumeIndicator

from .trading_logic import sma_crossover_strategy

# ... (import other necessary libraries for technical indicators, patterns, and CDS spread analysis)

class TechnicalAnalyst:
    """
    Advanced technical analysis module for analyzing price data, CDS spreads,
    and other market signals, generating trading signals across various asset classes.
    
    This module integrates with Adam's agent framework to access data from various sources
    and collaborates with other agents like the CDSSpreadAgent for a comprehensive analysis.

    Features:
        - Calculates various technical indicators (SMA, RSI, MACD, Bollinger Bands, etc.)
        - Trains and utilizes a Random Forest model for signal generation
        - Integrates CDS spread analysis for credit risk assessment
        - Analyzes order book data and capital stack information (if available)
        - Provides functionalities for analyzing derivatives data
        - Expandable for incorporating additional technical analysis methods

    Future Enhancements:
        - Implement pattern recognition (e.g., candlestick patterns, chart patterns)
        - Integrate sentiment analysis from news and social media
        - Incorporate seasonality and cyclical patterns
        - Develop more sophisticated ML models (e.g., deep learning)
        - Add risk management and position sizing strategies
    """
    def __init__(self, config: Dict[str, Any]):
        """
        Initializes the TechnicalAnalyst with configuration parameters.

        Args:
            config: Dictionary containing configuration parameters.
                - data_sources: Dictionary of data sources.
                - model_path: Path to the trained ML model file.
                - cds_spread_agent: Instance of the CDSSpreadAgent.
                - ... (other agent instances and parameters)
        """
        self.data_sources = config.get('data_sources', {})
        self.model_path = config.get('model_path', 'models/technical_model.pkl')
        self.cds_spread_agent = config.get('cds_spread_agent', None)
        # ... (initialize other agents and data sources as needed)

        try:
            self.model = self.load_model(self.model_path)
        except FileNotFoundError:
            self.model = None

    def analyze_asset(self, asset_data: Dict[str, Any], train_model: bool = False) -> Dict[str, Any]:
        """
        Analyzes asset data (including price data, CDS spreads, etc.) and generates trading signals.

        Args:
            asset_data: Dictionary containing comprehensive asset data.
                - name: Name of the asset.
                - price_data: Historical price data (if available).
                - identifier: Identifier for CDS spread lookup (if applicable).
                - order_book: Order book data (if available).
                - capital_stack: Capital stack information (if applicable).
                - derivatives: Derivatives data (if applicable).
                - ... (other asset-specific data)
            train_model: Boolean indicating whether to train the ML model.

        Returns:
            Dictionary containing:
                - signal: Trading signal (e.g., 'buy', 'sell', 'hold').
                - technical_indicators: Dictionary of calculated technical indicators.
                - model_output: Raw output from the ML model (if available).
                - cds_analysis: Results of CDS spread analysis (if available).
                - ... (other analysis results)
        """
        print(f"Analyzing {asset_data.get('name', 'Unknown Asset')}...")
        signal = "hold"  # Default signal
        technical_indicators = {}
        model_output = None
        cds_analysis = None

        # 1. Price Data Analysis (if available)
        if 'price_data' in asset_data and asset_data['price_data']:
            price_data = asset_data['price_data']
            df = pd.DataFrame(price_data)

            # Feature Engineering (Expanded)
            df['SMA_50'] = SMAIndicator(close=df['close'], window=50).sma_indicator()
            df['SMA_200'] = SMAIndicator(close=df['close'], window=200).sma_indicator()
            df['RSI'] = RSIIndicator(close=df['close'], window=14).rsi()
            df['MACD'] = MACD(close=df['close'], window_slow=26, window_fast=12, window_sign=9).macd()
            df['BB_high'] = BollingerBands(close=df['close'], window=20, window_dev=2).bollinger_hband()
            df['BB_low'] = BollingerBands(close=df['close'], window=20, window_dev=2).bollinger_lband()
            df['OBV'] = OnBalanceVolumeIndicator(close=df['close'], volume=df['volume']).on_balance_volume()
            df['ADX'] = ADXIndicator(high=df['high'], low=df['low'], close=df['close'], window=14).adx()
            df['Stochastic_K'] = StochasticOscillator(high=df['high'], low=df['low'], close=df['close'], window=14).stoch()
            df['Stochastic_D'] = SMAIndicator(close=df['Stochastic_K'], window=3).sma_indicator()

            # ML Model Training (if requested)
            if train_model:
                features, labels = self.prepare_training_data(df)
                self.model = RandomForestClassifier()
                self.model.fit(features, labels)
                self.save_model(self.model, self.model_path)

            # Signal Generation
            technical_indicators = df.dropna().to_dict('records')[-1]  # Get latest indicator values
            if self.model:
                features = df.dropna().drop(['signal'], axis=1, errors='ignore')
                model_output = self.model.predict_proba(features)[-1]  # Get probabilities for each class
                signal = self.model.predict(features)[-1]
                print(f"ML-Based Trading Signal: {signal}")
            else:
                print("No trained model available. Defaulting to technical indicator analysis.")

            # Technical Indicator Analysis
            signal = self._analyze_technical_indicators(df, signal)

            # SMA Crossover Strategy
            crossover_signals = sma_crossover_strategy(df)
            if crossover_signals['position'][-1] == 1:
                signal = 'buy'
            elif crossover_signals['position'][-1] == -1:
                signal = 'sell'


        # 2. CDS Spread Analysis (if available)
        if 'identifier' in asset_data and self.cds_spread_agent:
            identifier = asset_data['identifier']  # Assuming an identifier like CUSIP or ISIN
            cds_analysis = self.cds_spread_agent.analyze_cds_spread(identifier)
            # ... (Incorporate CDS analysis into signal generation logic)
            print(f"CDS Spread Analysis: {cds_analysis}")

        # 3. Trading Level Information Analysis (if available)
        if 'order_book' in asset_data:
            order_book = asset_data['order_book']
            # ... (Analyze order book data for signals, e.g., bid-ask spread, order flow imbalance)
            print(f"Order Book Analysis: {order_book}")

        # 4. Capital Stack Analysis (if applicable)
        if 'capital_stack' in asset_data:
            capital_stack = asset_data['capital_stack']
            # ... (Analyze capital stack for risk and opportunities, e.g., debt levels, seniority)
            print(f"Capital Stack Analysis: {capital_stack}")

        # 5. Derivatives and Other Securities Analysis (if applicable)
        if 'derivatives' in asset_data:
            derivatives = asset_data['derivatives']
            # ... (Analyze derivatives data, e.g., options pricing, implied volatility)
            print(f"Derivatives Analysis: {derivatives}")

        # ... (Add more analysis modules as needed)

        return {
            "signal": signal,
            "technical_indicators": technical_indicators,
            "model_output": model_output,
            "cds_analysis": cds_analysis,
            # ... (other analysis results)
        }

    def prepare_training_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepares features and labels for training the ML model.

        Args:
            df: DataFrame containing price data and calculated technical indicators.

        Returns:
            Tuple containing:
                - features: DataFrame of features for training.
                - labels: Series of labels for training.
        """
        # Create labels based on future price movement
        df['signal'] = np.where(df['close'].shift(-1) > df['close'], 1, 0)
        features = df.dropna().drop(['signal', 'close'], axis=1)
        labels = df.dropna()['signal']
        return features, labels

    def load_model(self, model_path: str) -> Any:
        """
        Loads the trained ML model from file.

        Args:
            model_path: Path to the model file.

        Returns:
            Loaded ML model object.
        """
        import pickle
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        return model

    def save_model(self, model: Any, model_path: str) -> None:
        """
        Saves the trained ML model to file.

        Args:
            model: Trained ML model object.
            model_path: Path to save the model.
        """
        import pickle
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)

    def _analyze_technical_indicators(self, df: pd.DataFrame, ml_signal: str) -> str:
        """
        Analyzes technical indicators and patterns to generate trading signals.

        Args:
            df: DataFrame containing price data and calculated technical indicators.
            ml_signal: Trading signal generated by the ML model (if available).

        Returns:
            Trading signal ('buy', 'sell', or 'hold').
        """
        # Example: RSI overbought/oversold
        if df['RSI'][-1] > 70:
            print("RSI overbought.")
            return 'sell' if ml_signal != 'buy' else 'hold'
        elif df['RSI'][-1] < 30:
            print("RSI oversold.")
            return 'buy' if ml_signal != 'sell' else 'hold'

        # Example: Bollinger Band breakout
        if df['close'][-1] > df['BB_high'][-1]:
            print("Price breakout above Bollinger Band.")
            return 'buy' if ml_signal != 'sell' else 'hold'
        elif df['close'][-1] < df['BB_low'][-1]:
            print("Price breakout below Bollinger Band.")
            return 'sell' if ml_signal != 'buy' else 'hold'

        # Example: ADX and DI+ DI- for trend strength and direction
        if df['ADX'][-1] > 25:  # Strong trend
            if df['DI+'][-1] > df['DI-'][-1]:
                print("Strong uptrend detected.")
                return 'buy' if ml_signal != 'sell' else 'hold'
            else:
                print("Strong downtrend detected.")
                return 'sell' if ml_signal != 'buy' else 'hold'

        # Example: Stochastic Oscillator overbought/oversold
        if df['Stochastic_K'][-1] > 80 and df['Stochastic_D'][-1] > 80:
            print("Stochastic overbought.")
            return 'sell' if ml_signal != 'buy' else 'hold'
        elif df['Stochastic_K'][-1] < 20 and df['Stochastic_D'][-1] < 20:
            print("Stochastic oversold.")
            return 'buy' if ml_signal != 'sell' else 'hold'

        # ... (add other technical indicator analysis)

        print("No strong technical signal detected. Defaulting to ML signal or 'hold'.")
        return ml_signal  # Default to ML signal if no strong technical signal is found
