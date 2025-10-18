# core/analysis/forecasting/hybrid_model.py

import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

class HybridModel:
    """
    A hybrid forecasting model that combines ARIMA and LSTM.
    """

    def __init__(self, arima_order=(5, 1, 0), lstm_units=50):
        """
        Initializes the HybridModel.

        Args:
            arima_order: The order of the ARIMA model.
            lstm_units: The number of units in the LSTM layer.
        """
        self.arima_order = arima_order
        self.lstm_units = lstm_units
        self.arima_model = None
        self.lstm_model = None

    def fit(self, data: pd.Series):
        """
        Fits the model to the data.

        Args:
            data: The time-series data.
        """
        # Fit the ARIMA model
        self.arima_model = ARIMA(data, order=self.arima_order).fit()

        # Fit the LSTM model on the ARIMA residuals
        residuals = self.arima_model.resid
        self.lstm_model = self._build_lstm_model()
        self.lstm_model.fit(residuals, epochs=100, verbose=0)

    def _build_lstm_model(self):
        """
        Builds the LSTM model.
        """
        model = Sequential()
        model.add(LSTM(self.lstm_units, input_shape=(1, 1)))
        model.add(Dense(1))
        model.compile(loss='mean_squared_error', optimizer='adam')
        return model

    def predict(self, n_periods: int) -> pd.Series:
        """
        Makes a forecast.

        Args:
            n_periods: The number of periods to forecast.

        Returns:
            The forecast.
        """
        # Make a forecast with the ARIMA model
        arima_forecast = self.arima_model.forecast(steps=n_periods)

        # Make a forecast with the LSTM model on the residuals
        lstm_forecast = self.lstm_model.predict(self.arima_model.resid[-n_periods:])

        # Combine the forecasts
        return arima_forecast + lstm_forecast
