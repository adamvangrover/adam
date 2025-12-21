# core/analysis/forecasting/hybrid_model.py

import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
import torch
import torch.nn as nn
import torch.optim as optim


class LSTMResidualModel(nn.Module):
    def __init__(self, units):
        super(LSTMResidualModel, self).__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=units, batch_first=True)
        self.fc = nn.Linear(units, 1)

    def forward(self, x):
        # x: (batch, seq, feature)
        out, _ = self.lstm(x)
        out = out[:, -1, :]  # Last time step
        out = self.fc(out)
        return out


class HybridModel:
    """
    A hybrid forecasting model that combines ARIMA and LSTM using PyTorch.
    """

    def __init__(self, arima_order=(5, 1, 0), lstm_units=50):
        self.arima_order = arima_order
        self.lstm_units = lstm_units
        self.arima_model = None
        self.lstm_model = None

    def fit(self, data: pd.Series):
        """
        Fits the model to the data.
        """
        # Fit the ARIMA model
        self.arima_model = ARIMA(data, order=self.arima_order).fit()
        residuals = self.arima_model.resid

        # Prepare data for LSTM (reconstruct residuals)
        # Reshape to (N, 1, 1)
        values = residuals.values.reshape(-1, 1, 1)

        input_tensor = torch.tensor(values, dtype=torch.float32)
        target_tensor = torch.tensor(values.reshape(-1, 1), dtype=torch.float32)

        self.lstm_model = LSTMResidualModel(self.lstm_units)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.lstm_model.parameters())

        self.lstm_model.train()
        # Simple training loop
        for _ in range(100):
            optimizer.zero_grad()
            out = self.lstm_model(input_tensor)
            loss = criterion(out, target_tensor)
            loss.backward()
            optimizer.step()

    def predict(self, n_periods: int) -> pd.Series:
        """
        Makes a forecast.
        """
        if self.arima_model is None or self.lstm_model is None:
            raise ValueError("Model must be fitted before prediction.")

        # Make a forecast with the ARIMA model
        arima_forecast = self.arima_model.forecast(steps=n_periods)

        # Make a forecast with the LSTM model on the residuals
        # Use last n_periods residuals as input to predict 'correction'
        last_resid = self.arima_model.resid[-n_periods:]
        if len(last_resid) < n_periods:
            # Handle case where we don't have enough history
            # Pad or use what we have.
            # For simplicity, just use last available
            pass

        # Reshape for LSTM
        last_resid_values = last_resid.values.reshape(-1, 1, 1)
        input_tensor = torch.tensor(last_resid_values, dtype=torch.float32)

        self.lstm_model.eval()
        with torch.no_grad():
            lstm_pred = self.lstm_model(input_tensor).numpy().flatten()

        lstm_forecast_series = pd.Series(lstm_pred, index=arima_forecast.index)

        # Combine the forecasts
        return arima_forecast + lstm_forecast_series
