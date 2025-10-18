# Hybrid Forecasting Model

## Overview

Adam v22.0 uses a hybrid forecasting model to improve predictive accuracy by combining traditional and modern forecasting techniques. The model combines a statistical model like ARIMA (to capture linear trends) with a deep learning model like an LSTM (to capture non-linear patterns).

## Model Architecture

The hybrid model consists of two components:

*   **ARIMA:** An Autoregressive Integrated Moving Average model that is used to capture linear trends in the data.
*   **LSTM:** A Long Short-Term Memory model that is used to capture non-linear patterns in the residuals of the ARIMA model.

The final forecast is a weighted average of the two models' outputs.

## Backtesting Results

Backtesting results have shown that the hybrid model outperforms standalone ARIMA and LSTM models in terms of accuracy.

## When to Use the Hybrid Model

The hybrid model is best suited for time-series data that exhibits both linear and non-linear patterns.
