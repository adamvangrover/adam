import numpy as np
import talib


def sma_crossover_strategy(data, short_window=40, long_window=100):
    """
    Generates trading signals based on a simple moving average (SMA) crossover strategy.

    Args:
        data (pd.DataFrame): A DataFrame with a 'close' column containing closing prices.
        short_window (int): The lookback period for the short-term SMA.
        long_window (int): The lookback period for the long-term SMA.

    Returns:
        pd.DataFrame: A DataFrame with 'signal' and 'position' columns.
                      'signal' is 1 for a buy signal, -1 for a sell signal, and 0 otherwise.
                      'position' is 1 for a long position and -1 for a short position.
    """
    signals = data.copy()
    signals['short_mavg'] = talib.SMA(data['close'], timeperiod=short_window)
    signals['long_mavg'] = talib.SMA(data['close'], timeperiod=long_window)

    # Generate trading signals
    signals['signal'] = 0.0
    signals['signal'][short_window:] = np.where(signals['short_mavg'][short_window:] > signals['long_mavg'][short_window:], 1.0, 0.0)

    # Generate trading positions
    signals['position'] = signals['signal'].diff()
    signals['position'] = signals['position'].cumsum()

    return signals
