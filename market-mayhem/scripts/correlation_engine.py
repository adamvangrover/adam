import pandas as pd
import numpy as np

class CorrelationEngine:
    """
    Computes cross-asset correlations to detect regime shifts.
    """
    def __init__(self, lookback_window: int = 60):
        self.lookback_window = lookback_window

    def calculate_correlations(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates a rolling correlation matrix.
        Expects a DataFrame where columns are asset prices/yields and index is dates.
        """
        if len(data) < self.lookback_window:
            return data.corr()

        recent_data = data.tail(self.lookback_window)
        return recent_data.corr()

    def identify_regime(self, correlation_matrix: pd.DataFrame) -> str:
        """
        Infers the macro regime based on asset correlations.
        """
        try:
            # Check for stock/bond correlation
            if 'SPX' in correlation_matrix.columns and 'US10Y' in correlation_matrix.columns:
                stock_bond_corr = correlation_matrix.loc['SPX', 'US10Y']

                # If yields go up and stocks go down (negative corr between price of stocks and yield)
                # Note: Bond prices move inversely to yields. If stock prices and bond yields are positively
                # correlated, it means stocks and bonds are moving in opposite directions (risk on/off).
                # If stock prices and bond yields are negatively correlated, they are moving together (inflation fear).

                if stock_bond_corr < -0.3:
                    return "Inflation Shock (Rates drive equities)"
                elif stock_bond_corr > 0.3:
                    return "Growth Shock (Risk-on / Risk-off)"

            return "Mixed / Transition"
        except KeyError:
            return "Insufficient data for regime identification"

if __name__ == "__main__":
    # Example usage
    dates = pd.date_range("2023-01-01", periods=100)
    df = pd.DataFrame({
        'SPX': np.random.randn(100).cumsum(),
        'US10Y': np.random.randn(100).cumsum()
    }, index=dates)

    engine = CorrelationEngine()
    corr = engine.calculate_correlations(df)
    print("Correlation Matrix:")
    print(corr)
    print(f"Detected Regime: {engine.identify_regime(corr)}")
