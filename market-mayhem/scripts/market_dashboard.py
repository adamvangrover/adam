import pandas as pd

class MarketDashboard:
    """
    Generates a cross-asset dashboard for institutional reports.
    """
    def __init__(self):
        pass

    def generate_table(self, data: pd.DataFrame) -> str:
        """
        Converts raw market data into a markdown table.
        Expects data with columns: ['Asset', 'Level', '1D_Change', '1W_Change', 'Trend']
        """
        markdown = "| Asset Class | Current Level | 1D Change | 1W Change | Trend |\n"
        markdown += "|-------------|---------------|-----------|-----------|-------|\n"

        # Performance optimization: Use itertuples instead of to_dict('records')
        for row in data.itertuples(index=False):
            markdown += f"| {row.Asset:<11} | {row.Level:<13} | {row._3:<9} | {row._4:<9} | {row.Trend:<5} |\n"

        return markdown

if __name__ == "__main__":
    df = pd.DataFrame({
        'Asset': ['S&P 500', 'US 10Y', 'US 2Y', 'DXY', 'Crude Oil'],
        'Level': ['4500', '4.50%', '4.80%', '105.0', '$85.00'],
        '1D_Change': ['+0.5%', '+2 bps', '+1 bps', '-0.2%', '+1.0%'],
        '1W_Change': ['+1.2%', '+10 bps', '+5 bps', '+0.5%', '-2.0%'],
        'Trend': ['Up', 'Up', 'Up', 'Neutral', 'Down']
    })

    dashboard = MarketDashboard()
    print(dashboard.generate_table(df))
