import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
import os
import shutil
from core.market_data.historical_loader import HistoricalLoader
from core.schemas.market_data_schema import MarketDataSchema


class TestHistoricalLoader(unittest.TestCase):
    def setUp(self):
        self.test_dir = "tests/temp_market_data"
        self.loader = HistoricalLoader(data_dir=self.test_dir)
        os.makedirs(self.test_dir, exist_ok=True)

    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    @patch('core.market_data.historical_loader.yf.download')
    def test_fetch_history_single(self, mock_download):
        # Mock single ticker response
        data = {
            'Open': [100.0, 101.0],
            'High': [102.0, 103.0],
            'Low': [99.0, 100.0],
            'Close': [101.0, 102.0],
            'Volume': [1000, 2000],
            'Adj Close': [101.0, 102.0]
        }
        df_mock = pd.DataFrame(data, index=pd.to_datetime(['2023-01-01', '2023-01-02']))
        mock_download.return_value = df_mock

        df = self.loader.fetch_history(['AAPL'])

        self.assertFalse(df.empty)
        self.assertIn('Ticker', df.columns)
        self.assertEqual(df['Ticker'].iloc[0], 'AAPL')
        self.assertEqual(len(df), 2)

    @patch('core.market_data.historical_loader.yf.download')
    def test_fetch_history_multiple(self, mock_download):
        # Mock multiple ticker response (MultiIndex columns)
        # Columns: (Price, Ticker)
        index = pd.to_datetime(['2023-01-01', '2023-01-02'])
        columns = pd.MultiIndex.from_product([['AAPL', 'MSFT'], ['Open', 'Close']], names=['Ticker', 'Price'])
        data = [[100, 101, 200, 201], [102, 103, 202, 203]]
        df_mock = pd.DataFrame(data, index=index, columns=columns)

        # yfinance group_by='ticker' returns (Ticker, Price)
        # Let's adjust mock to match group_by='ticker'
        # Top level: Ticker
        columns = pd.MultiIndex.from_product([['AAPL', 'MSFT'], ['Open', 'Close']], names=['Ticker', 'Price'])
        df_mock = pd.DataFrame(data, index=index, columns=columns)

        mock_download.return_value = df_mock

        df = self.loader.fetch_history(['AAPL', 'MSFT'])

        self.assertFalse(df.empty)
        self.assertIn('Ticker', df.columns)
        self.assertIn('AAPL', df['Ticker'].unique())
        self.assertIn('MSFT', df['Ticker'].unique())
        self.assertEqual(len(df), 4)  # 2 dates * 2 tickers

    def test_validate_data(self):
        # Create valid dataframe
        data = {
            'Date': pd.to_datetime(['2023-01-01']),
            'Ticker': ['AAPL'],
            'Open': [100.0],
            'High': [105.0],
            'Low': [95.0],
            'Close': [102.0],
            'Volume': [1000000.0]
        }
        df = pd.DataFrame(data)
        validated_df = self.loader.validate_data(df)
        self.assertFalse(validated_df.empty)

    def test_save_to_parquet(self):
        data = {
            'Date': pd.to_datetime(['2023-01-01']),
            'Ticker': ['AAPL'],
            'Open': [100.0],
            'High': [105.0],
            'Low': [95.0],
            'Close': [102.0],
            'Volume': [1000000.0]
        }
        df = pd.DataFrame(data)
        self.loader.save_to_parquet(df, filename="test.parquet")

        self.assertTrue(os.path.exists(os.path.join(self.test_dir, "test.parquet")))


if __name__ == '__main__':
    unittest.main()
