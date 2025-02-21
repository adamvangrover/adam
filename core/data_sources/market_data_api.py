# core/data_sources/market_data_api.py

import requests
import pandas as pd

class MarketDataAPI:
    def __init__(self, config):
        self.api_keys = config.get('api_keys', {})

    def get_price_data(self, symbol, source="iex_cloud", period="daily", start_date=None, end_date=None):
        if source == "iex_cloud":
            if period == "daily":
                # Fetch daily price data from IEX Cloud API
                url = f"https://cloud.iexapis.com/stable/stock/{symbol}/chart/1m?token={self.api_keys.get('iex_cloud')}"
                response = requests.get(url)
                if response.status_code == 200:
                    data = response.json()
                    price_data = [{'date': item['date'], 'open': item['open'], 'high': item['high'], 'low': item['low'], 'close': item['close'], 'volume': item['volume']} for item in data]
                else:
                    print(f"Error fetching data from IEX Cloud: {response.status_code} - {response.text}")
                    price_data = None
            elif period == "intraday":
                # Fetch intraday price data from IEX Cloud API
                url = f"https://cloud.iexapis.com/stable/stock/{symbol}/intraday-prices?token={self.api_keys.get('iex_cloud')}"
                response = requests.get(url)
                if response.status_code == 200:
                    data = response.json()
                    price_data = [{'date': item['date'], 'minute': item['minute'], 'open': item['open'], 'high': item['high'], 'low': item['low'], 'close': item['close'], 'volume': item['volume']} for item in data]
                else:
                    print(f"Error fetching data from IEX Cloud: {response.status_code} - {response.text}")
                    price_data = None
        elif source == "alpha_vantage":
            if period == "daily":
                # Fetch daily price data from Alpha Vantage API
                url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={symbol}&apikey={self.api_keys.get('alpha_vantage')}"
                response = requests.get(url)
                if response.status_code == 200:
                    data = response.json()['Time Series (Daily)']
                    price_data = [{'date': date, 'open': float(item['1. open']), 'high': float(item['2. high']), 'low': float(item['3. low']), 'close': float(item['4. close']), 'volume': int(item['5. volume'])} for date, item in data.items()]
                else:
                    print(f"Error fetching data from Alpha Vantage: {response.status_code} - {response.text}")
                    price_data = None
            elif period == "intraday":
                # Fetch intraday price data from Alpha Vantage API
                url = f"https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol={symbol}&interval=5min&apikey={self.api_keys.get('alpha_vantage')}"
                response = requests.get(url)
                if response.status_code == 200:
                    data = response.json()['Time Series (5min)']
                    price_data = [{'date': item['date'], 'minute': item['minute'], 'open': item['open'], 'high': item['high'], 'low': item['low'], 'close': item['close'], 'volume': item['volume']} for item in data]
                else:
                    print(f"Error fetching data from Alpha Vantage: {response.status_code} - {response.text}")
                    price_data = None
        #... (add handling for other data sources and periods)
        return price_data

    def get_historical_data(self, symbol, source="iex_cloud", start_date=None, end_date=None):
        #... (fetch historical data from the specified source)
        #... (handle date range filtering)
        pass  # Placeholder for actual implementation

    def get_quote(self, symbol, source="iex_cloud"):
        #... (fetch real-time quote for the specified symbol)
        pass  # Placeholder for actual implementation

    #... (add methods for other market data as needed, e.g., options data, fundamentals)
