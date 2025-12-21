# core/data_sources/market_data_api.py

import logging  # Added import

import requests

from core.utils.secrets_utils import get_api_key  # Added import


class MarketDataAPI:
    def __init__(self, config):
        # self.api_keys = config.get('api_keys', {}) # Removed API key loading from config
        # The config parameter might still be used for other settings if any.
        self.config = config

    def get_price_data(self, symbol, source="iex_cloud", period="daily", start_date=None, end_date=None):
        price_data = None # Initialize price_data to None

        if source == "iex_cloud":
            iex_key = get_api_key('IEX_CLOUD_API_KEY')
            if iex_key is None:
                logging.error("IEX Cloud API key not found. Cannot fetch price data for source 'iex_cloud'.")
                # price_data remains None, will be returned at the end
            else:
                if period == "daily":
                    # Fetch daily price data from IEX Cloud API
                    url = f"https://cloud.iexapis.com/stable/stock/{symbol}/chart/1m?token={iex_key}"
                    response = requests.get(url)
                    if response.status_code == 200:
                        data = response.json()
                        price_data = [{'date': item['date'], 'open': item['open'], 'high': item['high'], 'low': item['low'], 'close': item['close'], 'volume': item['volume']} for item in data]
                    else:
                        logging.error(f"Error fetching data from IEX Cloud: {response.status_code} - {response.text}")
                        # price_data remains None
                elif period == "intraday":
                    # Fetch intraday price data from IEX Cloud API
                    url = f"https://cloud.iexapis.com/stable/stock/{symbol}/intraday-prices?token={iex_key}"
                    response = requests.get(url)
                    if response.status_code == 200:
                        data = response.json()
                        price_data = [{'date': item['date'], 'minute': item['minute'], 'open': item['open'], 'high': item['high'], 'low': item['low'], 'close': item['close'], 'volume': item['volume']} for item in data]
                    else:
                        logging.error(f"Error fetching data from IEX Cloud: {response.status_code} - {response.text}")
                        # price_data remains None

        elif source == "alpha_vantage":
            alpha_vantage_key = get_api_key('ALPHA_VANTAGE_API_KEY')
            if alpha_vantage_key is None:
                logging.error("Alpha Vantage API key not found. Cannot fetch price data for source 'alpha_vantage'.")
                # price_data remains None, will be returned at the end
            else:
                if period == "daily":
                    # Fetch daily price data from Alpha Vantage API
                    url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={symbol}&apikey={alpha_vantage_key}"
                    response = requests.get(url)
                    if response.status_code == 200:
                        # It's good practice to check if 'Time Series (Daily)' key exists
                        data = response.json().get('Time Series (Daily)')
                        if data:
                            price_data = [{'date': date, 'open': float(item['1. open']), 'high': float(item['2. high']), 'low': float(item['3. low']), 'close': float(item['4. close']), 'volume': int(item['5. volume'])} for date, item in data.items()]
                        else:
                            logging.error(f"Error fetching data from Alpha Vantage: 'Time Series (Daily)' not found in response. Response: {response.text}")
                            # price_data remains None
                    else:
                        logging.error(f"Error fetching data from Alpha Vantage: {response.status_code} - {response.text}")
                        # price_data remains None
                elif period == "intraday":
                    # Fetch intraday price data from Alpha Vantage API
                    url = f"https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol={symbol}&interval=5min&apikey={alpha_vantage_key}"
                    response = requests.get(url)
                    if response.status_code == 200:
                        # Similar check for 'Time Series (5min)'
                        data = response.json().get('Time Series (5min)')
                        if data:
                            # The original code had a potential issue here: item['date'], item['minute'] not in Alpha Vantage intraday response
                            # Assuming the structure is similar to daily for keys like 'open', 'high', 'low', 'close', 'volume'
                            # This part might need adjustment based on actual Alpha Vantage intraday response structure.
                            # For now, I'll keep it similar to the original logic if data is found.
                            price_data = [{'date': date_time.split()[0], 'minute': date_time.split()[1][:5] if len(date_time.split()) > 1 else None, 'open': float(item['1. open']), 'high': float(item['2. high']), 'low': float(item['3. low']), 'close': float(item['4. close']), 'volume': int(item['5. volume'])} for date_time, item in data.items()]
                        else:
                            logging.error(f"Error fetching data from Alpha Vantage: 'Time Series (5min)' not found in response. Response: {response.text}")
                            # price_data remains None
                    else:
                        logging.error(f"Error fetching data from Alpha Vantage: {response.status_code} - {response.text}")
                        # price_data remains None
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
