# Data Sources

This directory contains modules for accessing various data sources, such as APIs and databases. Each module provides a standardized interface for retrieving data, regardless of the underlying source.

## Base Class

All data source modules should inherit from the `BaseDataSource` class in `core/data_access/base_data_source.py`. This class defines the common interface for all data sources, including:

*   **`__init__(self, config)`:** Initializes the data source with its configuration.
*   **`get_data(self, params)`:** Retrieves data from the source based on the given parameters.

## Usage Examples

Here are some examples of how to use the available data sources:

### `financial_news_api.py`

To use the financial news API, you first need to create an instance of the `FinancialNewsAPI` class with the appropriate configuration. Then, you can use the `get_data` method to retrieve news articles for a specific company.

```python
from core.data_sources.financial_news_api import FinancialNewsAPI

# Create a new instance of the FinancialNewsAPI class
config = {"api_key": "YOUR_API_KEY"}
news_api = FinancialNewsAPI(config)

# Retrieve news articles for Apple
params = {"query": "Apple"}
news_articles = news_api.get_data(params)

# Print the headlines of the news articles
for article in news_articles:
    print(article["headline"])
```

### `market_data_api.py`

To use the market data API, you first need to create an instance of the `MarketDataAPI` class with the appropriate configuration. Then, you can use the `get_data` method to retrieve market data for a specific stock.

```python
from core.data_sources.market_data_api import MarketDataAPI

# Create a new instance of the MarketDataAPI class
config = {"api_key": "YOUR_API_KEY"}
market_data_api = MarketDataAPI(config)

# Retrieve the latest price for Apple stock
params = {"ticker": "AAPL"}
market_data = market_data_api.get_data(params)

# Print the latest price
print(market_data["price"])
```

## Available Data Sources

*   **`financial_news_api.py`:** Accesses financial news from a third-party API.
*   **`government_stats_api.py`:** Retrieves economic statistics from a government API.
*   **`market_data_api.py`:** Fetches real-time and historical market data.
*   **`social_media_api.py`:** Gathers data from social media platforms.

## Adding a New Data Source

To add a new data source, follow these steps:

1.  **Create a new Python file** in this directory. The file name should be descriptive of the data source (e.g., `my_new_data_source.py`).
2.  **Import the `BaseDataSource` class** from `core/data_access/base_data_source.py`.
3.  **Create a new class** that inherits from the `BaseDataSource` class.
4.  **Implement the `__init__` method** to initialize the data source with its configuration. This should include any API keys or other credentials required to access the data source.
5.  **Implement the `get_data` method** to retrieve data from the source. This method should handle any authentication, request formatting, and data parsing required to access the data.
6.  **Add the new data source to the `config/data_sources.yaml` file.** This will make the data source available to the rest of the system.

## Configuration

The configuration for each data source is stored in the `config/data_sources.yaml` file. This file contains the necessary information to connect to and authenticate with each data source, such as API keys, URLs, and other parameters.

By following these guidelines, you can help to ensure that the data sources in the ADAM system are reliable, easy to use, and well-maintained.
