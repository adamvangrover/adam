import yfinance as yf
from typing import Dict, Any, List
import pandas as pd
import requests
from bs4 import BeautifulSoup

class FinancialDataTool:
    """
    A tool for retrieving financial data using yfinance and other sources.
    """
    def __init__(self):
        pass

    def get_stock_data(self, ticker: str, period: str = "1mo") -> pd.DataFrame:
        """
        Retrieves historical stock data for a given ticker.
        """
        stock = yf.Ticker(ticker)
        history = stock.history(period=period)
        return history

    def get_company_info(self, ticker: str) -> Dict[str, Any]:
        """
        Retrieves company information.
        """
        stock = yf.Ticker(ticker)
        return stock.info

    def get_financials(self, ticker: str) -> pd.DataFrame:
        """
        Retrieves financial statements (income statement, balance sheet, cash flow).
        """
        stock = yf.Ticker(ticker)
        # Returns a tuple of DataFrames: (income_stmt, balance_sheet, cashflow)
        # For simplicity, let's return the income statement primarily, or a dict of all 3
        # yfinance returns these as properties
        return stock.financials

    def search_news(self, query: str) -> List[Dict[str, Any]]:
        """
        Searches for financial news related to a query.
        (Placeholder implementation using a mock or simple scraper)
        """
        # In a real implementation, you might use Google News API or similar
        # Here is a simple mock for demonstration
        results = [
            {"title": f"Market update for {query}", "link": "http://example.com/news1", "summary": "Positive outlook..."},
            {"title": f"{query} earnings report", "link": "http://example.com/news2", "summary": "Revenue exceeded expectations..."}
        ]
        return results

    def get_treasury_yields(self) -> Dict[str, float]:
        """
        Retrieves current US Treasury yields (mock implementation).
        """
        # In reality, scrape from Treasury.gov or use an API
        return {
            "3M": 5.45,
            "2Y": 4.88,
            "10Y": 4.50,
            "30Y": 4.65
        }
