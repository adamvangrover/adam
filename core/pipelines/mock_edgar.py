"""
Mock EDGAR Financial Data Source
--------------------------------
Provides mocked annual financial data for major technology companies (AAPL, MSFT, GOOGL, AMZN, NVDA, TSLA, META).
Data is representative of publicly available 2023-2024 figures (approximately).
This module acts as a stand-in for a live SEC EDGAR API connection.
"""

from typing import Dict, Any

class EdgarSource:
    """
    Mock Data Source for Public Company Financials.
    """

    # Dictionary of Ticker -> Financial Data (in millions USD)
    FINANCIALS = {
        "AAPL": {
            "company_name": "Apple Inc.",
            "fiscal_year": 2023,
            "revenue": 383285,
            "ebitda": 114301,
            "total_debt": 111088,
            "cash_equivalents": 29965,
            "interest_expense": 3933,
            "total_assets": 352583,
            "total_liabilities": 290437,
            "total_equity": 62146
        },
        "MSFT": {
            "company_name": "Microsoft Corporation",
            "fiscal_year": 2023,
            "revenue": 211915,
            "ebitda": 102384,
            "total_debt": 47204,
            "cash_equivalents": 34704,
            "interest_expense": 1968,
            "total_assets": 411976,
            "total_liabilities": 205753,
            "total_equity": 206223
        },
        "GOOGL": {
            "company_name": "Alphabet Inc.",
            "fiscal_year": 2023,
            "revenue": 307394,
            "ebitda": 88164,
            "total_debt": 13253,
            "cash_equivalents": 24048,
            "interest_expense": 321,
            "total_assets": 402392,
            "total_liabilities": 119048,
            "total_equity": 283344
        },
        "AMZN": {
            "company_name": "Amazon.com, Inc.",
            "fiscal_year": 2023,
            "revenue": 574785,
            "ebitda": 85515,
            "total_debt": 58316,
            "cash_equivalents": 73890, # Including marketable securities
            "interest_expense": 3178,
            "total_assets": 527854,
            "total_liabilities": 326084,
            "total_equity": 201770
        },
        "NVDA": {
            "company_name": "NVIDIA Corporation",
            "fiscal_year": 2024, # Fiscal year ends Jan
            "revenue": 60922,
            "ebitda": 34480,
            "total_debt": 8461,
            "cash_equivalents": 25984,
            "interest_expense": 257,
            "total_assets": 65728,
            "total_liabilities": 22750,
            "total_equity": 42978
        }
    }

    def get_annual_financials(self, ticker: str) -> Dict[str, Any]:
        """
        Returns the annual financials for the given ticker.
        """
        data = self.FINANCIALS.get(ticker.upper())
        if not data:
            raise ValueError(f"Ticker {ticker} not found in Mock EDGAR Source.")
        return data

    def list_tickers(self):
        """
        Returns the list of available tickers.
        """
        return list(self.FINANCIALS.keys())
