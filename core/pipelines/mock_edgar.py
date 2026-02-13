"""
Mock EDGAR Financial Data Source
--------------------------------
Provides mocked annual financial data for major technology companies (AAPL, MSFT, GOOGL, AMZN, NVDA, TSLA, META).
Data is representative of publicly available 2021-2023 figures (approximately).
This module acts as a stand-in for a live SEC EDGAR API connection.
"""

from typing import Dict, Any, List

class EdgarSource:
    """
    Mock Data Source for Public Company Financials.
    """

    # Dictionary of Ticker -> Financial Data History (in millions USD)
    FINANCIALS = {
        "AAPL": {
            "company_name": "Apple Inc.",
            "history": [
                {
                    "fiscal_year": 2021,
                    "revenue": 365817,
                    "ebitda": 120233,
                    "total_debt": 124719,
                    "cash_equivalents": 34940,
                    "interest_expense": 2645,
                    "total_assets": 351002,
                    "total_liabilities": 287912,
                    "total_equity": 63090
                },
                {
                    "fiscal_year": 2022,
                    "revenue": 394328,
                    "ebitda": 130541,
                    "total_debt": 120069,
                    "cash_equivalents": 23646,
                    "interest_expense": 2931,
                    "total_assets": 352755,
                    "total_liabilities": 302083,
                    "total_equity": 50672
                },
                {
                    "fiscal_year": 2023,
                    "revenue": 383285,
                    "ebitda": 114301,
                    "total_debt": 111088,
                    "cash_equivalents": 29965,
                    "interest_expense": 3933,
                    "total_assets": 352583,
                    "total_liabilities": 290437,
                    "total_equity": 62146
                }
            ]
        },
        "MSFT": {
            "company_name": "Microsoft Corporation",
            "history": [
                {
                    "fiscal_year": 2021,
                    "revenue": 168088,
                    "ebitda": 80816,
                    "total_debt": 58120,
                    "cash_equivalents": 14221,
                    "interest_expense": 2346,
                    "total_assets": 333779,
                    "total_liabilities": 191791,
                    "total_equity": 141988
                },
                {
                    "fiscal_year": 2022,
                    "revenue": 198270,
                    "ebitda": 97843,
                    "total_debt": 49751,
                    "cash_equivalents": 13931,
                    "interest_expense": 2063,
                    "total_assets": 364840,
                    "total_liabilities": 198298,
                    "total_equity": 166542
                },
                {
                    "fiscal_year": 2023,
                    "revenue": 211915,
                    "ebitda": 102384,
                    "total_debt": 47204,
                    "cash_equivalents": 34704,
                    "interest_expense": 1968,
                    "total_assets": 411976,
                    "total_liabilities": 205753,
                    "total_equity": 206223
                }
            ]
        },
        "GOOGL": {
            "company_name": "Alphabet Inc.",
            "history": [
                {
                    "fiscal_year": 2021,
                    "revenue": 257637,
                    "ebitda": 91155,
                    "total_debt": 14817,
                    "cash_equivalents": 20945,
                    "interest_expense": 346,
                    "total_assets": 359268,
                    "total_liabilities": 107633,
                    "total_equity": 251635
                },
                {
                    "fiscal_year": 2022,
                    "revenue": 282836,
                    "ebitda": 74842,
                    "total_debt": 14701,
                    "cash_equivalents": 21879,
                    "interest_expense": 357,
                    "total_assets": 365264,
                    "total_liabilities": 109120,
                    "total_equity": 256144
                },
                {
                    "fiscal_year": 2023,
                    "revenue": 307394,
                    "ebitda": 88164,
                    "total_debt": 13253,
                    "cash_equivalents": 24048,
                    "interest_expense": 321,
                    "total_assets": 402392,
                    "total_liabilities": 119048,
                    "total_equity": 283344
                }
            ]
        },
        "AMZN": {
            "company_name": "Amazon.com, Inc.",
            "history": [
                {
                    "fiscal_year": 2021,
                    "revenue": 469822,
                    "ebitda": 59175,
                    "total_debt": 48744,
                    "cash_equivalents": 36220,
                    "interest_expense": 1809,
                    "total_assets": 420549,
                    "total_liabilities": 282304,
                    "total_equity": 138245
                },
                {
                    "fiscal_year": 2022,
                    "revenue": 513983,
                    "ebitda": 54169,
                    "total_debt": 67150,
                    "cash_equivalents": 53888,
                    "interest_expense": 2367,
                    "total_assets": 462675,
                    "total_liabilities": 316632,
                    "total_equity": 146043
                },
                {
                    "fiscal_year": 2023,
                    "revenue": 574785,
                    "ebitda": 85515,
                    "total_debt": 58316,
                    "cash_equivalents": 73890,
                    "interest_expense": 3178,
                    "total_assets": 527854,
                    "total_liabilities": 326084,
                    "total_equity": 201770
                }
            ]
        },
        "NVDA": {
            "company_name": "NVIDIA Corporation",
            "history": [
                {
                    "fiscal_year": 2021,
                    "revenue": 16675,
                    "ebitda": 4532,
                    "total_debt": 6965,
                    "cash_equivalents": 11561,
                    "interest_expense": 184,
                    "total_assets": 28791,
                    "total_liabilities": 11898,
                    "total_equity": 16893
                },
                {
                    "fiscal_year": 2022,
                    "revenue": 26914,
                    "ebitda": 11216,
                    "total_debt": 10946,
                    "cash_equivalents": 1991,
                    "interest_expense": 236,
                    "total_assets": 44187,
                    "total_liabilities": 17575,
                    "total_equity": 26612
                },
                {
                    "fiscal_year": 2023,
                    "revenue": 26974,
                    "ebitda": 5600,
                    "total_debt": 11130,
                    "cash_equivalents": 3389,
                    "interest_expense": 272,
                    "total_assets": 41182,
                    "total_liabilities": 19081,
                    "total_equity": 22101
                }
            ]
        },
        "TSLA": {
            "company_name": "Tesla, Inc.",
            "history": [
                {
                    "fiscal_year": 2021,
                    "revenue": 53823,
                    "ebitda": 9600,
                    "total_debt": 6834,
                    "cash_equivalents": 17576,
                    "interest_expense": 371,
                    "total_assets": 62131,
                    "total_liabilities": 30548,
                    "total_equity": 30189
                },
                {
                    "fiscal_year": 2022,
                    "revenue": 81462,
                    "ebitda": 17660,
                    "total_debt": 3099,
                    "cash_equivalents": 22185,
                    "interest_expense": 191,
                    "total_assets": 82338,
                    "total_liabilities": 36440,
                    "total_equity": 44704
                },
                {
                    "fiscal_year": 2023,
                    "revenue": 96773,
                    "ebitda": 14997,
                    "total_debt": 4350,
                    "cash_equivalents": 29072,
                    "interest_expense": 156,
                    "total_assets": 106618,
                    "total_liabilities": 43009,
                    "total_equity": 62634
                }
            ]
        },
        "META": {
            "company_name": "Meta Platforms, Inc.",
            "history": [
                {
                    "fiscal_year": 2021,
                    "revenue": 117929,
                    "ebitda": 54720,
                    "total_debt": 13876,
                    "cash_equivalents": 16601,
                    "interest_expense": 0,
                    "total_assets": 165987,
                    "total_liabilities": 41108,
                    "total_equity": 124879
                },
                {
                    "fiscal_year": 2022,
                    "revenue": 116609,
                    "ebitda": 40380,
                    "total_debt": 26402,
                    "cash_equivalents": 14681,
                    "interest_expense": 109,
                    "total_assets": 185727,
                    "total_liabilities": 60014,
                    "total_equity": 125713
                },
                {
                    "fiscal_year": 2023,
                    "revenue": 134902,
                    "ebitda": 62310,
                    "total_debt": 37043,
                    "cash_equivalents": 41862,
                    "interest_expense": 371,
                    "total_assets": 229623,
                    "total_liabilities": 76016,
                    "total_equity": 153607
                }
            ]
        }
    }

    def get_financial_history(self, ticker: str) -> Dict[str, Any]:
        """
        Returns the historical financials for the given ticker.
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
