import logging
import yfinance as yf
import pandas as pd
import numpy as np
from typing import Dict, Any, List

logger = logging.getLogger("RealtimeFetcher")

class RealtimeFetcher:
    """
    Fetches real-time financial data using yfinance and formats it
    to match the schema expected by the ICATEngine and CreditMemoOrchestrator.
    """
    def __init__(self):
        pass

    def fetch_data(self, ticker: str) -> Dict[str, Any]:
        """
        Main entry point to fetch all necessary data for a ticker.
        """
        logger.info(f"Fetching live data for {ticker}...")
        try:
            stock = yf.Ticker(ticker)
            info = stock.info

            # 1. Fetch Financials (Income Statement, Balance Sheet, Cash Flow)
            # yfinance returns DataFrames with columns as dates (descending usually)
            income_stmt = stock.financials
            balance_sheet = stock.balance_sheet
            cash_flow = stock.cashflow

            if income_stmt.empty or balance_sheet.empty:
                logger.warning(f"No financial data found for {ticker}")
                return None

            # 2. Extract Historical Data (Last 3-4 years)
            # Sort columns ascending (oldest to newest)
            income_stmt = income_stmt[sorted(income_stmt.columns)]

            # Align Balance Sheet and Cash Flow to Income Statement columns
            cols = income_stmt.columns
            balance_sheet = balance_sheet.reindex(columns=cols)
            cash_flow = cash_flow.reindex(columns=cols)

            years = [d.year for d in cols]

            # Helper to safely get data
            def get_row(df, keys):
                for k in keys:
                    if k in df.index:
                        return df.loc[k].fillna(0.0).tolist()
                return [0.0] * len(years)

            revenue = get_row(income_stmt, ['Total Revenue', 'Revenue'])
            net_income = get_row(income_stmt, ['Net Income', 'Net Income Common Stockholders'])
            ebitda = get_row(income_stmt, ['EBITDA', 'Normalized EBITDA'])
            interest_expense = get_row(income_stmt, ['Interest Expense', 'Interest Expense Non Operating'])

            total_assets = get_row(balance_sheet, ['Total Assets'])
            total_liabilities = get_row(balance_sheet, ['Total Liabilities Net Minority Interest', 'Total Liabilities'])
            total_debt = get_row(balance_sheet, ['Total Debt', 'Long Term Debt', 'Total Long Term Debt']) # Simplified
            cash = get_row(balance_sheet, ['Cash And Cash Equivalents', 'Cash Financial'])

            capex = get_row(cash_flow, ['Capital Expenditure', 'Capital Expenditures'])
            # Capex is usually negative in cash flow, make it positive for the model if needed,
            # but MOCK_LIBRARY has positive numbers. Let's abs() it.
            capex = [abs(x) for x in capex]

            historical = {
                "revenue": revenue,
                "ebitda": ebitda,
                "net_income": net_income,
                "total_assets": total_assets,
                "total_liabilities": total_liabilities,
                "total_debt": total_debt,
                "cash": cash,
                "interest_expense": interest_expense,
                "capex": capex,
                "year": years
            }

            # 3. Market Data
            history = stock.history(period="1y")
            price_data = history['Close'].tolist() if not history.empty else []
            # Downsample price data to last ~6 points for the mock UI graph if needed, or keep all
            # MOCK_LIBRARY has ~6 points. Let's give ~10-20 points for better granularity or just last 30 days?
            # The UI probably handles arrays. Let's give last 30 days.
            if len(price_data) > 30:
                price_data = price_data[-30:]

            market_data = {
                "share_price": info.get('currentPrice', 0.0),
                "market_cap": info.get('marketCap', 0.0),
                "beta": info.get('beta', 1.0),
                "pe_ratio": info.get('trailingPE', 0.0),
                "price_data": price_data
            }

            # 4. Construct Full Object
            return {
                "ticker": ticker,
                "name": info.get('longName', ticker),
                "sector": info.get('sector', 'Unknown'),
                "description": info.get('longBusinessSummary', 'No description available.')[:500] + "...",
                "historical": historical,
                "forecast_assumptions": {
                    "revenue_growth": [0.05] * 5, # Default 5%
                    "ebitda_margin": [0.20] * 5,  # Default 20%
                    "discount_rate": 0.10,
                    "terminal_growth_rate": 0.03
                },
                "market_data": market_data,
                "docs": {
                    "10-K": "Live data fetched. Refer to official SEC filings.",
                    "Credit_Agreement": "Not available in public feed."
                }
            }

        except Exception as e:
            logger.error(f"Error fetching data for {ticker}: {e}")
            return None
