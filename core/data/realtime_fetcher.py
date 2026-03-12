import logging
import yfinance as yf
import pandas as pd
from typing import Dict, Any, List, Optional
import datetime

logger = logging.getLogger("RealtimeFetcher")

class RealtimeFetcher:
    """
    Fetches real-time market data using yfinance.
    Provides historical data, current price, and forecast assumptions.
    """

    def __init__(self):
        pass

    def fetch_market_data(self, ticker: str) -> Dict[str, Any]:
        """
        Fetches current market data (price, beta, PE, etc.).
        """
        try:
            t = yf.Ticker(ticker)
            info = t.info

            # yfinance info keys can vary, using safe gets
            current_price = info.get("currentPrice") or info.get("regularMarketPrice") or info.get("previousClose")

            # Basic fallback for price if absolutely nothing found (should rarely happen for valid tickers)
            if not current_price:
                hist = t.history(period="1d")
                if not hist.empty:
                    current_price = hist["Close"].iloc[-1]
                else:
                    logger.warning(f"Could not fetch price for {ticker}")
                    current_price = 0.0

            market_data = {
                "share_price": current_price,
                "market_cap": info.get("marketCap", 0),
                "beta": info.get("beta", 1.0), # Default to market beta if missing
                "pe_ratio": info.get("trailingPE", 0.0),
                "price_data": [], # Populated via history fetch if needed
                "consensus_target_price": info.get("targetMeanPrice", 0.0),
                "consensus_rating": info.get("recommendationKey", "N/A").upper(),
                "analyst_count": info.get("numberOfAnalystOpinions", 0)
            }

            # Additional consensus insights (e.g. forward EPS)
            forward_eps = info.get("forwardEps", 0.0)
            if forward_eps:
                market_data["forward_eps"] = forward_eps

            return market_data
        except Exception as e:
            logger.error(f"Error fetching market data for {ticker}: {e}")
            return {}

    def fetch_historical_data(self, ticker: str, period: str = "3y") -> Dict[str, List[Any]]:
        """
        Fetches historical financials and price history.
        Note: yfinance financials are annual/quarterly. We align to annual for this pipeline.
        """
        try:
            t = yf.Ticker(ticker)

            # 1. Price History (for volatility/charting)
            # Fetching 1y for charts, but maybe 3y for financials?
            # Let's get 5 years of financials if available
            financials = t.financials
            balance_sheet = t.balance_sheet
            cash_flow = t.cashflow

            if financials.empty:
                logger.warning(f"No financials found for {ticker}")
                return {}

            # Transpose to have dates as index (yfinance default is dates as columns)
            fin_T = financials.T
            bs_T = balance_sheet.T
            cf_T = cash_flow.T

            # Sort ascending by date
            fin_T = fin_T.sort_index()
            bs_T = bs_T.sort_index()
            cf_T = cf_T.sort_index()

            # Extract key metrics
            # yfinance field names can change, mapping common ones
            years = []
            revenue = []
            ebitda = []
            net_income = []
            total_assets = []
            total_liabilities = []
            total_debt = []
            cash = []
            interest_expense = []
            capex = []

            # Loop through available years (up to last 3-5)
            # Alignment is tricky if statements have different dates.
            # We will drive off the Income Statement dates.

            for date in fin_T.index:
                year_str = date.year
                years.append(year_str)

                # Safe extractor
                def safe_val(df, key, default=0.0):
                    val = df.get(key)
                    if val is None: return default
                    if isinstance(val, pd.Series): val = val.iloc[0]
                    if pd.isna(val): return default
                    return float(val)

                # Revenue
                rev = safe_val(fin_T.loc[date], "Total Revenue") or safe_val(fin_T.loc[date], "Revenue")
                revenue.append(rev / 1_000_000)

                # EBITDA (Approximation if missing: EBIT + D&A)
                ebitda_val = safe_val(fin_T.loc[date], "EBITDA") or safe_val(fin_T.loc[date], "Normalized EBITDA")
                if ebitda_val == 0.0:
                     ebit = safe_val(fin_T.loc[date], "EBIT") or safe_val(fin_T.loc[date], "Net Income")
                     da_val = safe_val(cf_T.loc[date], "Depreciation And Amortization") if date in cf_T.index else 0
                     ebitda_val = ebit + da_val
                ebitda.append(ebitda_val / 1_000_000)

                # Net Income
                ni = safe_val(fin_T.loc[date], "Net Income") or safe_val(fin_T.loc[date], "Net Income Common Stockholders")
                net_income.append(ni / 1_000_000)

                # Balance Sheet Items
                bs_row = bs_T.loc[date] if date in bs_T.index else pd.Series()

                ta = safe_val(bs_row, "Total Assets")
                total_assets.append(ta / 1_000_000)

                tl = safe_val(bs_row, "Total Liabilities Net Minority Interest") or safe_val(bs_row, "Total Liabilities")
                total_liabilities.append(tl / 1_000_000)

                # Total Debt might be sum of parts
                td = safe_val(bs_row, "Total Debt")
                if td == 0.0:
                    td = safe_val(bs_row, "Long Term Debt") + safe_val(bs_row, "Current Debt")
                total_debt.append(td / 1_000_000)

                c = safe_val(bs_row, "Cash And Cash Equivalents")
                cash.append(c / 1_000_000)

                # Cash Flow Items
                cf_row = cf_T.loc[date] if date in cf_T.index else pd.Series()

                # Interest Expense (Income Statement)
                ie = safe_val(fin_T.loc[date], "Interest Expense")
                interest_expense.append(abs(ie) / 1_000_000)

                cx = safe_val(cf_row, "Capital Expenditure")
                capex.append(abs(cx) / 1_000_000)

            # Price History (Last 6 months for vol calc)
            # Not part of the financial statements alignment, so can be different length
            # ICAT Engine handles this separation if data structure is correct.
            # However, if 'price_data' is mixed into the 'historical' dictionary passed to DataFrame constructor,
            # pandas will complain if lengths mismatch.
            # SOLUTION: Separate 'price_data' or pad/trim?
            # Actually, CreditMemoOrchestrator passes data['historical'] to ICAT.clean(),
            # which creates a DataFrame.
            # We should exclude price_data from the main return dict if it breaks DataFrame creation,
            # or ensure the orchestrator separates it.

            # Let's verify how ICAT uses it.
            # icat.py -> clean(raw_data) -> pd.DataFrame(raw_data['historical'])
            # So 'historical' MUST contain only aligned lists.

            # We will move 'price_data' to a separate key in the orchestrator enrichment,
            # or exclude it here if it's not strictly financial statement history.

            # Let's keep it here but separate it in the Orchestrator?
            # Or better, the fetcher returns a dict where 'historical' is the aligned stuff,
            # and we return price_data separately?
            # But the signature is `fetch_historical_data` -> Dict.

            # Current fix: Do NOT include price_data in the main dict that becomes the DataFrame.
            # Instead, the Orchestrator should fetch market data separately, which already has 'price_data' key.
            # Wait, fetch_market_data returns 'price_data' as empty list currently.

            hist_price = t.history(period="6mo")
            price_series = hist_price["Close"].tolist() if not hist_price.empty else []

            # Return aligned financials
            return {
                "revenue": revenue,
                "ebitda": ebitda,
                "net_income": net_income,
                "total_assets": total_assets,
                "total_liabilities": total_liabilities,
                "total_debt": total_debt,
                "cash": cash,
                "interest_expense": interest_expense,
                "capex": capex,
                "year": years,
                "_price_history": price_series # Underscore to indicate auxiliary
            }

        except Exception as e:
            logger.error(f"Error fetching historical data for {ticker}: {e}")
            return {}

    def fetch_forecast_assumptions(self, ticker: str) -> Dict[str, Any]:
        """
        Derives forecast assumptions from analyst estimates or uses safe defaults.
        """
        try:
            t = yf.Ticker(ticker)

            # Try to enrich
            info = t.info

            # Growth estimates are hard to get reliably via free yfinance.
            # Look at analyst estimates if available.
            # For robustness, we will create a forward looking array for ICAT Engine.

            # Default Assumptions
            assumptions = {
                "revenue_growth": [0.05] * 5,
                "ebitda_margin": [0.25] * 5,
                "discount_rate": 0.10, # WACC proxy
                "terminal_growth_rate": 0.03
            }

            # Revenue Growth
            rev_growth = info.get("revenueGrowth")
            if rev_growth:
                assumptions["revenue_growth"] = [rev_growth] * 5

            # See if we can extract earnings growth to temper revenue growth
            eps_growth = info.get("earningsGrowth")
            if eps_growth and rev_growth:
                # Blend the two for a synthetic multi-year forecast curve (mean reversion)
                # Year 1: rev_growth. Year 5: converges to 3%
                curve = []
                for i in range(5):
                    weight = i / 4.0
                    curve.append(round(rev_growth * (1 - weight) + 0.03 * weight, 4))
                assumptions["revenue_growth"] = curve

            # Margins
            margins = info.get("ebitdaMargins")
            if margins:
                assumptions["ebitda_margin"] = [margins] * 5

            # WACC? Not direct. Use Beta.
            # Cost of Equity = Rf + Beta * MRP
            # Rf ~ 4.2%, MRP ~ 5.5%
            beta = info.get("beta", 1.0)
            ke = 0.042 + (beta * 0.055)
            # Cost of Debt? (Interest / Debt). Let's assume 6%
            kd = 0.06
            # WACC weight? D/E.
            # Simplified:
            assumptions["discount_rate"] = ke # Use Cost of Equity as conservative proxy for WACC in this simplified context

            return assumptions

        except Exception as e:
            logger.warning(f"Could not fetch forecast assumptions for {ticker}, using defaults: {e}")
            return {
                "revenue_growth": [0.05] * 5,
                "ebitda_margin": [0.25] * 5,
                "discount_rate": 0.10,
                "terminal_growth_rate": 0.03
            }
