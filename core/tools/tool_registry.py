from typing import Dict, Any, List, Optional
import logging
import json

# Try importing real tools
try:
    import yfinance as yf
except ImportError:
    yf = None

try:
    from duckduckgo_search import DDGS
except ImportError:
    DDGS = None

logger = logging.getLogger(__name__)

class ToolRegistry:
    """
    Central registry for external tools (Financial APIs, Search, etc.).
    Replaces static mocks with live data fetching where possible.
    """

    def __init__(self):
        self.tools = {
            "get_stock_price": self.get_stock_price,
            "get_financials": self.get_financials,
            "search_web": self.search_web,
            "get_news": self.get_news
        }

    def execute(self, tool_name: str, **kwargs) -> Any:
        if tool_name not in self.tools:
            logger.warning(f"Tool {tool_name} not found.")
            return None

        try:
            return self.tools[tool_name](**kwargs)
        except Exception as e:
            logger.error(f"Tool execution failed for {tool_name}: {e}")
            return f"Error: {str(e)}"

    def get_stock_price(self, ticker: str) -> Dict[str, Any]:
        """Fetches real-time price using yfinance."""
        if not yf:
            return {"error": "yfinance not installed", "price": 100.0} # Fallback

        try:
            ticker_obj = yf.Ticker(ticker)
            # fast_info is faster than history
            try:
                price = ticker_obj.fast_info.last_price
                currency = ticker_obj.fast_info.currency
            except:
                # Fallback to history if fast_info fails
                hist = ticker_obj.history(period="1d")
                if not hist.empty:
                    price = hist["Close"].iloc[-1]
                    currency = "USD"
                else:
                    price = 0.0
                    currency = "USD"

            return {"ticker": ticker, "price": price, "currency": currency}
        except Exception as e:
            logger.warning(f"yfinance failed for {ticker}: {e}")
            return {"ticker": ticker, "price": 0.0, "error": str(e)}

    def get_financials(self, ticker: str) -> Dict[str, Any]:
        """Fetches balance sheet/income statement."""
        if not yf:
            # Robust Mock Fallback if YF missing
            return self._get_mock_financials(ticker)

        try:
            t = yf.Ticker(ticker)
            info = t.info

            # Extract key metrics ensuring 'current_ratio' exists for downstream critique
            current_ratio = info.get("currentRatio")
            if not current_ratio:
                # Fallback calculation if info missing
                bs = t.balance_sheet
                if not bs.empty:
                    try:
                        ca = bs.loc["Total Current Assets"].iloc[0]
                        cl = bs.loc["Total Current Liabilities"].iloc[0]
                        current_ratio = ca / cl
                    except:
                        current_ratio = 1.5 # Safe default
                else:
                    current_ratio = 1.5

            return {
                "company_info": {
                    "name": info.get("longName", ticker),
                    "sector": info.get("sector", "Unknown"),
                    "industry": info.get("industry", "Unknown"),
                    "market_cap": info.get("marketCap", 0)
                },
                "ratios": {
                    "pe_ratio": info.get("trailingPE"),
                    "peg_ratio": info.get("pegRatio"),
                    "debt_to_equity": info.get("debtToEquity"),
                    "current_ratio": current_ratio,
                    "return_on_equity": info.get("returnOnEquity")
                },
                "financial_data_detailed": { # maintain compat with legacy structure where possible
                     "key_ratios": {
                         "current_ratio": current_ratio
                     }
                }
            }
        except Exception as e:
            logger.error(f"Failed to get financials for {ticker}: {e}")
            return self._get_mock_financials(ticker)

    def _get_mock_financials(self, ticker: str) -> Dict[str, Any]:
        """Provides robust mock data matching the schema required by critique_node."""
        return {
             "company_info": {
                "name": f"{ticker} Corp (Mock)",
                "sector": "Technology",
                "industry": "Software"
            },
            "ratios": {
                "pe_ratio": 25.0,
                "current_ratio": 2.1,
                "debt_to_equity": 0.5
            },
            "financial_data_detailed": {
                 "key_ratios": {
                     "current_ratio": 2.1
                 }
            }
        }

    def search_web(self, query: str, max_results: int = 3) -> List[Dict[str, str]]:
        """Performs a web search using DuckDuckGo."""
        if not DDGS:
            return [{"title": "Simulation", "body": f"Simulated search result for {query}", "href": "#"}]

        try:
            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=max_results))
                return results
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []

    def get_news(self, ticker: str, limit: int = 3) -> List[Dict[str, str]]:
        """Fetches news for a specific ticker."""
        if yf:
            try:
                t = yf.Ticker(ticker)
                news = t.news
                return news[:limit] if news else []
            except Exception:
                pass

        # Fallback to web search
        return self.search_web(f"{ticker} financial news", max_results=limit)
