from pydantic import BaseModel, Field
from typing import List, Dict, Optional

class AssetQuote(BaseModel):
    exchange: str
    symbol: str
    bid_price: float
    ask_price: float

class ArbitrageOpportunity(BaseModel):
    symbol: str
    buy_exchange: str
    sell_exchange: str
    spread_pct: float
    estimated_profit: float
    is_executable: bool

class CryptoArbitrageAgent:
    """
    Agent designed to find pricing inefficiencies across multiple crypto exchanges.
    """
    def __init__(self, min_spread_pct: float = 0.5):
        self.min_spread_pct = min_spread_pct

    def analyze_market(self, quotes: List[AssetQuote]) -> List[ArbitrageOpportunity]:
        opportunities = []
        # Group quotes by symbol
        grouped_quotes: Dict[str, List[AssetQuote]] = {}
        for quote in quotes:
            grouped_quotes.setdefault(quote.symbol, []).append(quote)

        for symbol, q_list in grouped_quotes.items():
            if len(q_list) < 2:
                continue

            # Find max bid and min ask
            max_bid_quote = max(q_list, key=lambda x: x.bid_price)
            min_ask_quote = min(q_list, key=lambda x: x.ask_price)

            # Cross exchange check
            if max_bid_quote.exchange != min_ask_quote.exchange:
                spread = max_bid_quote.bid_price - min_ask_quote.ask_price
                spread_pct = (spread / min_ask_quote.ask_price) * 100

                if spread_pct >= self.min_spread_pct:
                    opp = ArbitrageOpportunity(
                        symbol=symbol,
                        buy_exchange=min_ask_quote.exchange,
                        sell_exchange=max_bid_quote.exchange,
                        spread_pct=round(spread_pct, 4),
                        estimated_profit=round(spread, 4),
                        is_executable=True
                    )
                    opportunities.append(opp)

        return sorted(opportunities, key=lambda x: x.spread_pct, reverse=True)

    def execute(self, **kwargs):
        quotes_data = kwargs.get('quotes', [])
        quotes = [AssetQuote(**q) if isinstance(q, dict) else q for q in quotes_data]
        return self.analyze_market(quotes)
