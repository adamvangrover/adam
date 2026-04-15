import pytest
from src.agents.crypto_arbitrage import CryptoArbitrageAgent, AssetQuote

def test_arbitrage_opportunity_found():
    agent = CryptoArbitrageAgent(min_spread_pct=0.1)
    quotes = [
        AssetQuote(exchange="Binance", symbol="BTC/USD", bid_price=50000, ask_price=50050),
        AssetQuote(exchange="Coinbase", symbol="BTC/USD", bid_price=50200, ask_price=50250)
    ]
    # Should buy on Binance (ask 50050) and sell on Coinbase (bid 50200)
    opportunities = agent.analyze_market(quotes)

    assert len(opportunities) == 1
    opp = opportunities[0]
    assert opp.symbol == "BTC/USD"
    assert opp.buy_exchange == "Binance"
    assert opp.sell_exchange == "Coinbase"
    assert opp.estimated_profit == 150.0  # 50200 - 50050
    assert opp.spread_pct > 0.2

def test_no_arbitrage_opportunity():
    agent = CryptoArbitrageAgent(min_spread_pct=0.5)
    quotes = [
        AssetQuote(exchange="Binance", symbol="ETH/USD", bid_price=3000, ask_price=3010),
        AssetQuote(exchange="Coinbase", symbol="ETH/USD", bid_price=3005, ask_price=3015)
    ]
    # Max bid (3005) - Min ask (3010) = -5 (No spread)
    opportunities = agent.analyze_market(quotes)
    assert len(opportunities) == 0

def test_execute_kwargs():
    agent = CryptoArbitrageAgent(min_spread_pct=0.1)
    quotes_dict = [
        {"exchange": "Kraken", "symbol": "SOL/USD", "bid_price": 100, "ask_price": 101},
        {"exchange": "FTX", "symbol": "SOL/USD", "bid_price": 105, "ask_price": 106}
    ]
    res = agent.execute(quotes=quotes_dict)
    assert len(res) == 1
    assert res[0].estimated_profit == 4.0
