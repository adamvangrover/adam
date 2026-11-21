import pytest

from src.agents.universal_arbitrage import (
    OrderBook,
    OrderBookLevel,
    UniversalArbitrageEngine,
)


def test_arbitrage_opportunity_found():
    engine = UniversalArbitrageEngine(min_ev=0.0)

    binance_book = OrderBook(
        exchange="Binance",
        symbol="BTC/USD",
        bids=[OrderBookLevel(price=50000, volume=10.0)],
        asks=[OrderBookLevel(price=50050, volume=10.0)],
        taker_fee_bps=10,
        maker_fee_bps=0,
        network_latency_ms=1.0,
    )

    coinbase_book = OrderBook(
        exchange="Coinbase",
        symbol="BTC/USD",
        bids=[OrderBookLevel(price=50200, volume=10.0)],
        asks=[OrderBookLevel(price=50250, volume=10.0)],
        taker_fee_bps=10,
        maker_fee_bps=0,
        network_latency_ms=1.0,
    )

    # We want to buy on Binance (Ask 50050) and sell on Coinbase (Bid 50200)
    # Trade size = 1.0
    opportunities = engine.analyze_market(
        [binance_book, coinbase_book], trade_sizes=[1.0]
    )

    assert len(opportunities) >= 1

    # The best opportunity should be buying on Binance and selling on Coinbase
    best_opp = opportunities[0]

    assert best_opp.symbol == "BTC/USD"
    assert best_opp.buy_exchange == "Binance"
    assert best_opp.sell_exchange == "Coinbase"

    # Gross revenue: Sell 1 BTC @ 50200 - Buy 1 BTC @ 50050 = 150
    assert best_opp.gross_revenue == 150.0

    # Taker fee on buy: 50050 * 10 / 10000 = 50.05
    # Taker fee on sell: 50200 * 10 / 10000 = 50.20
    # Net profit: 150 - 50.05 - 50.20 = 49.75
    assert best_opp.net_profit == pytest.approx(49.75)


def test_no_arbitrage_opportunity():
    engine = UniversalArbitrageEngine(min_ev=0.0)

    binance_book = OrderBook(
        exchange="Binance",
        symbol="ETH/USD",
        bids=[OrderBookLevel(price=3000, volume=10.0)],
        asks=[OrderBookLevel(price=3010, volume=10.0)],
        taker_fee_bps=10,
        maker_fee_bps=0,
        network_latency_ms=10.0,
    )

    coinbase_book = OrderBook(
        exchange="Coinbase",
        symbol="ETH/USD",
        bids=[OrderBookLevel(price=3005, volume=10.0)],
        asks=[OrderBookLevel(price=3015, volume=10.0)],
        taker_fee_bps=10,
        maker_fee_bps=0,
        network_latency_ms=10.0,
    )

    # Max Bid is 3005 on Coinbase, Min Ask is 3010 on Binance. No spread.
    opportunities = engine.analyze_market(
        [binance_book, coinbase_book], trade_sizes=[1.0]
    )

    assert len(opportunities) == 0


def test_execute_kwargs():
    engine = UniversalArbitrageEngine(min_ev=0.0)

    books_dict = [
        {
            "exchange": "Kraken",
            "symbol": "SOL/USD",
            "bids": [{"price": 100, "volume": 10}],
            "asks": [{"price": 101, "volume": 10}],
            "taker_fee_bps": 0,
            "maker_fee_bps": 0,
            "network_latency_ms": 0,
        },
        {
            "exchange": "FTX",
            "symbol": "SOL/USD",
            "bids": [{"price": 105, "volume": 10}],
            "asks": [{"price": 106, "volume": 10}],
            "taker_fee_bps": 0,
            "maker_fee_bps": 0,
            "network_latency_ms": 0,
        },
    ]

    res = engine.execute(order_books=books_dict, trade_sizes=[1.0])

    assert len(res) >= 1
    best_opp = res[0]

    # Buy on Kraken at 101, Sell on FTX at 105 -> Gross Revenue = 4
    # Since fees are 0, Net Profit = 4
    assert best_opp.gross_revenue == 4.0
    assert best_opp.net_profit == 4.0
