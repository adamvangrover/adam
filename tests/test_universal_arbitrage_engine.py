import pytest
import math
from core.agents.specialized.universal_arbitrage_engine import UniversalArbitrageEngine, L2OrderBook, OrderBookLevel

@pytest.fixture
def engine():
    config = {
        "maker_fee": 0.0001,
        "taker_fee": 0.0002,
        "latency_ms": 10.0,
        "latency_decay_k": 0.1,
        "latency_decay_midpoint": 50.0
    }
    return UniversalArbitrageEngine(config=config)

def test_calculate_micro_price(engine):
    book = L2OrderBook(
        bids=[
            OrderBookLevel(price=99.0, volume=10.0),
            OrderBookLevel(price=98.0, volume=20.0)
        ],
        asks=[
            OrderBookLevel(price=101.0, volume=5.0),
            OrderBookLevel(price=102.0, volume=15.0)
        ]
    )

    # VWAP Bids: (99*10 + 98*20) / 30 = 2950 / 30 = 98.333
    # VWAP Asks: (101*5 + 102*15) / 20 = 2035 / 20 = 101.75
    # Total Bid Vol = 30, Total Ask Vol = 20
    # Imbalance I = 30 / (30 + 20) = 0.6
    # Micro-price = (0.6 * 101.75) + (0.4 * 98.333) = 61.05 + 39.333 = 100.3833

    micro_price = engine.calculate_micro_price(book, depth=5)
    assert pytest.approx(micro_price, 0.001) == 100.3833

def test_walk_book(engine):
    levels = [
        OrderBookLevel(price=100.0, volume=10.0),
        OrderBookLevel(price=101.0, volume=20.0),
    ]

    # Target volume exactly matches first level
    vol, cost, avg = engine.walk_book(levels, 10.0)
    assert vol == 10.0
    assert cost == 1000.0
    assert avg == 100.0

    # Target volume eats into second level
    vol, cost, avg = engine.walk_book(levels, 15.0)
    assert vol == 15.0
    # 10 @ 100 + 5 @ 101 = 1000 + 505 = 1505
    assert cost == 1505.0
    assert avg == 1505.0 / 15.0

def test_survival_probability(engine):
    # k=0.1, midpoint=50
    # at latency=10, exponent = 0.1 * (10 - 50) = -4
    # prob = 1 / (1 + e^-4) ≈ 0.982
    prob = engine.calculate_survival_probability()
    assert pytest.approx(prob, 0.01) == 0.982

    # Increase latency, prob should drop
    engine.latency_ms = 50.0
    prob2 = engine.calculate_survival_probability()
    assert prob2 == 0.5  # exactly at midpoint

    engine.latency_ms = 90.0
    prob3 = engine.calculate_survival_probability()
    assert prob3 < 0.05

def test_evaluate_arbitrage_profitable(engine):
    # Exchange A: Cheaper (Buy here)
    book_a = L2OrderBook(
        bids=[OrderBookLevel(price=98.0, volume=100.0)],
        asks=[OrderBookLevel(price=100.0, volume=10.0), OrderBookLevel(price=101.0, volume=10.0)]
    )
    # Exchange B: More expensive (Sell here)
    book_b = L2OrderBook(
        bids=[OrderBookLevel(price=105.0, volume=15.0), OrderBookLevel(price=104.0, volume=10.0)],
        asks=[OrderBookLevel(price=107.0, volume=100.0)]
    )

    # Target volume = 15
    # Buy on A: Need 15. Walk Asks. 10 @ 100 + 5 @ 101 = 1000 + 505 = 1505 cost
    # Sell on B: Need 15. Walk Bids. 15 @ 105 = 1575 revenue
    # Gross Profit = 1575 - 1505 = 70

    # Taker fees: 0.0002 on both sides
    # Cost fee = 1505 * 0.0002 = 0.301
    # Rev fee = 1575 * 0.0002 = 0.315
    # Total exec cost = 0.616

    # Net Profit = 70 - 0.616 = 69.384

    result = engine.evaluate_arbitrage(book_a, book_b, target_volume=15.0)

    assert result.volume_executed == 15.0
    assert pytest.approx(result.gross_spread, 0.01) == (105.0 - (1505.0/15.0))
    assert pytest.approx(result.execution_cost, 0.001) == 0.616
    assert pytest.approx(result.net_profit, 0.001) == 69.384

    # EV calculation
    # survival at 10ms = 0.982
    prob = engine.calculate_survival_probability()
    failure_penalty = 70 + 0.616 # abs(gross_profit) + exec_cost
    expected_ev = (prob * 69.384) - ((1 - prob) * failure_penalty)

    assert pytest.approx(result.ev, 0.01) == expected_ev
    assert pytest.approx(result.ev_penalty, 0.01) == (1 - prob) * failure_penalty
    assert result.is_profitable == True

def test_evaluate_arbitrage_unprofitable(engine):
    # Spreads are too tight, fees eat the profit
    book_a = L2OrderBook(
        bids=[OrderBookLevel(price=99.0, volume=100.0)],
        asks=[OrderBookLevel(price=100.0, volume=10.0)]
    )
    book_b = L2OrderBook(
        bids=[OrderBookLevel(price=100.01, volume=10.0)],
        asks=[OrderBookLevel(price=101.0, volume=100.0)]
    )

    result = engine.evaluate_arbitrage(book_a, book_b, target_volume=10.0)

    # Buy A at 100, Sell B at 100.01 -> Gross profit is 0.01 per unit * 10 = 0.1
    # Fees = 1000 * 0.0002 + 1000.1 * 0.0002 = 0.2 + 0.20002 = 0.40002
    # Net Profit = 0.1 - 0.40002 = -0.30002

    assert result.net_profit < 0
    assert result.ev < 0
    assert result.is_profitable == False
