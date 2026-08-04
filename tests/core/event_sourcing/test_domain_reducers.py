"""
Tests for Domain Models and Reducers in the Event Sourcing Core.
"""

import pytest
from src.core.event_sourcing.ledger import EventLedger, FinancialEvent
from src.core.event_sourcing.domain import SecurityState, MarketState
from src.core.event_sourcing.reducers import security_reducer, market_reducer

@pytest.fixture
def ledger():
    return EventLedger()

def test_security_lifecycle_replay(ledger):
    """Test the complete lifecycle of a security via event replay."""
    ticker = "NVDA"

    # 1. Issuance
    ledger.append_event(FinancialEvent(
        event_id="evt_sec_01",
        aggregate_id=ticker,
        event_type="SECURITY_ISSUED",
        payload={
            "ticker": ticker,
            "asset_class": "Equity",
            "initial_shares": 1000000,
            "initial_price": 50.0
        },
        timestamp="2023-01-01T10:00:00Z"
    ))

    # 2. Risk Rating Update
    ledger.append_event(FinancialEvent(
        event_id="evt_sec_02",
        aggregate_id=ticker,
        event_type="RISK_RATING_UPDATED",
        payload={
            "agency": "Moody's",
            "rating": "A1",
            "outlook": "Positive"
        },
        timestamp="2023-02-01T10:00:00Z"
    ))

    # 3. Trading Level Change
    ledger.append_event(FinancialEvent(
        event_id="evt_sec_03",
        aggregate_id=ticker,
        event_type="TRADING_LEVEL_CHANGED",
        payload={
            "price": 150.0,
            "volume": 50000,
            "liquidity_score": 0.95
        },
        timestamp="2023-03-01T10:00:00Z"
    ))

    # 4. News Trigger
    ledger.append_event(FinancialEvent(
        event_id="evt_sec_04",
        aggregate_id=ticker,
        event_type="NEWS_TRIGGERED",
        payload={
            "source": "Bloomberg",
            "headline": "Earnings beat expectations",
            "sentiment_score": 0.8,
            "impact_severity": 4
        },
        timestamp="2023-03-01T10:30:00Z"
    ))

    # 5. Pricing Target Update
    ledger.append_event(FinancialEvent(
        event_id="evt_sec_05",
        aggregate_id=ticker,
        event_type="PRICING_TARGET_UPDATED",
        payload={
            "analyst_id": "Goldman",
            "target_price": 200.0,
            "horizon_months": 12
        },
        timestamp="2023-03-02T10:00:00Z"
    ))

    # Replay all events
    initial_state = SecurityState(aggregate_id=ticker)
    final_state = ledger.replay_aggregate(ticker, security_reducer, initial_state)

    assert final_state.asset_class == "Equity"
    assert final_state.shares_outstanding == 1000000
    assert final_state.current_price == 150.0
    assert final_state.risk_ratings["Moody's"]["rating"] == "A1"
    assert final_state.analyst_targets["Goldman"] == 200.0
    assert final_state.news_count == 1
    assert final_state.news_sentiment_aggregate == 0.8
    assert final_state.last_trading_volume == 50000

def test_market_state_replay(ledger):
    """Test macro condition events updating the MarketState."""
    market_id = "global_market"

    # 1. Inflation spike
    ledger.append_event(FinancialEvent(
        event_id="evt_mac_01",
        aggregate_id=market_id,
        event_type="MACRO_CONDITION_CHANGED",
        payload={
            "indicator": "CPI",
            "value": 5.4,
            "regime": "High-Inflation"
        },
        timestamp="2023-01-01T10:00:00Z"
    ))

    # 2. Fed rate hike
    ledger.append_event(FinancialEvent(
        event_id="evt_mac_02",
        aggregate_id=market_id,
        event_type="MACRO_CONDITION_CHANGED",
        payload={
            "indicator": "Fed Funds Rate",
            "value": 4.5,
            "regime": "Tightening"
        },
        timestamp="2023-02-01T10:00:00Z"
    ))

    initial_state = MarketState(aggregate_id=market_id)
    final_state = ledger.replay_aggregate(market_id, market_reducer, initial_state)

    assert final_state.current_regime == "Tightening"
    assert final_state.indicators["CPI"] == 5.4
    assert final_state.indicators["Fed Funds Rate"] == 4.5

def test_invalid_payload_skipped(ledger):
    """Test that an event with an invalid payload for its type is caught and doesn't break replay."""
    ticker = "AAPL"

    # 1. Valid issuance
    ledger.append_event(FinancialEvent(
        event_id="evt_sec_01",
        aggregate_id=ticker,
        event_type="SECURITY_ISSUED",
        payload={
            "ticker": ticker,
            "asset_class": "Equity",
            "initial_shares": 1000000,
            "initial_price": 50.0
        },
        timestamp="2023-01-01T10:00:00Z"
    ))

    # 2. Invalid risk rating (missing required fields like 'agency')
    ledger.append_event(FinancialEvent(
        event_id="evt_sec_02",
        aggregate_id=ticker,
        event_type="RISK_RATING_UPDATED",
        payload={
            "wrong_field": "Moody's"
        },
        timestamp="2023-02-01T10:00:00Z"
    ))

    initial_state = SecurityState(aggregate_id=ticker)
    final_state = ledger.replay_aggregate(ticker, security_reducer, initial_state)

    # State should reflect the first event, and the second should be safely ignored
    assert final_state.asset_class == "Equity"
    assert final_state.current_price == 50.0
    assert len(final_state.risk_ratings) == 0
