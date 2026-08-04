"""
Tests for the Event Sourcing Core Ledger.
"""

import pytest
from src.core.event_sourcing.ledger import EventLedger, FinancialEvent

@pytest.fixture
def ledger():
    return EventLedger()

def test_financial_event_hash_generation():
    """Test that a hash is generated deterministically upon event creation."""
    event = FinancialEvent(
        event_id="evt_01",
        aggregate_id="portfolio_01",
        event_type="FUNDS_DEPOSITED",
        payload={"amount_usd": 10000.0},
        timestamp="2023-10-27T10:00:00Z"
    )
    assert event.event_hash != ""
    assert isinstance(event.event_hash, str)

def test_ledger_append_and_retrieve(ledger):
    """Test appending an event and retrieving it by aggregate_id."""
    event1 = FinancialEvent(
        event_id="evt_01",
        aggregate_id="portfolio_01",
        event_type="FUNDS_DEPOSITED",
        payload={"amount_usd": 10000.0},
        timestamp="2023-10-27T10:00:00Z"
    )
    event2 = FinancialEvent(
        event_id="evt_02",
        aggregate_id="portfolio_02",
        event_type="FUNDS_DEPOSITED",
        payload={"amount_usd": 5000.0},
        timestamp="2023-10-27T10:05:00Z"
    )

    ledger.append_event(event1)
    ledger.append_event(event2)

    portfolio_1_events = ledger.get_events_for_aggregate("portfolio_01")
    assert len(portfolio_1_events) == 1
    assert portfolio_1_events[0].event_id == event1.event_id
    assert portfolio_1_events[0].payload["amount_usd"] == 10000.0

def test_ledger_tampering_prevention(ledger):
    """Test that appending a tampered event raises an error."""
    event = FinancialEvent(
        event_id="evt_01",
        aggregate_id="portfolio_01",
        event_type="FUNDS_DEPOSITED",
        payload={"amount_usd": 10000.0},
        timestamp="2023-10-27T10:00:00Z"
    )

    # Tamper with the payload after creation (and hash generation)
    event.payload["amount_usd"] = 1000000.0

    with pytest.raises(ValueError, match="failed integrity check"):
        ledger.append_event(event)

def test_ledger_replay_aggregate(ledger):
    """Test reconstructing state from a series of events."""

    # Generate a series of events
    ledger.append_event(FinancialEvent(
        event_id="evt_01",
        aggregate_id="portfolio_01",
        event_type="FUNDS_DEPOSITED",
        payload={"amount": 10000.0},
        timestamp="2023-10-27T10:00:00Z"
    ))
    ledger.append_event(FinancialEvent(
        event_id="evt_02",
        aggregate_id="portfolio_01",
        event_type="ASSET_PURCHASED",
        payload={"amount": 4000.0, "asset": "AAPL"},
        timestamp="2023-10-27T10:05:00Z"
    ))
    ledger.append_event(FinancialEvent(
        event_id="evt_03",
        aggregate_id="portfolio_01",
        event_type="FUNDS_WITHDRAWN",
        payload={"amount": 1000.0},
        timestamp="2023-10-27T10:10:00Z"
    ))

    # Define a simple reducer
    def portfolio_reducer(state: dict, event: FinancialEvent) -> dict:
        if event.event_type == "FUNDS_DEPOSITED":
            state["balance"] += event.payload["amount"]
        elif event.event_type == "FUNDS_WITHDRAWN":
            state["balance"] -= event.payload["amount"]
        elif event.event_type == "ASSET_PURCHASED":
            state["balance"] -= event.payload["amount"]
            state["assets"].append(event.payload["asset"])
        return state

    # Replay
    initial_state = {"balance": 0.0, "assets": []}
    final_state = ledger.replay_aggregate("portfolio_01", portfolio_reducer, initial_state)

    assert final_state["balance"] == 5000.0
    assert "AAPL" in final_state["assets"]
