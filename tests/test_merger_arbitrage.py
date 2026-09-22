import pytest
from scripts.merger_arbitrage_forecaster import calculate_market_implied_probability, process_merger_arbitrage
from src.schemas.core_types import AgentOutput

def test_calculate_market_implied_probability():
    prob = calculate_market_implied_probability(110.0, 80.0, 120.0)
    assert prob == pytest.approx(0.75)

    prob_clamped = calculate_market_implied_probability(130.0, 80.0, 120.0)
    assert prob_clamped == pytest.approx(1.0)

    prob_clamped_low = calculate_market_implied_probability(70.0, 80.0, 120.0)
    assert prob_clamped_low == pytest.approx(0.0)

def test_process_merger_arbitrage():
    data = {
        "deal_id": "DEAL-123",
        "acquirer": "Corp A",
        "target": "Corp B",
        "current_price": 110.0,
        "downside_price": 80.0,
        "upside_price": 120.0,
        "company_guided_days": 150
    }
    output = process_merger_arbitrage(data, "Antitrust clearance expected.")

    assert isinstance(output, AgentOutput)
    assert output.provenance_trace.source_data_object == "DEAL-123"
    assert "probabilities" in output.data
    assert output.data["probabilities"]["Succeed+"] == pytest.approx(0.75 * 0.9)
    assert output.data["days_to_completion"] == 150
