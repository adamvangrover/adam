import pytest
import asyncio
import pandas as pd
from unittest.mock import MagicMock, patch, AsyncMock
from core.v30_architecture.python_intelligence.agents.market_scanner import MarketScanner

@pytest.mark.asyncio
async def test_market_scanner_fetch_and_emit():
    # Setup
    scanner = MarketScanner(tickers=["TEST"])

    # Mock yfinance result structure for single ticker
    mock_df = pd.DataFrame({
        "Close": [100.0, 101.0, 102.0, 103.0, 105.0],
        "Volume": [1000, 1100, 1200, 1300, 1400]
    }, index=pd.date_range("2023-01-01", periods=5, freq="15min"))

    # Mock emit because it's async and on the base class
    scanner.emit = AsyncMock()

    # Act
    # We test the internal logic `_process_and_emit` directly
    await scanner._process_and_emit(mock_df)

    # Assert
    assert scanner.emit.call_count == 1
    call_args = scanner.emit.call_args[0]
    assert call_args[0] == "market_data"
    payload = call_args[1]

    assert payload["symbol"] == "TEST"
    assert payload["price"] == 105.0
    assert payload["volume"] == 1400

    # Check calc logic: (105 - 100) / 100 * 100 = 5.0 (since len=5, it uses index -5 which is 100)
    assert payload["change_pct"] == 5.0

@pytest.mark.asyncio
async def test_market_scanner_multi_ticker():
    scanner = MarketScanner(tickers=["AAPL", "GOOG"])
    scanner.emit = AsyncMock()

    # Mock MultiIndex DataFrame
    # Columns: (Ticker, Level)
    arrays = [
        ["AAPL", "AAPL", "GOOG", "GOOG"],
        ["Close", "Volume", "Close", "Volume"]
    ]
    tuples = list(zip(*arrays))
    index = pd.MultiIndex.from_tuples(tuples, names=["Ticker", "Price"])

    data = [
        [150.0, 1000, 2800.0, 500],
        [151.0, 1100, 2810.0, 510],
        [152.0, 1200, 2820.0, 520],
        [153.0, 1300, 2830.0, 530],
        [155.0, 1400, 2850.0, 550]
    ]
    mock_df = pd.DataFrame(data, index=pd.date_range("2023-01-01", periods=5, freq="15min"), columns=index)

    await scanner._process_and_emit(mock_df)

    assert scanner.emit.call_count == 2

    # Verify payloads
    calls = scanner.emit.call_args_list
    payloads = {c[0][1]["symbol"]: c[0][1] for c in calls}

    assert "AAPL" in payloads
    assert payloads["AAPL"]["price"] == 155.0
    # Change: (155 - 150)/150 = 3.33
    assert payloads["AAPL"]["change_pct"] == 3.33

    assert "GOOG" in payloads
    assert payloads["GOOG"]["price"] == 2850.0
    # Change: (2850 - 2800)/2800 = 1.79
    assert payloads["GOOG"]["change_pct"] == 1.79
