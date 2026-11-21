
import pytest
import asyncio
from unittest.mock import AsyncMock, patch, MagicMock
from core.agents.specialized.crypto_arbitrage_agent import CryptoArbitrageAgent

@pytest.mark.asyncio
async def test_arbitrage_detection_success():
    """Test that arbitrage is correctly identified."""
    config = {"agent_id": "crypto_arb_test"}
    agent = CryptoArbitrageAgent(config)

    # Mock _fetch_price to return controlled values
    # BTC on binance: 50000
    # BTC on kraken: 51000 (2% spread)
    async def mock_fetch_price(exchange_id, symbol):
        if symbol == "BTC/USDT":
            if exchange_id == "binance": return 50000.0
            if exchange_id == "kraken": return 51000.0
        return None

    # Patch _fetch_price instead of ccxt to test the logic core
    with patch.object(agent, '_fetch_price', side_effect=mock_fetch_price):
        # Also mock _init_exchanges and _close_exchanges to avoid CCXT calls
        with patch.object(agent, '_init_exchanges', new_callable=AsyncMock), \
             patch.object(agent, '_close_exchanges', new_callable=AsyncMock):

            result = await agent.execute(symbols=["BTC/USDT"], exchanges=["binance", "kraken"], min_spread=1.0)

            assert result["status"] == "success"
            opps = result["opportunities"]
            assert len(opps) == 1
            opp = opps[0]
            assert opp["symbol"] == "BTC/USDT"
            assert opp["buy_exchange"] == "binance"
            assert opp["sell_exchange"] == "kraken"
            assert opp["spread_percentage"] == 2.0
            assert opp["estimated_profit"] == 1000.0

@pytest.mark.asyncio
async def test_no_arbitrage():
    """Test that small spreads are ignored."""
    config = {"agent_id": "crypto_arb_test"}
    agent = CryptoArbitrageAgent(config)

    # Spread is only 0.2%
    async def mock_fetch_price(exchange_id, symbol):
        if symbol == "ETH/USDT":
            if exchange_id == "binance": return 3000.0
            if exchange_id == "kraken": return 3006.0
        return None

    with patch.object(agent, '_fetch_price', side_effect=mock_fetch_price):
        with patch.object(agent, '_init_exchanges', new_callable=AsyncMock), \
             patch.object(agent, '_close_exchanges', new_callable=AsyncMock):

            result = await agent.execute(symbols=["ETH/USDT"], exchanges=["binance", "kraken"], min_spread=1.0)

            assert result["status"] == "no_opportunities_found"
            assert len(result["opportunities"]) == 0

@pytest.mark.asyncio
async def test_missing_data():
    """Test handling of missing price data."""
    config = {"agent_id": "crypto_arb_test"}
    agent = CryptoArbitrageAgent(config)

    # One exchange fails to return data
    async def mock_fetch_price(exchange_id, symbol):
        if exchange_id == "binance": return 50000.0
        return None

    with patch.object(agent, '_fetch_price', side_effect=mock_fetch_price):
        with patch.object(agent, '_init_exchanges', new_callable=AsyncMock), \
             patch.object(agent, '_close_exchanges', new_callable=AsyncMock):

            result = await agent.execute(symbols=["BTC/USDT"], exchanges=["binance", "kraken"], min_spread=1.0)

            # Should be empty because we need at least 2 prices
            assert len(result["opportunities"]) == 0
