import pytest
import asyncio
from unittest.mock import MagicMock, AsyncMock, patch
from core.agents.crypto_arbitrage_agent import CryptoArbitrageAgent
from core.schemas.agent_schema import AgentInput, AgentOutput

# Mock ccxt
@pytest.fixture
def mock_ccxt():
    with patch('core.agents.crypto_arbitrage_agent.ccxt') as mock:
        yield mock

@pytest.mark.asyncio
async def test_crypto_arbitrage_agent_init(mock_ccxt):
    config = {'exchanges': ['binance', 'coinbase'], 'symbols': ['BTC/USDT']}
    agent = CryptoArbitrageAgent(config)
    assert agent.exchanges == ['binance', 'coinbase']
    assert agent.symbols == ['BTC/USDT']
    assert agent.min_profit_threshold == 0.5

@pytest.mark.asyncio
async def test_crypto_arbitrage_execution_no_arb(mock_ccxt):
    # Setup mocks
    mock_binance = AsyncMock()
    mock_binance.fetch_ticker.return_value = {'last': 50000.0}

    mock_coinbase = AsyncMock()
    mock_coinbase.fetch_ticker.return_value = {'last': 50010.0} # Small difference

    # Mock the exchange classes
    mock_ccxt.binance.return_value = mock_binance
    mock_ccxt.coinbase.return_value = mock_coinbase

    config = {'exchanges': ['binance', 'coinbase'], 'symbols': ['BTC/USDT'], 'min_profit_threshold': 0.5}
    agent = CryptoArbitrageAgent(config)

    result = await agent.execute("Scan")

    assert isinstance(result, AgentOutput)
    assert "No arbitrage opportunities" in result.answer
    assert len(result.metadata['opportunities']) == 0

@pytest.mark.asyncio
async def test_crypto_arbitrage_execution_with_arb(mock_ccxt):
    # Setup mocks for arbitrage opportunity
    # Binance: 50000, Coinbase: 51000 (2% diff)
    mock_binance = AsyncMock()
    mock_binance.fetch_ticker.return_value = {'last': 50000.0}

    mock_coinbase = AsyncMock()
    mock_coinbase.fetch_ticker.return_value = {'last': 51000.0}

    mock_ccxt.binance.return_value = mock_binance
    mock_ccxt.coinbase.return_value = mock_coinbase

    config = {'exchanges': ['binance', 'coinbase'], 'symbols': ['BTC/USDT'], 'min_profit_threshold': 1.0}
    agent = CryptoArbitrageAgent(config)

    result = await agent.execute("Scan")

    assert isinstance(result, AgentOutput)
    assert "Found" in result.answer
    assert len(result.metadata['opportunities']) == 1

    op = result.metadata['opportunities'][0]
    assert op['buy_exchange'] == 'binance'
    assert op['sell_exchange'] == 'coinbase'
    assert op['profit_pct'] == 2.0

@pytest.mark.asyncio
async def test_crypto_arbitrage_error_handling(mock_ccxt):
    # Setup mocks to raise exception
    mock_binance = AsyncMock()
    mock_binance.fetch_ticker.side_effect = Exception("Network Error")

    mock_coinbase = AsyncMock()
    mock_coinbase.fetch_ticker.return_value = {'last': 50000.0}

    mock_ccxt.binance.return_value = mock_binance
    mock_ccxt.coinbase.return_value = mock_coinbase

    config = {'exchanges': ['binance', 'coinbase'], 'symbols': ['BTC/USDT']}
    agent = CryptoArbitrageAgent(config)

    result = await agent.execute("Scan")

    # Should handle error gracefully and return 0 ops (since we need 2 prices)
    assert len(result.metadata['opportunities']) == 0
