import pytest
import pandas as pd
import asyncio
from unittest.mock import MagicMock, patch, AsyncMock
import sys
import os

# Add repo root to path so we can import the module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

class FakeNeuralPacket:
    def __init__(self, source_agent, packet_type, payload):
        self.source_agent = source_agent
        self.packet_type = packet_type
        self.payload = payload

# Mock the bridge module before importing anything that depends on it
mock_bridge = MagicMock()
mock_bridge.NeuralPacket = FakeNeuralPacket
sys.modules["core.v30_architecture.python_intelligence.bridge.neural_mesh"] = mock_bridge

# Now import the module under test
import core.v30_architecture.python_intelligence.agents.quantitative_analyst as quant_module
from core.v30_architecture.python_intelligence.agents.quantitative_analyst import QuantitativeAnalyst

@pytest.mark.asyncio
async def test_quantitative_analyst_analysis():
    # Mock yfinance
    with patch.object(quant_module, 'yf') as mock_yf:
        # Create a mock DataFrame with MultiIndex columns to simulate batch download
        dates = pd.date_range(start="2023-01-01", periods=100, freq="15min")

        # Base data
        data_spy = {
            "Open": [100.0] * 100,
            "High": [105.0] * 100,
            "Low": [95.0] * 100,
            "Close": [100.0 + i * 0.1 for i in range(100)],
            "Volume": [1000] * 100
        }

        data_btc = {
             "Open": [30000.0] * 100,
             "High": [31000.0] * 100,
             "Low": [29000.0] * 100,
             "Close": [30000.0 + i * 10 for i in range(100)],
             "Volume": [500] * 100
        }

        # Construct MultiIndex DataFrame
        # Columns: (Ticker, Field)
        # Using pd.concat to build it properly
        df_spy = pd.DataFrame(data_spy, index=dates)
        df_btc = pd.DataFrame(data_btc, index=dates)

        # yfinance with group_by='ticker' returns top level as Ticker
        # Structure: Ticker -> Open, High...
        # So we can concat with keys
        df = pd.concat([df_spy, df_btc], axis=1, keys=['SPY', 'BTC-USD'])

        # Mock yf.download return value
        mock_yf.download.return_value = df

        # Mock emit_packet in base_agent
        with patch("core.v30_architecture.python_intelligence.agents.base_agent.emit_packet", new_callable=AsyncMock) as mock_emit:
            agent = QuantitativeAnalyst()
            # Override tickers to match our mock data
            agent.tickers = ["SPY", "BTC-USD"]

            await agent.analyze_all_tickers()

            # Verify emit was called twice (once for SPY, once for BTC)
            assert mock_emit.call_count == 2

            # Inspect calls
            calls = mock_emit.call_args_list

            # Check SPY payload
            # SPY Last Close: 100 + 99*0.1 = 109.9
            spy_call = [c for c in calls if c[0][0].payload['symbol'] == 'SPY'][0]
            spy_packet = spy_call[0][0]
            assert spy_packet.payload['price'] == 109.9
            assert spy_packet.payload['rsi'] is not None

            # Check BTC payload
            # BTC Last Close: 30000 + 99*10 = 30990.0
            btc_call = [c for c in calls if c[0][0].payload['symbol'] == 'BTC-USD'][0]
            btc_packet = btc_call[0][0]
            assert btc_packet.payload['price'] == 30990.0
