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
    # We patch 'yf' in the module namespace
    with patch.object(quant_module, 'yf') as mock_yf:
        # Create a mock DataFrame
        dates = pd.date_range(start="2023-01-01", periods=100, freq="15min")
        data = {
            "Open": [100.0] * 100,
            "High": [105.0] * 100,
            "Low": [95.0] * 100,
            "Close": [100.0 + i * 0.1 for i in range(100)],
            "Volume": [1000] * 100
        }
        df = pd.DataFrame(data, index=dates)

        mock_instance = MagicMock()
        mock_instance.history.return_value = df
        mock_yf.Ticker.return_value = mock_instance

        # Mock emit_packet in base_agent
        with patch("core.v30_architecture.python_intelligence.agents.base_agent.emit_packet", new_callable=AsyncMock) as mock_emit:
            agent = QuantitativeAnalyst()
            await agent.analyze_ticker("SPY")

            # Verify emit was called
            assert mock_emit.called
            args, _ = mock_emit.call_args
            packet = args[0]

            # Now packet should be an instance of FakeNeuralPacket
            assert packet.source_agent == "Quant-V30"
            assert packet.packet_type == "technical_analysis"
            assert packet.payload["symbol"] == "SPY"
            # Last close is 100 + 99*0.1 = 109.9
            assert packet.payload["price"] == 109.9
            assert packet.payload["rsi"] is not None
            assert packet.payload["sma_20"] is not None
