import unittest
from unittest.mock import MagicMock, patch
import asyncio
import sys
import os
import types
import numpy as np

# --- MOCKING DEPENDENCIES FOR ISOLATION ---
# Mock core.agents.agent_base
mock_agent_base_module = types.ModuleType("core.agents.agent_base")
class MockAgentBase:
    def __init__(self, config, **kwargs):
        self.config = config
    async def execute(self, *args, **kwargs):
        pass
mock_agent_base_module.AgentBase = MockAgentBase
sys.modules["core.agents.agent_base"] = mock_agent_base_module

# Mock core.schemas.agent_schema
mock_schema_module = types.ModuleType("core.schemas.agent_schema")
class AgentInput:
    def __init__(self, query, context=None):
        self.query = query
        self.context = context or {}
class AgentOutput:
    def __init__(self, answer, sources, confidence, metadata):
        self.answer = answer
        self.sources = sources
        self.confidence = confidence
        self.metadata = metadata
mock_schema_module.AgentInput = AgentInput
mock_schema_module.AgentOutput = AgentOutput
sys.modules["core.schemas.agent_schema"] = mock_schema_module

# Mock yfinance
mock_yf = MagicMock()
mock_ticker_obj = MagicMock()
mock_ticker_obj.info = {
    "longName": "Mock Apple",
    "sector": "Information Technology",
    "industry": "Technology Hardware, Storage & Peripherals",
    "longBusinessSummary": "Designs smartphones, personal computers, tablets, wearables, and accessories."
}
# yf.Ticker is a class, so calling it returns an instance
mock_yf.Ticker.return_value = mock_ticker_obj
sys.modules["yfinance"] = mock_yf

# Mock sklearn
mock_sklearn_text = types.ModuleType("sklearn.feature_extraction.text")
mock_vectorizer = MagicMock()
mock_vectorizer.fit_transform.return_value = [[1, 0], [0, 1]]
mock_tfidf_cls = MagicMock(return_value=mock_vectorizer)
mock_sklearn_text.TfidfVectorizer = mock_tfidf_cls
sys.modules["sklearn.feature_extraction.text"] = mock_sklearn_text

mock_sklearn_pairwise = types.ModuleType("sklearn.metrics.pairwise")
# Return numpy array so .flatten() works
mock_sklearn_pairwise.cosine_similarity = MagicMock(return_value=np.array([[1.0, 0.8, 0.9]]))
sys.modules["sklearn.metrics.pairwise"] = mock_sklearn_pairwise
sys.modules["sklearn"] = MagicMock()

# Now import the agent under test
import importlib.util
agent_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../core/agents/peer_set_agent.py'))
spec = importlib.util.spec_from_file_location("peer_set_agent_test", agent_path)
psa_module = importlib.util.module_from_spec(spec)
sys.modules["peer_set_agent_test"] = psa_module
spec.loader.exec_module(psa_module)

PeerSetAgent = psa_module.PeerSetAgent


class TestPeerSetAgent(unittest.TestCase):
    def setUp(self):
        self.agent = PeerSetAgent(config={"agent_id": "test_peer_set"})

    def test_find_peers_real_data_mocked(self):
        # We need to run the async method
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        # Calling find_peers("AAPL", "semantic")
        # Logic:
        # 1. fetch_real_data("AAPL") -> returns info
        # 2. candidates -> filtered by sector (Information Technology)
        # 3. corpus built -> 1 (target) + N (candidates)
        # 4. cosine_similarity -> returns [1.0, 0.8, 0.9]
        # 5. skips first (self), loops over 0.8, 0.9
        # 6. appends candidates if score > 0.1

        # We need to make sure filtered_candidates is not empty.
        # Mock DB has AAPL, DELL, HPQ in Info Tech.
        # So filtered_candidates should be [AAPL, DELL, HPQ, MSFT, ORCL, ADBE, CRM, NVDA, AMD, INTC]
        # (Assuming these are in the updated Mock DB in peer_set_agent.py)

        # The mocked cosine similarity returns shape (1, 3).
        # flatten -> [1.0, 0.8, 0.9]
        # candidates loop:
        # i=0, score=0.8. Candidate 0 (after self).
        # i=1, score=0.9. Candidate 1.

        # So we should get 2 peers.

        peers = loop.run_until_complete(self.agent.find_peers("AAPL", "semantic"))
        loop.close()

        self.assertTrue(len(peers) >= 1)
        self.assertIn("similarity", peers[0])
        # Verify scores match our mock (checking 0.8 or 0.9)
        scores = [p['similarity'] for p in peers]
        self.assertTrue(0.8 in scores or 0.9 in scores)

    def test_fallback_mock_db(self):
        # Patch fetch_real_data on the instance to return None, forcing fallback
        with patch.object(self.agent, 'fetch_real_data', return_value=None):
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            peers = loop.run_until_complete(self.agent.find_peers("AAPL", "gics"))
            loop.close()

            # Should use Mock DB logic
            tickers = [p['ticker'] for p in peers]
            # AAPL (45202010) matches DELL, HPQ in Mock DB
            self.assertIn("DELL", tickers)
            self.assertIn("HPQ", tickers)

if __name__ == '__main__':
    unittest.main()
