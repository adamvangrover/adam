import unittest
import os
from unittest.mock import patch
from core.v30_architecture.python_intelligence.agents.news_bot import NewsBotAgent, NewsChunk, ActionableInsight, Provenance

class TestNewsBotAgentV30(unittest.TestCase):
    def setUp(self):
        os.environ["MOCK_MODE"] = "false"
        self.agent = NewsBotAgent()

    def test_semantic_chunking_macro_tags(self):
        """Test semantic chunking now extracts macro themes and relevance correctly."""
        doc = {
            "title": "Apple lawsuit",
            "content": "AAPL faces massive SEC lawsuit. Also a supply chain shortage.",
            "source": "Bloomberg",
            "doc_type": "article"
        }
        chunks = self.agent._process_document(doc)

        self.assertEqual(len(chunks), 2)

        # Test mapping and tags
        self.assertEqual(chunks[0].symbol, "AAPL")
        self.assertIn("legal_risk", chunks[0].macro_themes)
        self.assertGreater(chunks[0].relevance_score, 0.5) # Symbol mentioned + Theme mentioned

        self.assertIn("supply_chain", chunks[1].macro_themes)

    def test_synthesis_logic(self):
        """Test the _analyze_and_synthesize aggregation logic."""
        prov = Provenance(source_uri="test", content_hash="hash", timestamp=0.0)

        chunks = [
            NewsChunk(
                chunk_id="1", symbol="AAPL", source="test", doc_type="test",
                content="AAPL releases an incredible new product.", original_title="test",
                provenance=prov, macro_themes=["innovation"], relevance_score=0.8
            ),
            NewsChunk(
                chunk_id="2", symbol="AAPL", source="test", doc_type="test",
                content="Another release is coming.", original_title="test",
                provenance=prov, macro_themes=["innovation"], relevance_score=0.5
            )
        ]

        insight = self.agent._analyze_and_synthesize("AAPL", chunks)

        self.assertIsNotNone(insight)
        self.assertEqual(insight.symbol, "AAPL")
        self.assertGreater(insight.sentiment, 0.5) # Positive
        self.assertIn("innovation", insight.macro_themes)
        self.assertIn("bullish momentum", insight.market_impact_estimate)
        self.assertGreater(insight.conviction, 0.3)

    @patch("core.v30_architecture.python_intelligence.agents.news_bot.NewsBotAgent._ingest_data")
    def test_end_to_end_synthesis(self, mock_ingest):
        """Test that the full cycle correctly returns the new synthesized ActionableInsight."""
        mock_ingest.return_value = [
            {"title": "Microsoft under investigation", "content": "MSFT lawsuit pending due to SEC investigation.", "source": "Bloomberg", "doc_type": "news"},
        ]

        insights_data = self.agent.run_cycle()

        self.assertEqual(len(insights_data), 1)

        insight = insights_data[0]
        self.assertEqual(insight["symbol"], "MSFT")
        self.assertLess(insight["sentiment"], -0.5)
        self.assertIn("legal_risk", insight["macro_themes"])
        self.assertIn("bearish", insight["market_impact_estimate"])

if __name__ == '__main__':
    unittest.main()
