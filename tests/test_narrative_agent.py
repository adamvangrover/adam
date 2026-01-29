import unittest
import asyncio
from core.agents.specialized.narrative_intelligence_agent import NarrativeIntelligenceAgent
from core.utils.narrative_weaver import NarrativeWeaver

class TestNarrativeIntelligence(unittest.TestCase):

    def test_narrative_identification(self):
        config = {"agent_id": "NarrativeIntel_01"}
        agent = NarrativeIntelligenceAgent(config)

        # We need to run the async method
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        narratives = loop.run_until_complete(agent.identify_narratives())
        loop.close()

        self.assertIsInstance(narratives, list)
        self.assertTrue(len(narratives) > 0)

        # Check structure
        first_narrative = narratives[0]
        self.assertIn("theme", first_narrative)
        self.assertIn("headline", first_narrative)
        self.assertIn("sentiment", first_narrative)

    def test_narrative_weaver(self):
        weaver = NarrativeWeaver()

        ctx = {
            "sentiment": "BULLISH",
            "driver": "AI Hype",
            "risk_agent": "ComplianceBot"
        }

        story = weaver.weave(ctx)
        print(f"Woven Story: {story}")

        self.assertIsInstance(story, str)
        self.assertIn("AI Hype", story)
        self.assertIn("ComplianceBot", story)

    def test_weaver_fallback(self):
        weaver = NarrativeWeaver()
        # Empty context should fallback gracefully
        story = weaver.weave({})
        print(f"Fallback Story: {story}")
        self.assertIsInstance(story, str)
        self.assertTrue(len(story) > 10)

if __name__ == '__main__':
    unittest.main()
