import json
import os
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional
from textblob import TextBlob

from core.swarms.memory_matrix import MemoryMatrix
from core.utils.narrative_weaver import NarrativeWeaver

class ConsensusEngineV2:
    """
    Protocol: ADAM-V-NEXT
    Verified by Jules.

    The ConsensusEngineV2 analyzes the persistent SwarmMemory (MemoryMatrix) to generate
    high-level 'Strategic Directives' (Bullish/Bearish/Neutral) and actionable plans.
    It synthesizes a 'House View' that drives the strategic command dashboard.
    """

    def __init__(self, memory_matrix: Optional[MemoryMatrix] = None):
        self.memory = memory_matrix if memory_matrix else MemoryMatrix()
        self.weaver = NarrativeWeaver()
        self.logger = logging.getLogger(__name__)

    def analyze_memory(self) -> Dict[str, Any]:
        """
        Analyzes the MemoryMatrix to determine the current strategic stance.
        """
        nodes = self.memory.memory_store.get("nodes", {})

        if not nodes:
            return self._generate_fallback_analysis()

        total_sentiment = 0.0
        insight_count = 0
        topics = []

        # Analyze insights across all nodes
        for key, node in nodes.items():
            topic = node.get("topic", "Unknown")
            topics.append(topic)

            # Simple sentiment analysis on insights
            for insight in node.get("insights", []):
                content = insight.get("content", "")
                confidence = insight.get("confidence", 0.5)

                try:
                    blob = TextBlob(content)
                    sentiment = blob.sentiment.polarity
                    # Weight by confidence
                    total_sentiment += sentiment * confidence
                    insight_count += 1
                except Exception as e:
                    self.logger.warning(f"Sentiment analysis failed for insight: {e}")

        # Normalize sentiment (-1.0 to 1.0)
        avg_sentiment = total_sentiment / insight_count if insight_count > 0 else 0.0

        # Determine Stance
        stance = "NEUTRAL"
        if avg_sentiment > 0.15:
            stance = "BULLISH"
        elif avg_sentiment < -0.15:
            stance = "BEARISH"

        # Generate Narrative
        context = {
            "sentiment": stance,
            "driver": topics[0] if topics else "Market Structure",
            "risk_factor": "Volatility" if avg_sentiment < 0 else "Overvaluation",
            "sector": "Broad Market"
        }
        narrative = self.weaver.weave(context)

        return {
            "house_view": stance,
            "sentiment_score": round(avg_sentiment, 2),
            "narrative": narrative,
            "active_topics": topics[:5],
            "insight_count": insight_count,
            "timestamp": datetime.utcnow().isoformat()
        }

    def _generate_fallback_analysis(self) -> Dict[str, Any]:
        """Generates a default analysis if memory is empty."""
        return {
            "house_view": "NEUTRAL",
            "sentiment_score": 0.0,
            "narrative": "System initializing. Awaiting swarm consensus data to form a strategic view.",
            "active_topics": [],
            "insight_count": 0,
            "timestamp": datetime.utcnow().isoformat()
        }

    def generate_strategic_plan(self, analysis: Dict[str, Any]) -> List[Dict[str, str]]:
        """
        Generates actionable plans based on the analysis.
        """
        stance = analysis.get("house_view", "NEUTRAL")
        plans = []

        if stance == "BULLISH":
            plans = [
                {"action": "INCREASE_EXPOSURE", "target": "Growth Sectors", "rationale": "Positive sentiment momentum detected."},
                {"action": "LEVERAGE_OPTIMIZATION", "target": "Portfolio", "rationale": "Market conditions favor risk-on positioning."}
            ]
        elif stance == "BEARISH":
            plans = [
                {"action": "HEDGE_DOWNSIDE", "target": "SPX Puts", "rationale": "Negative sentiment trend requires protection."},
                {"action": "LIQUIDITY_PRESERVATION", "target": "Cash Reserves", "rationale": "Market uncertainty dictates caution."}
            ]
        else:
            plans = [
                {"action": "MAINTAIN_BALANCE", "target": "Current Allocation", "rationale": "No clear directional signal."},
                {"action": "VOLATILITY_HARVESTING", "target": "Options Writing", "rationale": "Range-bound market expected."}
            ]

        return plans

    def generate_report(self, output_path: str = "showcase/data/strategic_command.json") -> str:
        """
        Generates the full strategic command report and saves it to JSON.
        """
        analysis = self.analyze_memory()
        plans = self.generate_strategic_plan(analysis)

        report = {
            "meta": {
                "engine": "ConsensusEngineV2",
                "version": "2.0",
                "generated_at": datetime.utcnow().isoformat()
            },
            "strategic_directives": {
                "house_view": analysis["house_view"],
                "score": analysis["sentiment_score"],
                "narrative": analysis["narrative"]
            },
            "insights": {
                "total_analyzed": analysis["insight_count"],
                "active_topics": analysis["active_topics"]
            },
            "actionable_plans": plans
        }

        # Ensure directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)

        self.logger.info(f"Strategic Command Report generated at {output_path}")
        return output_path
