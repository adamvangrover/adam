import json
import os
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional
from textblob import TextBlob
import statistics
import hashlib

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
        # Expanded risk keywords for scanning
        self.risk_keywords = [
            "insolvency", "crash", "bubble", "default", "liquidity crisis",
            "regulatory crackdown", "war", "pandemic", "fraud", "recession",
            "hyperinflation", "depeg", "systemic failure"
        ]

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

        # Accumulate all insights for conflict detection
        topic_sentiments: Dict[str, List[float]] = {}

        # Analyze insights across all nodes
        for key, node in nodes.items():
            topic = node.get("topic", "Unknown")
            if topic not in topics:
                topics.append(topic)

            if topic not in topic_sentiments:
                topic_sentiments[topic] = []

            # Sentiment analysis on insights
            for insight in node.get("insights", []):
                content = insight.get("content", "")
                confidence = insight.get("confidence", 0.5)

                try:
                    blob = TextBlob(content)
                    sentiment = blob.sentiment.polarity
                    # Weight by confidence for overall score
                    total_sentiment += sentiment * confidence
                    insight_count += 1

                    # Store raw sentiment for variance analysis
                    topic_sentiments[topic].append(sentiment)
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

        # Specialized Analysis
        conflicts = self.identify_conflicting_narratives(topic_sentiments)
        risks = self.extract_critical_risks(nodes)
        system_health = self.assess_system_health(nodes)

        # Generate Narrative
        # Priority: Critical Risk > Conflict > Stance
        driver = topics[0] if topics else "Market Structure"
        risk_factor = risks[0]["term"] if risks else ("Volatility" if avg_sentiment < 0 else "Overvaluation")

        context = {
            "sentiment": stance,
            "driver": driver,
            "risk_factor": risk_factor,
            "sector": "Broad Market", # TODO: Infer from topics
            "conflicts": conflicts,
            "risks": risks
        }

        # Weave narrative
        narrative = self.weaver.weave(context)

        return {
            "house_view": stance,
            "sentiment_score": round(avg_sentiment, 2),
            "narrative": narrative,
            "active_topics": topics[:10], # Expanded limit
            "insight_count": insight_count,
            "conflicts": conflicts,
            "risks": risks,
            "system_health": system_health,
            "timestamp": datetime.utcnow().isoformat()
        }

    def identify_conflicting_narratives(self, topic_sentiments: Dict[str, List[float]]) -> List[Dict[str, Any]]:
        """
        Identifies topics where agent consensus is fractured (high variance in sentiment).
        """
        conflicts = []
        for topic, sentiments in topic_sentiments.items():
            if len(sentiments) < 3:
                continue # Need minimum sample size

            variance = statistics.variance(sentiments) if len(sentiments) > 1 else 0.0

            # Threshold for conflict: Variance > 0.2 (indicates mix of pos/neg)
            if variance > 0.2:
                conflicts.append({
                    "topic": topic,
                    "variance": round(variance, 2),
                    "status": "CONTESTED",
                    "description": "High divergence in agent sentiment."
                })
        return conflicts

    def extract_critical_risks(self, nodes: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Scans all insights for specific high-impact risk keywords.
        """
        identified_risks = []
        seen_risks = set()

        for key, node in nodes.items():
            topic = node.get("topic", "")
            for insight in node.get("insights", []):
                content = insight.get("content", "").lower()
                confidence = insight.get("confidence", 0.0)

                # Only care about high confidence warnings
                if confidence < 0.6:
                    continue

                for keyword in self.risk_keywords:
                    if keyword in content and keyword not in seen_risks:
                        identified_risks.append({
                            "term": keyword.upper(),
                            "source_topic": topic,
                            "severity": "CRITICAL" if confidence > 0.8 else "ELEVATED"
                        })
                        seen_risks.add(keyword)

        return identified_risks

    def assess_system_health(self, nodes: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extracts system health insights published by RepoKnowledgeAgent.
        """
        health_status = {
            "status": "UNKNOWN",
            "agent_count": 0,
            "core_modules": 0,
            "compliance": "Unknown"
        }

        # Look for "System Architecture" and "Compliance" topics
        # We need to hash the topic to find the node key

        def get_topic_node(topic_name):
            key = hashlib.sha256(topic_name.encode()).hexdigest()
            return nodes.get(key)

        arch_node = get_topic_node("System Architecture")
        if arch_node:
            for insight in arch_node.get("insights", []):
                content = insight.get("content", "")
                if "agent modules" in content:
                    try:
                        # Extract number: "Detected X agent modules..."
                        health_status["agent_count"] = int([s for s in content.split() if s.isdigit()][0])
                    except: pass
                if "Core system footprint" in content:
                    try:
                        health_status["core_modules"] = int([s for s in content.split() if s.isdigit()][0])
                    except: pass

        comp_node = get_topic_node("Compliance")
        if comp_node:
            # Check latest insight
            insights = comp_node.get("insights", [])
            if insights:
                latest = insights[-1].get("content", "")
                if "CRITICAL" in latest:
                    health_status["compliance"] = "CRITICAL"
                elif "active" in latest.lower():
                    health_status["compliance"] = "ACTIVE"

        if health_status["agent_count"] > 0:
            health_status["status"] = "OPERATIONAL"

        return health_status

    def _generate_fallback_analysis(self) -> Dict[str, Any]:
        """Generates a default analysis if memory is empty."""
        return {
            "house_view": "NEUTRAL",
            "sentiment_score": 0.0,
            "narrative": "System initializing. Awaiting swarm consensus data to form a strategic view.",
            "active_topics": [],
            "insight_count": 0,
            "conflicts": [],
            "risks": [],
            "system_health": {},
            "timestamp": datetime.utcnow().isoformat()
        }

    def generate_strategic_plan(self, analysis: Dict[str, Any]) -> List[Dict[str, str]]:
        """
        Generates actionable plans based on the analysis, incorporating risks and conflicts.
        """
        stance = analysis.get("house_view", "NEUTRAL")
        risks = analysis.get("risks", [])
        conflicts = analysis.get("conflicts", [])

        plans = []

        # 1. Address Critical Risks First
        for risk in risks:
            plans.append({
                "action": f"MITIGATE_{risk['term'].replace(' ', '_')}",
                "target": risk['source_topic'],
                "rationale": f"Detected {risk['severity']} risk signal in {risk['source_topic']}."
            })
            if len(plans) >= 2: break # Limit risk actions

        # 2. Address Conflicts
        for conflict in conflicts:
             plans.append({
                "action": "ARBITRAGE_VOLATILITY",
                "target": conflict['topic'],
                "rationale": f"High sentiment variance ({conflict['variance']}) indicates pricing inefficiency."
            })

        # 3. Fill with Stance-based Plans
        if stance == "BULLISH":
            plans.append({"action": "INCREASE_EXPOSURE", "target": "Growth Sectors", "rationale": "Positive sentiment momentum detected."})
            if len(plans) < 4:
                plans.append({"action": "LEVERAGE_OPTIMIZATION", "target": "Portfolio", "rationale": "Market conditions favor risk-on positioning."})
        elif stance == "BEARISH":
            plans.append({"action": "HEDGE_DOWNSIDE", "target": "SPX Puts", "rationale": "Negative sentiment trend requires protection."})
            if len(plans) < 4:
                plans.append({"action": "LIQUIDITY_PRESERVATION", "target": "Cash Reserves", "rationale": "Market uncertainty dictates caution."})
        else: # NEUTRAL
            plans.append({"action": "MAINTAIN_BALANCE", "target": "Current Allocation", "rationale": "No clear directional signal."})
            if len(plans) < 4:
                plans.append({"action": "VOLATILITY_HARVESTING", "target": "Options Writing", "rationale": "Range-bound market expected."})

        return plans[:5] # Limit total plans

    def generate_report(self, output_path: str = "showcase/data/strategic_command.json") -> str:
        """
        Generates the full strategic command report and saves it to JSON.
        """
        analysis = self.analyze_memory()
        plans = self.generate_strategic_plan(analysis)

        report = {
            "meta": {
                "engine": "ConsensusEngineV2",
                "version": "2.2", # Bumped version
                "generated_at": datetime.utcnow().isoformat()
            },
            "strategic_directives": {
                "house_view": analysis["house_view"],
                "score": analysis["sentiment_score"],
                "narrative": analysis["narrative"]
            },
            "insights": {
                "total_analyzed": analysis["insight_count"],
                "active_topics": analysis["active_topics"],
                "conflicts": analysis["conflicts"],
                "risks": analysis["risks"],
                "system_health": analysis["system_health"]
            },
            "actionable_plans": plans
        }

        # Ensure directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)

        self.logger.info(f"Strategic Command Report generated at {output_path}")
        return output_path
