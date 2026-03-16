# core/agents/behavioral_economics_agent.py
import logging
from typing import Any, Dict, List

from transformers import pipeline

from core.agents.agent_base import AgentBase
from core.schemas.agent_schema import AgentInput, AgentOutput


class BehavioralEconomicsAgent(AgentBase):
    """
    Analyzes market data and user interactions for signs of cognitive biases and irrational behavior.
    """

    def __init__(self, config: Dict[str, Any], **kwargs):
        super().__init__(config, **kwargs)
        # Initialization specific to this agent, e.g., loading bias models or patterns
        self.bias_patterns = self.config.get("bias_patterns", {})
        logging.info(f"BehavioralEconomicsAgent initialized with {len(self.bias_patterns)} bias patterns.")
        # Load a pre-trained sentiment analysis model
        self.sentiment_analyzer = pipeline(
            "sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

    async def execute(self, input_data: AgentInput) -> AgentOutput:
        """
        Executes the behavioral analysis.

        Args:
            input_data (AgentInput): Input data.

        Returns:
            AgentOutput: A dictionary containing identified biases and insights.
        """
        logging.info("Executing behavioral economics analysis...")
        analysis_content = input_data.query
        user_query_history = input_data.context.get("user_query_history", [])
        results = {
            "market_biases": [],
            "user_biases": [],
            "insights": ""
        }

        # 1. Analyze market content for biases
        if analysis_content:
            results["market_biases"] = self._identify_market_biases(analysis_content)

        # 2. Analyze user query history for biases
        if user_query_history:
            results["user_biases"] = self._identify_user_biases(user_query_history)

        # 3. Generate insights based on findings
        results["insights"] = self._generate_insights(results)

        logging.info(
            f"Behavioral economics analysis complete. Found {len(results['market_biases'])} market biases and {len(results['user_biases'])} user biases.")

        return AgentOutput(
            answer=results["insights"],
            sources=[],
            confidence=1.0,  # Default neutral baseline
            metadata={
                "market_biases": results["market_biases"],
                "user_biases": results["user_biases"]
            }
        )

    def _identify_market_biases(self, content: str) -> List[Dict[str, str]]:
        """
        Identifies common market biases in a given text using sentiment analysis.
        """
        identified_biases = []
        # Use the sentiment analyzer to get the sentiment of the text
        sentiment_result = self.sentiment_analyzer(content)
        sentiment_label = sentiment_result[0]['label']
        sentiment_score = sentiment_result[0]['score']

        if sentiment_label == 'NEGATIVE' and sentiment_score > 0.8:
            identified_biases.append({
                "bias": "Fear/Panic",
                "evidence": f"High negative sentiment (score: {sentiment_score:.2f})"
            })
        elif sentiment_label == 'POSITIVE' and sentiment_score > 0.8:
            identified_biases.append({
                "bias": "Greed/Irrational Exuberance",
                "evidence": f"High positive sentiment (score: {sentiment_score:.2f})"
            })

        # Also run the simple pattern matching for other biases
        for bias, patterns in self.bias_patterns.get("market", {}).items():
            for pattern in patterns:
                if pattern.lower() in content.lower():
                    identified_biases.append({
                        "bias": bias,
                        "evidence": f"Found pattern '{pattern}'"
                    })
        return identified_biases

    def _identify_user_biases(self, queries: List[str]) -> List[Dict[str, str]]:
        """
        Identifies cognitive biases in user query patterns.
        Placeholder implementation.
        """
        identified_biases = []
        full_query_text = " ".join(queries).lower()
        for bias, patterns in self.bias_patterns.get("user", {}).items():
            for pattern in patterns:
                if pattern.lower() in full_query_text:
                    identified_biases.append({
                        "bias": bias,
                        "evidence": f"Found pattern '{pattern}' in query history."
                    })
        return identified_biases

    def _generate_insights(self, analysis_results: Dict[str, Any]) -> str:
        """
        Generates a summary of insights based on the identified biases.
        Placeholder implementation.
        """
        if not analysis_results["market_biases"] and not analysis_results["user_biases"]:
            return "No significant cognitive biases detected."

        insight_text = "Behavioral Analysis Insights:\n"
        if analysis_results["market_biases"]:
            insight_text += "- Market sentiment may be influenced by: "
            insight_text += ", ".join(list(set([b["bias"] for b in analysis_results["market_biases"]]))) + ".\n"
        if analysis_results["user_biases"]:
            insight_text += "- User interaction patterns suggest potential for: "
            insight_text += ", ".join(list(set([b["bias"] for b in analysis_results["user_biases"]]))) + ".\n"

        return insight_text
