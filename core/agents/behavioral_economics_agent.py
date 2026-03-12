# core/agents/behavioral_economics_agent.py
from core.agents.agent_base import AgentBase
from core.schemas.agent_schema import AgentInput, AgentOutput
from typing import Any, Dict, List, Optional, Union
import logging

try:
    from transformers import pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logging.warning("Transformers library not available. BehavioralEconomicsAgent will use fallback logic.")

logger = logging.getLogger(__name__)

class BehavioralEconomicsAgent(AgentBase):
    """
    Analyzes market data and user interactions for signs of cognitive biases and irrational behavior.
    """

    def __init__(self, config: Dict[str, Any], **kwargs):
        super().__init__(config, **kwargs)

        # Expanded Bias Patterns
        self.bias_patterns = self.config.get("bias_patterns", {
            "market": {
                "Herd Mentality": ["everyone is buying", "FOMO", "can't miss out", "moon"],
                "Panic Selling": ["capitulation", "dumping", "get out now", "bloodbath"],
                "Confirmation Bias": ["as I predicted", "told you so", "ignoring the FUD"],
                "Anchoring": ["still down from ATH", "waiting for breakeven", "bounce back to 100"]
            },
            "user": {
                "Overconfidence": ["guaranteed", "100% sure", "can't lose"],
                "Loss Aversion": ["hold until green", "not a loss until you sell"],
                "Recency Bias": ["it's been going up all week", "trend will continue forever"]
            }
        })

        logger.info(f"BehavioralEconomicsAgent initialized with {len(self.bias_patterns.get('market', {})) + len(self.bias_patterns.get('user', {}))} bias patterns.")

        if TRANSFORMERS_AVAILABLE:
            try:
                # Limit model instantiation if testing or low resource
                self.sentiment_analyzer = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english", truncation=True, max_length=512)
            except Exception as e:
                logger.error(f"Failed to load sentiment model: {e}")
                self.sentiment_analyzer = None
        else:
            self.sentiment_analyzer = None

    async def execute(self, input_data: Union[str, AgentInput, Dict[str, Any]] = None, **kwargs) -> Union[Dict[str, Any], AgentOutput]:
        """
        Executes the behavioral analysis.
        Supports standard AgentInput.
        """
        logger.info("Executing behavioral economics analysis...")

        is_standard_mode = False
        analysis_content = ""
        user_query_history = []
        query = "Behavioral Analysis"

        if input_data is not None:
            if isinstance(input_data, AgentInput):
                query = input_data.query
                analysis_content = input_data.context.get("analysis_content", query)
                user_query_history = input_data.context.get("user_query_history", [])
                is_standard_mode = True
            elif isinstance(input_data, dict):
                analysis_content = input_data.get("analysis_content", "")
                user_query_history = input_data.get("user_query_history", [])
                kwargs.update(input_data)
            elif isinstance(input_data, str):
                analysis_content = input_data

        # Fallbacks to kwargs
        if not analysis_content:
            analysis_content = kwargs.get("analysis_content", "")
        if not user_query_history:
            user_query_history = kwargs.get("user_query_history", [])

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

        logger.info(f"Behavioral economics analysis complete. Found {len(results['market_biases'])} market biases and {len(results['user_biases'])} user biases.")

        if is_standard_mode:
            answer = f"Behavioral Economics Analysis for '{query}':\n\n"
            answer += results["insights"]

            return AgentOutput(
                answer=answer,
                sources=["Sentiment Analysis", "Heuristic Bias Patterns"],
                confidence=0.85 if self.sentiment_analyzer else 0.5,
                metadata=results
            )

        return results

    def _identify_market_biases(self, content: str) -> List[Dict[str, str]]:
        """
        Identifies common market biases in a given text using sentiment analysis and patterns.
        """
        identified_biases = []

        if self.sentiment_analyzer:
            try:
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
            except Exception as e:
                logger.warning(f"Sentiment analysis failed during execution: {e}")

        # Pattern matching
        for bias, patterns in self.bias_patterns.get("market", {}).items():
            for pattern in patterns:
                if pattern.lower() in content.lower():
                    identified_biases.append({
                        "bias": bias,
                        "evidence": f"Found pattern '{pattern}'"
                    })

        # Deduplicate
        unique_biases = {v['bias']:v for v in identified_biases}.values()
        return list(unique_biases)

    def _identify_user_biases(self, queries: List[str]) -> List[Dict[str, str]]:
        """
        Identifies cognitive biases in user query patterns.
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

        unique_biases = {v['bias']:v for v in identified_biases}.values()
        return list(unique_biases)

    def _generate_insights(self, analysis_results: Dict[str, Any]) -> str:
        """
        Generates a summary of insights based on the identified biases.
        """
        if not analysis_results["market_biases"] and not analysis_results["user_biases"]:
            return "No significant cognitive biases detected in the provided text or history."

        insight_text = "Behavioral Analysis Insights:\n"
        if analysis_results["market_biases"]:
            insight_text += "- Market sentiment exhibits signs of: "
            insight_text += ", ".join(list(set([b["bias"] for b in analysis_results["market_biases"]]))) + ".\n"
        if analysis_results["user_biases"]:
            insight_text += "- User interaction patterns suggest potential for: "
            insight_text += ", ".join(list(set([b["bias"] for b in analysis_results["user_biases"]]))) + ".\n"

        return insight_text
