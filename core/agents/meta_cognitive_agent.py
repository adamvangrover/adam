from typing import Any, Dict, List, Optional
import logging
import re
from core.agents.agent_base import AgentBase, AgentInput, AgentOutput
from core.agents.system_health_agent import SystemHealthAgent

logger = logging.getLogger(__name__)

class MetaCognitiveAgent(AgentBase):
    """
    The Meta-Cognitive Agent monitors the reasoning and outputs of other agents
    to ensure logical consistency, coherence, and alignment with core principles.
    It acts as a "Logical Consistency Guardian".
    """

    def __init__(self, config: Dict[str, Any] = None, kernel: Optional[Any] = None):
        if config is None:
            config = {"name": "MetaCognitiveAgent"}
            
        super().__init__(config, kernel=kernel)
        self.agent_performance = {}
        self.system_health_agent = SystemHealthAgent(agent_name="meta_cognitive_health_monitor")

        # Load specific configurations for fallacy detection
        self.logical_fallacies = self.config.get("logical_fallacies", {
            "appeal_to_authority": [r"\b(expert|authority|source) says\b", r"\b(studies|research) shows?\b"],
            "hasty_generalization": [r"\b(always|never|everyone|no one) (is|are|does|do)\b"],
            "false_cause": [r"\b(happened after|coincidence|caused by)\b"]
        })
        self.positive_keywords = self.config.get("positive_keywords", [
            r"\b(strong buy|outperform|bullish|upward trend|high confidence|positive outlook)\b"
        ])
        self.negative_keywords = self.config.get("negative_keywords", [
            r"\b(high risk|underperform|bearish|downward trend|low confidence|negative outlook|risky)\b"
        ])

    async def execute(self, input_data: AgentInput) -> AgentOutput:
        """
        Analyzes content for logical consistency, fallacies, and contradictions.

        Args:
            input_data (AgentInput): Strongly typed Pydantic input. The text to analyze should be in 'query'.

        Returns:
            AgentOutput: Structured analysis report including a coherence score and detected issues.
        """
        content_to_analyze = input_data.query
        agent_name = input_data.context.get("agent_name", "UnknownAgent")

        logger.info(f"MetaCognitiveAgent analyzing output from {agent_name}...")

        if not content_to_analyze:
             return AgentOutput(
                 answer="No content provided for analysis.",
                 sources=[],
                 confidence=0.0,
                 metadata={"status": "error"}
             )

        # 1. Detect Logical Fallacies
        detected_fallacies = self._detect_logical_fallacies(content_to_analyze)

        # 2. Check for Contradictions (Naive approach: mixed extreme sentiment without nuance)
        contradictions = self._detect_contradictions(content_to_analyze)

        # 3. Calculate Coherence Score (10.0 scale)
        coherence_score = 10.0

        # Penalize for fallacies
        coherence_score -= (len(detected_fallacies) * 1.5)

        # Penalize for severe contradictions
        if contradictions:
            coherence_score -= 3.0

        coherence_score = max(0.0, min(10.0, coherence_score))

        # Record performance metric
        self.record_performance(agent_name, "coherence_score", coherence_score)

        # Determine status
        verification_status = "PASS"
        if coherence_score < 7.0:
            verification_status = "NEEDS_REVISION"

        # Check system health
        health_metrics = {}
        try:
            health_input = AgentInput(query="health check", context={})
            health_output = await self.system_health_agent.execute(health_input)
            health_metrics = health_output.metadata if hasattr(health_output, 'metadata') else health_output
        except Exception as e:
            logger.warning(f"Failed to execute system_health_agent: {e}")
            health_metrics = {"status": "unknown", "error": str(e)}

        
        final_answer = f"Coherence Score: {coherence_score}/10. Status: {verification_status}."
        if detected_fallacies:
            final_answer += f" {len(detected_fallacies)} Fallacies Detected."
        if contradictions:
            final_answer += f" Contradictions Present."

        return AgentOutput(
            answer=final_answer,
            sources=["LogicalConsistencyGuardian", "RegexFallacyDetector"],
            confidence=coherence_score / 10.0,
            metadata={
                "agent_analyzed": agent_name,
                "original_content_snippet": content_to_analyze[:150] + "...",
                "detected_fallacies": detected_fallacies,
                "contradictions_detected": contradictions,
                "coherence_score": coherence_score,
                "verification_status": verification_status,
                "system_health": health_metrics
            }
        )

    def _detect_logical_fallacies(self, text: str) -> List[Dict[str, str]]:
        """
        Scans text for configured logical fallacies using regex.
        """
        found = []
        text_lower = text.lower()

        for fallacy_name, patterns in self.logical_fallacies.items():
            for pattern in patterns:
                matches = re.finditer(pattern, text_lower)
                for match in matches:
                    found.append({
                        "fallacy": fallacy_name,
                        "evidence": match.group(0)
                    })
        return found

    def _detect_contradictions(self, text: str) -> List[str]:
        """
        Checks if the text contains both highly positive and highly negative assertions
        in close proximity without obvious nuance (e.g., "However", "Although").
        """
        contradictions = []
        text_lower = text.lower()

        has_pos = any(re.search(pat, text_lower) for pat in self.positive_keywords)
        has_neg = any(re.search(pat, text_lower) for pat in self.negative_keywords)

        # If both extreme positive and extreme negative exist
        if has_pos and has_neg:
            # Check for nuance words
            nuance_words = ["however", "although", "despite", "but", "while", "on the other hand"]
            has_nuance = any(word in text_lower for word in nuance_words)

            if not has_nuance:
                contradictions.append(
                    "Detected conflicting extreme assertions (e.g., 'strong buy' and 'high risk') "
                    "without clear contextual nuance."
                )

        return contradictions

    def record_performance(self, agent_name: str, metric: str, value: float):
        """
        Records a performance metric for an agent.
        """
        if agent_name not in self.agent_performance:
            self.agent_performance[agent_name] = {}
        self.agent_performance[agent_name][metric] = value

        # Optional: Log trend if accumulating history
        logger.debug(f"Recorded {metric}={value} for {agent_name}")

    def ping_health(self) -> str:
        try:
            return self.system_health_agent.ping()
        except AttributeError:
             return "PONG: SystemHealthAgent lacks ping method."
