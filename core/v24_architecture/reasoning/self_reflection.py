import logging
from typing import Dict, Any, List

logger = logging.getLogger(__name__)

class SelfReflectionAgent:
    """
    Implements the 'Critic' role in the Cycle.
    Evaluates drafts against a 'Constitution'.
    """

    CONSTITUTION = """
    You are the Senior Risk Officer. Review the drafted analysis.

    1. Source Check: Every quantitative claim must be supported by a node in the Subgraph.
    2. Logic Check: If the outlook is 'Bullish', the fundamental metrics must show a positive trend.
    3. Completeness: The report must address: Liquidity, Credit, and Market Risk.

    If the draft passes, output: PASS.
    If the draft fails, output: FAIL: <reason>.
    """

    def __init__(self, llm_client=None):
        self.llm = llm_client

    async def critique(self, draft: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluates the draft.
        """
        logger.info("Self-Reflection Agent: Critiquing Draft...")

        # In production: response = await self.llm.generate(prompt)

        # Mock Logic for V24 Skeleton
        score = 0.85
        feedback = []
        status = "PASS"

        if "Liquidity" not in draft:
            feedback.append("Missing Liquidity Risk Analysis.")
            score -= 0.2
            status = "FAIL"

        if "Conviction" not in draft and "Confidence" not in draft:
             feedback.append("Missing Conviction/Confidence Score.")
             score -= 0.1
             status = "FAIL"

        return {
            "status": status,
            "score": max(0.0, score),
            "feedback": feedback,
            "refined_instructions": f"Please address: {'; '.join(feedback)}" if feedback else ""
        }
