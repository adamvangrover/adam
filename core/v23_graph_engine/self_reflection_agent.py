import logging
from core.schemas.v23_5_schema import CritiqueResult
from core.v23_graph_engine.states import GraphState

logger = logging.getLogger(__name__)

class SelfReflectionAgent:
    """
    The 'Senior Editor' agent.
    Critiques drafts against a 'Constitution' of financial reporting standards.
    """

    def __init__(self, llm_client=None):
        self.llm_client = llm_client
        # The Constitution Prompt
        self.system_prompt = """
        You are the Senior Risk Officer and Chief Editor.
        Review the drafted analysis below.

        Rules:
        1. Source Check: Every quantitative claim must be supported by data.
        2. Logic Check: Conclusions must follow from the premises.
        3. Completeness: Address Liquidity, Credit, and Market Risk.
        4. Tone: Professional, objective, institutional grade.

        Output valid JSON conforming to the CritiqueResult schema.
        """

    def critique(self, state: GraphState) -> CritiqueResult:
        """
        Evaluates the current draft in the state.
        """
        draft = state.get("draft", "")
        logger.info("SelfReflectionAgent: Critiquing draft...")

        if not draft:
            return CritiqueResult(
                passed=False,
                feedback="Draft is empty.",
                score=0.0,
                missing_data=["All Content"]
            )

        # In a real impl, we call the LLM here.
        # response = self.llm_client.chat(messages=[...])

        # Mock Logic for POC:
        # Pass if draft is long enough and contains key sections
        if len(draft) > 100 and "Risk Analysis" in draft:
             return CritiqueResult(
                passed=True,
                feedback="Draft meets minimum standards.",
                score=0.9,
                missing_data=[]
            )
        else:
             return CritiqueResult(
                passed=False,
                feedback="Draft is too short or missing 'Risk Analysis' section.",
                score=0.4,
                missing_data=["Risk Analysis Section", "Quantitative Backing"]
            )
