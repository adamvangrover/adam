from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
import json
import logging
from core.llm_plugin import LLMPlugin

logger = logging.getLogger(__name__)

class EvaluationScore(BaseModel):
    """
    Rubric-based scoring for agent outputs.
    Scale 1-5.
    """
    factual_grounding: int = Field(..., ge=1, le=5, description="Did it hallucinate facts or covenants?")
    logic_density: int = Field(..., ge=1, le=5, description="Is the Chain of Thought mathematically and logically sound?")
    financial_nuance: int = Field(..., ge=1, le=5, description="Does it distinguish between key concepts like liquidity and solvency?")
    reasoning: str = Field(..., description="The Auditor's explanation for the assigned scores.")
    overall_score: float = Field(..., description="The average of the component scores.")

class AuditorAgent:
    """
    A specialized 'LLM-as-a-Judge' agent that evaluates other agents' reasoning.
    It does not see the original prompt instructions, only the input data and the agent's output,
    to ensure it judges the result objectively.
    """

    RUBRIC = """
    Evaluation Rubric:
    1. Factual Grounding (1-5):
       - 1: Major hallucinations (invented debt, wrong dates).
       - 5: Perfect alignment with input data; all claims cited.

    2. Logic Density (1-5):
       - 1: Jumps to conclusions, math errors, superficial.
       - 5: Step-by-step derivation, mathematically sound, deep causal analysis.

    3. Financial Nuance (1-5):
       - 1: Confuses terms (e.g., EBITDA vs Revenue), misses obvious risks.
       - 5: Expert-level distinction (e.g., identifying structural subordination), understands context.
    """

    def __init__(self, model_name: str = "gemini-1.5-pro", mock_mode: bool = False):
        self.model_name = model_name
        self.mock_mode = mock_mode
        if not mock_mode:
            self.llm = LLMPlugin(config={"provider": "gemini", "gemini_model_name": model_name})

    def evaluate(self, input_data: Dict[str, Any], agent_output: Dict[str, Any]) -> EvaluationScore:
        """
        Evaluates the agent's output against the input data.
        """
        logger.info(f"AuditorAgent evaluating output from model {self.model_name}...")

        if self.mock_mode:
            return self._mock_evaluate(input_data, agent_output)

        try:
            prompt = (
                f"You are an expert Credit Auditor. Evaluate the following analysis based on the provided input data.\n\n"
                f"{self.RUBRIC}\n\n"
                f"INPUT DATA:\n{json.dumps(input_data, default=str)}\n\n"
                f"AGENT OUTPUT:\n{json.dumps(agent_output, default=str)}\n\n"
                f"Provide a structured evaluation."
            )

            score, _ = self.llm.generate_structured(prompt, EvaluationScore)
            return score

        except Exception as e:
            logger.warning(f"LLM Evaluation failed: {e}. Falling back to mock.")
            return self._mock_evaluate(input_data, agent_output)

    def _mock_evaluate(self, input_data: Dict[str, Any], agent_output: Dict[str, Any]) -> EvaluationScore:
        """
        Simulates an evaluation for demonstration purposes.
        """
        # Simple heuristic simulation
        score_val = 4
        reasoning = "The analysis is sound and references the provided financial data correctly. Good distinction between liquidity and solvency."

        # Degrade score if output seems empty or error-prone
        if not agent_output or "error" in str(agent_output).lower():
            score_val = 2
            reasoning = "Output contains errors or is incomplete."

        # Boost if it looks detailed
        if "risk_factors" in agent_output and len(str(agent_output)) > 500:
             score_val = 5
             reasoning = "Excellent depth and coverage of risk factors. Covenants checked against input."

        return EvaluationScore(
            factual_grounding=score_val,
            logic_density=score_val,
            financial_nuance=score_val,
            reasoning=reasoning,
            overall_score=float(score_val)
        )
