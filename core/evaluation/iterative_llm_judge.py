from typing import Dict, Any, List
import json
from pydantic import BaseModel, Field
from core.evaluation.system_judge import EvaluationRubric

class LLMJudgeScore(BaseModel):
    criteria_scores: Dict[str, float] = Field(..., description="Scores for each criteria in the rubric")
    overall_score: float = Field(..., description="Weighted average overall score")
    critique: str = Field(..., description="Detailed markdown critique explaining the scores")
    improvement_suggestions: List[str] = Field(..., description="Actionable suggestions for prompt or model improvement")

class IterativeLLMJudge:
    """
    LLM-as-a-Judge for semantic and qualitative review based on a defined Rubric.
    """
    def __init__(self, rubrics: List[EvaluationRubric], llm_plugin: Any = None):
        self.rubrics = rubrics
        self.llm_plugin = llm_plugin

    def evaluate(self, input_prompt: str, output_text: str) -> LLMJudgeScore:
        """
        Evaluates the output_text against the rubrics using the LLM.
        """
        if self.llm_plugin:
            return self._real_evaluate(input_prompt, output_text)
        return self._mock_evaluate(input_prompt, output_text)

    def _mock_evaluate(self, input_prompt: str, output_text: str) -> LLMJudgeScore:
        # Simulated logic for fallback/testing
        scores = {}
        total_weight = 0.0
        weighted_sum = 0.0
        for r in self.rubrics:
            # Simulate a score (e.g. 80% of max_score)
            score = r.max_score * 0.8
            scores[r.criteria] = score
            weighted_sum += (score / r.max_score) * r.weight
            total_weight += r.weight

        overall = (weighted_sum / total_weight) * 100 if total_weight > 0 else 0.0

        return LLMJudgeScore(
            criteria_scores=scores,
            overall_score=overall,
            critique=f"**Mock Critique**\nThe output addressed the prompt but lacked depth in `{self.rubrics[0].criteria}`.",
            improvement_suggestions=["Add more context to the prompt.", "Ask for step-by-step reasoning."]
        )

    def _real_evaluate(self, input_prompt: str, output_text: str) -> LLMJudgeScore:
        rubric_text = "\n".join([f"- **{r.criteria}** (Weight {r.weight}, Max {r.max_score}): {r.description}" for r in self.rubrics])
        prompt = f"""
        You are an expert evaluator. Evaluate the following output based on the provided rubric.

        --- RUBRIC ---
        {rubric_text}

        --- INPUT PROMPT ---
        {input_prompt}

        --- OUTPUT TO EVALUATE ---
        {output_text}
        """
        try:
            score, _ = self.llm_plugin.generate_structured(prompt, LLMJudgeScore)
            return score
        except Exception:
            return self._mock_evaluate(input_prompt, output_text)
