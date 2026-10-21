"""
Clean-Room LLM-as-a-Judge Evaluator & Multi-Criteria Grading Engine.

Provides clean-room benchmark grading, multi-criteria rubric evaluation,
offline replay capabilities, and deterministic fallback heuristics for System 1/System 2
agent evaluation in ADAM AFOS v30.1.
"""

from dataclasses import dataclass, field
import logging
import re
from typing import Any, Callable, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


@dataclass
class DimensionScore:
    """Individual evaluation dimension score."""

    name: str
    score: float  # Normalized 0.0 to 1.0
    weight: float = 1.0
    reasoning: str = ""


@dataclass
class JudgeResult:
    """Comprehensive structured judge evaluation result."""

    overall_score: float
    passed: bool
    dimensions: List[DimensionScore] = field(default_factory=list)
    verdict: str = "PASS"
    feedback: str = ""
    raw_response: Optional[str] = None


class CleanRoomLLMJudge:
    """
    Clean-Room LLM-as-a-Judge Evaluator.

    Supports offline replay evaluation, multi-criteria rubrics, configurable weights,
    and fallback heuristic evaluations for zero external API dependencies.
    """

    DEFAULT_WEIGHTS: Dict[str, float] = {
        "correctness": 0.4,
        "faithfulness": 0.3,
        "reasoning_coherence": 0.2,
        "provenance_citation": 0.1,
    }

    def __init__(
        self,
        llm_callable: Optional[Callable[[str], str]] = None,
        replay_store: Optional[Dict[str, Dict[str, Any]]] = None,
        weights: Optional[Dict[str, float]] = None,
        pass_threshold: float = 0.70,
    ) -> None:
        self.llm_callable = llm_callable
        self.replay_store = replay_store or {}
        self.weights = weights or self.DEFAULT_WEIGHTS.copy()
        self.pass_threshold = pass_threshold

    def evaluate(
        self,
        question: str,
        agent_answer: str,
        golden_answer: str,
        context: Optional[str] = None,
        replay_key: Optional[str] = None,
    ) -> JudgeResult:
        """
        Evaluate an agent's answer against a golden answer using clean-room multi-criteria rubrics.
        """
        # Check offline replay store first
        if replay_key and replay_key in self.replay_store:
            return self._parse_replay_entry(self.replay_store[replay_key])

        # If LLM callable is available, attempt clean-room LLM grading
        if self.llm_callable is not None:
            try:
                prompt = self._build_evaluation_prompt(
                    question=question,
                    agent_answer=agent_answer,
                    golden_answer=golden_answer,
                    context=context,
                )
                raw_response = self.llm_callable(prompt)
                parsed_result = self._parse_llm_response(raw_response)
                if parsed_result:
                    return parsed_result
            except Exception as e:
                logger.warning(f"LLM callable evaluation failed: {e}. Falling back to clean-room heuristics.")

        # Fallback heuristic evaluation
        return self._evaluate_heuristics(
            question=question,
            agent_answer=agent_answer,
            golden_answer=golden_answer,
            context=context,
        )

    def _build_evaluation_prompt(
        self,
        question: str,
        agent_answer: str,
        golden_answer: str,
        context: Optional[str] = None,
    ) -> str:
        prompt = (
            f"You are an expert financial and risk evaluation judge in a clean-room benchmark harness.\n"
            f"Evaluate the agent's answer against the golden reference answer across 4 key dimensions:\n"
            f"1. Correctness (0.0 - 1.0): Numerical accuracy and factual alignment.\n"
            f"2. Faithfulness (0.0 - 1.0): Grounding in provided context, no hallucinations.\n"
            f"3. Reasoning Coherence (0.0 - 1.0): Logical structure and step-by-step clarity.\n"
            f"4. Provenance Citation (0.0 - 1.0): Citation of sources, line items, or W3C PROV metadata.\n\n"
            f"Question: {question}\n"
            f"Golden Answer: {golden_answer}\n"
        )
        if context:
            prompt += f"Context: {context}\n"
        prompt += f"Agent Answer: {agent_answer}\n\n"
        prompt += "Provide scores and reasoning for each dimension, followed by OVERALL_SCORE: <float>."
        return prompt

    def _parse_llm_response(self, response: str) -> Optional[JudgeResult]:
        """Extract scores from raw LLM text response."""
        try:
            match = re.search(r"OVERALL_SCORE:\s*([0-9.]+)", response, re.IGNORECASE)
            if match:
                score = float(match.group(1))
                score = max(0.0, min(1.0, score))
                passed = score >= self.pass_threshold
                return JudgeResult(
                    overall_score=score,
                    passed=passed,
                    verdict="PASS" if passed else "FAIL",
                    feedback="LLM Judge Evaluation Complete.",
                    raw_response=response,
                )
        except Exception as e:
            logger.error(f"Failed to parse LLM response: {e}")
        return None

    def _parse_replay_entry(self, entry: Dict[str, Any]) -> JudgeResult:
        """Hydrate JudgeResult from an offline replay dictionary."""
        dims = []
        for dim_data in entry.get("dimensions", []):
            dims.append(
                DimensionScore(
                    name=dim_data.get("name", "unknown"),
                    score=float(dim_data.get("score", 0.0)),
                    weight=float(dim_data.get("weight", 1.0)),
                    reasoning=dim_data.get("reasoning", ""),
                )
            )
        score = float(entry.get("overall_score", 0.0))
        passed = score >= self.pass_threshold
        return JudgeResult(
            overall_score=score,
            passed=passed,
            dimensions=dims,
            verdict=entry.get("verdict", "PASS" if passed else "FAIL"),
            feedback=entry.get("feedback", "Replay Evaluation"),
            raw_response=entry.get("raw_response"),
        )

    def _evaluate_heuristics(
        self,
        question: str,
        agent_answer: str,
        golden_answer: str,
        context: Optional[str] = None,
    ) -> JudgeResult:
        """Deterministic clean-room heuristic evaluation."""
        dimensions = []

        # 1. Correctness
        correctness_score = 0.0
        c_reasoning = "No match found."
        if golden_answer in agent_answer or golden_answer.replace("B", " Billion") in agent_answer:
            correctness_score = 1.0
            c_reasoning = "Exact string/phrase match with golden answer."
        else:
            gold_nums = [float(n) for n in re.findall(r"(\d+\.?\d*)", golden_answer)]
            agent_nums = [float(n) for n in re.findall(r"(\d+\.?\d*)", agent_answer)]
            if gold_nums and agent_nums:
                gold_num = gold_nums[0]
                best_diff = min(abs(gold_num - an) for an in agent_nums)
                rel_diff = best_diff / gold_num if gold_num != 0 else best_diff
                if rel_diff < 0.01:
                    correctness_score = 1.0
                    c_reasoning = "Numerical match within 1% tolerance."
                elif rel_diff < 0.05:
                    correctness_score = 0.75
                    c_reasoning = "Numerical match within 5% tolerance."
                elif rel_diff < 0.10:
                    correctness_score = 0.50
                    c_reasoning = "Numerical match within 10% tolerance."
                else:
                    c_reasoning = f"Numerical deviation: {rel_diff:.2%} relative error."

        dimensions.append(
            DimensionScore(
                name="correctness",
                score=correctness_score,
                weight=self.weights.get("correctness", 0.4),
                reasoning=c_reasoning,
            )
        )

        # 2. Faithfulness / Grounding
        faith_score = 0.8
        f_reasoning = "Answer appears plausible without obvious hallucinations."
        if context:
            context_words = set(re.findall(r"\w+", context.lower()))
            answer_words = set(re.findall(r"\w+", agent_answer.lower()))
            if context_words:
                overlap = len(answer_words.intersection(context_words)) / len(answer_words) if answer_words else 0.0
                faith_score = min(1.0, overlap * 1.5)
                f_reasoning = f"Context keyword overlap ratio: {overlap:.2f}"

        dimensions.append(
            DimensionScore(
                name="faithfulness",
                score=faith_score,
                weight=self.weights.get("faithfulness", 0.3),
                reasoning=f_reasoning,
            )
        )

        # 3. Reasoning Coherence
        coherent_score = 0.5
        coh_reasoning = "Standard response length."
        if len(agent_answer.split()) > 10 or "because" in agent_answer.lower() or "therefore" in agent_answer.lower():
            coherent_score = 0.9
            coh_reasoning = "Answer demonstrates structured reasoning indicators."

        dimensions.append(
            DimensionScore(
                name="reasoning_coherence",
                score=coherent_score,
                weight=self.weights.get("reasoning_coherence", 0.2),
                reasoning=coh_reasoning,
            )
        )

        # 4. Provenance / Citation
        prov_score = 0.0
        p_reasoning = "No citations or PROV metadata found."
        prov_keywords = ["source", "prov", "according to", "section", "table", "ref", "http"]
        if any(kw in agent_answer.lower() for kw in prov_keywords):
            prov_score = 1.0
            p_reasoning = "Explicit provenance or source citation present."

        dimensions.append(
            DimensionScore(
                name="provenance_citation",
                score=prov_score,
                weight=self.weights.get("provenance_citation", 0.1),
                reasoning=p_reasoning,
            )
        )

        # Calculate weighted average
        total_weight = sum(d.weight for d in dimensions)
        weighted_sum = sum(d.score * d.weight for d in dimensions)
        overall_score = round(weighted_sum / total_weight, 4) if total_weight > 0 else 0.0
        passed = overall_score >= self.pass_threshold

        return JudgeResult(
            overall_score=overall_score,
            passed=passed,
            dimensions=dimensions,
            verdict="PASS" if passed else "FAIL",
            feedback=f"Heuristic evaluation score: {overall_score:.4f} (Threshold: {self.pass_threshold})",
        )


def grade_answer(question: str, agent_answer: str, golden_answer: str) -> float:
    """
    Backward-compatible entry point for answer grading.

    Returns float overall score between 0.0 and 1.0.
    """
    judge = CleanRoomLLMJudge()
    result = judge.evaluate(question=question, agent_answer=agent_answer, golden_answer=golden_answer)
    return result.overall_score
