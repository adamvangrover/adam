from typing import Dict, Any, Optional
import json
from core.agents.agent_base import AgentBase, AgentInput, AgentOutput
from core.evaluation.system_judge import SystemJudge, EvaluationRubric
from core.evaluation.iterative_llm_judge import IterativeLLMJudge
from core.learning.prompt_refinery import PromptRefinery
from core.monitoring.drift_monitor import ModelDriftMonitor
from core.evaluation.rubric_logger import EvaluationMarkdownLogger
import logging

logger = logging.getLogger(__name__)

class RefinementAgent(AgentBase):
    """
    An agent that integrates the LLM-as-a-Judge and Prompt Refinery pipelines.
    It takes an initial prompt, iteratively refines it against a Rubric, logs the results
    in Markdown, and returns the optimized prompt.
    """
    def __init__(self, config: Dict[str, Any] = None, kernel: Optional[Any] = None):
        if config is None:
            config = {"name": "RefinementAgent"}
        super().__init__(config, kernel=kernel)

        # Setup Default Rubric
        default_rubrics = [
            EvaluationRubric(criteria="Clarity", max_score=10, weight=0.4, description="Prompt is unambiguous and clear."),
            EvaluationRubric(criteria="Formatting constraints", max_score=5, weight=0.6, description="Prompt explicitly asks for a structured format.")
        ]

        self.system_judge = SystemJudge()
        self.llm_judge = IterativeLLMJudge(rubrics=default_rubrics, llm_plugin=self.kernel.get_llm() if self.kernel else None)
        self.refinery = PromptRefinery(llm_plugin=self.kernel.get_llm() if self.kernel else None, judge=self.llm_judge, system_judge=self.system_judge)
        self.drift_monitor = ModelDriftMonitor()
        self.markdown_logger = EvaluationMarkdownLogger()

    async def execute(self, input_data: AgentInput) -> AgentOutput:
        initial_prompt = input_data.query

        logger.info(f"Starting prompt refinement loop for prompt: {initial_prompt[:50]}...")

        refinement_result = await self.refinery.refine_prompt_loop(
            initial_prompt=initial_prompt,
            max_iterations=self.config.get("max_iterations", 3),
            target_score=self.config.get("target_score", 90.0)
        )

        # Log to Drift Monitor
        if refinement_result["history"]:
            last_iteration = refinement_result["history"][-1]
            self.drift_monitor.log_execution(
                last_iteration.get("system_metrics", {}),
                last_iteration.get("llm_metrics", {}).get("overall_score", 0.0)
            )

        drift_report = self.drift_monitor.check_drift()

        # Create Markdown Log
        log_path = self.markdown_logger.log_refinement_session(refinement_result, drift_report)
        logger.info(f"Refinement session logged to {log_path}")

        return AgentOutput(
            answer=refinement_result["final_prompt"],
            sources=["IterativeLLMJudge", "PromptRefinery"],
            confidence=refinement_result["history"][-1]["llm_metrics"]["overall_score"] / 100.0 if refinement_result["history"] else 0.0,
            metadata={
                "iterations": refinement_result["iterations"],
                "drift_report": drift_report,
                "log_path": log_path
            }
        )
