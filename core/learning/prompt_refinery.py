import time
from typing import Dict, Any, List, Optional
from core.evaluation.iterative_llm_judge import IterativeLLMJudge, LLMJudgeScore
from core.evaluation.system_judge import SystemJudge, SystemJudgeMetrics

class PromptRefinery:
    """
    Manages the iterative loop: generating, judging, and refining prompts to maximize scores.
    """
    def __init__(self, llm_plugin: Any, judge: IterativeLLMJudge, system_judge: SystemJudge):
        self.llm_plugin = llm_plugin
        self.judge = judge
        self.system_judge = system_judge

    async def refine_prompt_loop(self, initial_prompt: str, max_iterations: int = 3, target_score: float = 90.0) -> Dict[str, Any]:
        """
        Iteratively refines a prompt until the target score is reached or max_iterations is hit.
        """
        current_prompt = initial_prompt
        history = []

        for i in range(max_iterations):
            start_time = time.time()
            # Generate output (Mocked generation if no true async generate available in plugin)
            if hasattr(self.llm_plugin, 'generate_text_async'):
                output_text = await self.llm_plugin.generate_text_async(current_prompt)
            else:
                output_text = "Simulated output based on: " + current_prompt
            end_time = time.time()

            # System Audit
            exec_stats = {
                "start_time": start_time,
                "end_time": end_time,
                "token_usage": {"prompt_tokens": len(current_prompt)//4, "completion_tokens": len(output_text)//4}
            }
            sys_metrics = self.system_judge.evaluate(exec_stats, output_text)

            # LLM Audit
            llm_metrics = self.judge.evaluate(current_prompt, output_text)

            iteration_data = {
                "iteration": i + 1,
                "prompt": current_prompt,
                "output": output_text,
                "system_metrics": sys_metrics.model_dump(),
                "llm_metrics": llm_metrics.model_dump()
            }
            history.append(iteration_data)

            if llm_metrics.overall_score >= target_score:
                break

            # Refine prompt based on critique
            current_prompt = self._generate_refined_prompt(current_prompt, llm_metrics.critique, llm_metrics.improvement_suggestions)

        return {
            "final_prompt": current_prompt,
            "iterations": len(history),
            "history": history
        }

    def _generate_refined_prompt(self, current_prompt: str, critique: str, suggestions: List[str]) -> str:
        """
        Uses the LLM to rewrite the prompt based on judge feedback.
        """
        refinement_instructions = f"""
        You are a Prompt Engineer. Improve the following prompt based on the critique and suggestions provided.
        Return ONLY the improved prompt text.

        --- CURRENT PROMPT ---
        {current_prompt}

        --- CRITIQUE ---
        {critique}

        --- SUGGESTIONS ---
        {" ".join(suggestions)}
        """
        if hasattr(self.llm_plugin, 'generate_text'):
            return self.llm_plugin.generate_text(refinement_instructions)

        # Fallback
        return current_prompt + "\n\n# Updated based on feedback:\n" + "\n".join(suggestions)
