# core/engine/evaluator.py

from typing import Dict
from core.evaluation.llm_judge import ConvictionScorer
from evals.graders.llm_judge import grade_answer

class EvaluatorKernel:
    def __init__(self, golden_dataset: Dict):
        self.scorer = ConvictionScorer()
        self.golden_dataset = golden_dataset

    def run_full_validation(self, prompt: str, output: str, golden_ref: str) -> Dict:
        # 1. Fast Filter
        conviction = self.scorer.score(output)
        if conviction['score'] < 0.6:
            return {"status": "REJECTED", "reason": "Insufficient conviction/data"}
        
        # 2. Logic Check
        logic_score = grade_answer(prompt, output, golden_ref)
        if logic_score < 0.5:
            return {"status": "REJECTED", "reason": "Logic failure"}
            
        # 3. Meta-Judge Lifecycle
        if logic_score < 1.0:
            # Trigger your llm_as_judge_lifecycle prompt here
            return self.trigger_meta_judge(prompt, output, golden_ref)
            
        return {"status": "ACCEPTED", "score": logic_score}
