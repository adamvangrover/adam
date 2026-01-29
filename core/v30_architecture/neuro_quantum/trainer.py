from typing import List, Tuple
from .ontology import SemanticLabel, OntologyTuple

class LivePromptTrainer:
    """
    Simulates the "live training set" functionality.
    It takes raw prompts and converts them into labeled datasets suitable for the Reservoir.
    """

    def __init__(self):
        # In a real system, this might use an LLM to generate labels ("Auto-Labeling")
        # For now, we use simple heuristic keyword matching to simulate 'ground truth'.
        pass

    def generate_dataset(self, prompts: List[str]) -> Tuple[List[str], List[SemanticLabel]]:
        """
        Dynamically labels the incoming prompts to create a training set.
        """
        labeled_prompts = []
        labels = []

        for p in prompts:
            label = self._heuristic_label(p)
            labeled_prompts.append(p)
            labels.append(label)

        return labeled_prompts, labels

    def _heuristic_label(self, prompt: str) -> SemanticLabel:
        p_lower = prompt.lower()
        if "risk" in p_lower or "crash" in p_lower or "drop" in p_lower:
            return SemanticLabel.RISK
        elif "growth" in p_lower or "surge" in p_lower or "profit" in p_lower:
            return SemanticLabel.GROWTH
        elif "stable" in p_lower or "flat" in p_lower:
            return SemanticLabel.STABILITY
        elif "chaos" in p_lower or "uncertain" in p_lower:
            return SemanticLabel.CHAOS
        else:
            return SemanticLabel.STAGNATION
