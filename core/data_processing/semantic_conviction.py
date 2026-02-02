import logging
import random
import numpy as np
from typing import List, Tuple
from core.schemas.v23_5_schema import IngestedArtifact

# Try to import sentence_transformers, mock if unavailable
try:
    from sentence_transformers import CrossEncoder
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

logger = logging.getLogger(__name__)

class SemanticConvictionEngine:
    """
    Implements Semantic Conviction Scoring using Cross-Encoders.
    Calculates the probability that a Claim is entailed by a Source.
    """

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model_name = model_name
        self.model = None
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                # In a real environment, we would load the model here.
                # self.model = CrossEncoder(model_name)
                # For this environment, we might not have internet or the model weights.
                # So we will wrap it in a try-except block or mock it if instantiation fails.
                self.model = CrossEncoder(model_name)
            except Exception as e:
                logger.warning(f"Failed to load CrossEncoder model: {e}. Falling back to mock logic.")
                self.model = None
        else:
            logger.warning("sentence-transformers not installed. Using heuristic fallback.")

    def calculate_conviction(self, claim: str, source_text: str) -> float:
        """
        Calculates the conviction score (0.0 to 1.0) for a claim against a source.
        """
        if self.model:
            try:
                # The model returns a logit or a score. We might need to normalize it via sigmoid if it's not already.
                # ms-marco models usually output raw logits.
                scores = self.model.predict([(claim, source_text)])
                # Simple normalization for demonstration (logits can be negative)
                # Ideally use a sigmoid function: 1 / (1 + exp(-x))
                import math
                logit = float(scores[0]) if isinstance(scores, (list, tuple, np.ndarray)) else float(scores)
                score = 1 / (1 + math.exp(-logit))
                return score
            except Exception as e:
                logger.error(f"Error during Cross-Encoder prediction: {e}")
                return self._heuristic_fallback(claim, source_text)
        else:
            return self._heuristic_fallback(claim, source_text)

    def _heuristic_fallback(self, claim: str, source_text: str) -> float:
        """
        A robust fallback that uses basic overlap if the heavy model is missing.
        """
        # Simple Jaccard similarity or keyword overlap
        claim_words = set(claim.lower().split())
        source_words = set(source_text.lower().split())

        if not claim_words:
            return 0.0

        intersection = claim_words.intersection(source_words)
        overlap_score = len(intersection) / len(claim_words)

        # Boost if numbers match (simple extraction)
        claim_nums = [s for s in claim.split() if s.replace('.', '', 1).isdigit()]
        source_nums = [s for s in source_text.split() if s.replace('.', '', 1).isdigit()]

        num_match_bonus = 0.0
        for num in claim_nums:
            if num in source_nums:
                num_match_bonus += 0.2

        final_score = min(1.0, overlap_score + num_match_bonus)
        return final_score

    def verify_artifact(self, artifact: IngestedArtifact, trusted_source: str) -> IngestedArtifact:
        """
        Verifies an ingested artifact against a trusted source and updates its conviction score.
        """
        score = self.calculate_conviction(artifact.content[:500], trusted_source[:2000]) # Limit length for perf
        artifact.conviction_score = score
        return artifact
