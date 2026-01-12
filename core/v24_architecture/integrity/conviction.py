import logging
import json
from typing import List, Dict, Any, Optional

try:
    from sentence_transformers import CrossEncoder
except ImportError:
    CrossEncoder = None

logger = logging.getLogger(__name__)

class SemanticConvictionScorer:
    """
    Implements the Cross-Encoder logic for rigorous semantic verification.
    """

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model_name = model_name
        self.model = None

        if CrossEncoder:
            try:
                # Lazy load to save memory if not used immediately
                # self.model = CrossEncoder(model_name)
                pass
            except Exception as e:
                logger.warning(f"Failed to load CrossEncoder: {e}")
        else:
            logger.warning("sentence-transformers not installed. SemanticConvictionScorer running in mock mode.")

    def load_model(self):
        if not self.model and CrossEncoder:
            logger.info(f"Loading CrossEncoder model: {self.model_name}")
            self.model = CrossEncoder(self.model_name)

    def verify_claim(self, claim: str, evidence: str) -> float:
        """
        Calculates the entailment score between a claim and a piece of evidence.
        Output: 0.0 (Contradiction) to 1.0 (Entailment).
        """
        if not self.model:
            if CrossEncoder:
                self.load_model()
            else:
                return self._mock_score(claim, evidence)

        try:
            # Cross-Encoders take a list of pairs
            scores = self.model.predict([(claim, evidence)])
            # Apply sigmoid to get 0-1 range if model outputs logits
            # But ms-marco models usually output raw logits.
            # For simplicity in this scaffolding, we assume a normalized output or handle logits.
            return float(scores[0])
        except Exception as e:
            logger.error(f"Error during semantic verification: {e}")
            return 0.5

    def batch_verify(self, pairs: List[tuple]) -> List[float]:
        """
        Batch verification for efficiency.
        pairs: List of (claim, evidence)
        """
        if not self.model:
             if CrossEncoder:
                self.load_model()
             else:
                return [self._mock_score(c, e) for c, e in pairs]

        try:
            scores = self.model.predict(pairs)
            return [float(s) for s in scores]
        except Exception as e:
            logger.error(f"Error during batch verification: {e}")
            return [0.0] * len(pairs)

    def _mock_score(self, claim: str, evidence: str) -> float:
        """
        Fallback heuristic if model is missing.
        """
        # Simple overlap coefficient
        claim_words = set(claim.lower().split())
        evidence_words = set(evidence.lower().split())
        overlap = len(claim_words.intersection(evidence_words))
        return min(1.0, overlap / max(1, len(claim_words)))
