import logging
import random
import math
import numpy as np
from typing import List, Tuple, Union, Optional

# Mock schema import if not available in current environment for standalone testing
try:
    from core.schemas.v23_5_schema import IngestedArtifact
except ImportError:
    class IngestedArtifact:
        def __init__(self, content, conviction_score=0.0):
            self.content = content
            self.conviction_score = conviction_score

# Try to import sentence_transformers, mock if unavailable
try:
    from sentence_transformers import CrossEncoder
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

logger = logging.getLogger(__name__)

# ==========================================
# 1. Statistical & Mathematical Utilities
#    (derived from the functional snippet)
# ==========================================

def softmax(x: np.ndarray) -> np.ndarray:
    """Compute softmax values for each set of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

def calculate_statistical_conviction(scores: Union[List[float], np.ndarray]) -> float:
    """
    Calculates a conviction score based on the variance of a list of scores.
    Formula: mean - (0.5 * std_dev). Penalizes high variance.
    """
    if not scores:
        return 0.0

    scores_array = np.array(scores)
    mean_score = np.mean(scores_array)
    std_dev = np.std(scores_array)

    # A simple conviction metric that rewards consistency
    conviction = mean_score - (0.5 * std_dev)
    
    return max(0.0, min(1.0, conviction))

def calculate_entropy_conviction(logits: np.ndarray) -> float:
    """
    Calculates conviction based on the entropy of logits (Semantic Conviction).
    Lower entropy (high certainty) -> Higher conviction.
    """
    probabilities = softmax(logits)
    # Add epsilon to prevent log(0)
    entropy = -np.sum(probabilities * np.log(probabilities + 1e-9))
    max_entropy = np.log(len(logits))
    
    if max_entropy == 0:
        return 0.0
        
    normalized_entropy = entropy / max_entropy
    conviction = 1.0 - normalized_entropy
    return conviction

def aggregate_conviction_scores(scores: List[float]) -> float:
    """
    Smart wrapper to determine the best conviction metric for a list of scores.
    Decides between Entropy (for logits) or Statistical Variance (for probabilities).
    """
    try:
        # Flatten list of lists if necessary
        if scores and isinstance(scores[0], list):
            flat_scores = [item for sublist in scores for item in sublist]
            scores_array = np.array(flat_scores)
        else:
            scores_array = np.array(scores)

        if len(scores_array) == 0:
            return 0.0

        # Heuristic: if values are outside [0,1], treat as logits -> Entropy
        if np.any(scores_array > 1.0) or np.any(scores_array < 0.0):
            return calculate_entropy_conviction(scores_array)
        else:
            # Treat as normalized probabilities -> Statistical
            return calculate_statistical_conviction(scores)

    except Exception as e:
        logger.error(f"Error calculating aggregated conviction: {e}")
        return 0.0

def hybrid_conviction(
    semantic_score: float,
    statistical_score: float,
    semantic_weight: float = 0.5
) -> float:
    """Combines semantic (model) and statistical (distribution) scores."""
    return (semantic_score * semantic_weight) + (statistical_score * (1 - semantic_weight))


# ==========================================
# 2. Semantic Engine Class
#    (derived from the class-based snippet)
# ==========================================

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
                # In production, ensure model weights are cached or downloaded
                self.model = CrossEncoder(model_name)
            except Exception as e:
                logger.warning(f"Failed to load CrossEncoder model: {e}. Falling back to mock logic.")
                self.model = None
        else:
            logger.warning("sentence-transformers not installed. Using heuristic fallback.")

    def calculate_pair_conviction(self, claim: str, source_text: str) -> float:
        """
        Calculates the conviction score (0.0 to 1.0) for a single claim against a single source.
        """
        if self.model:
            try:
                # Predict returns logits for ms-marco models
                scores = self.model.predict([(claim, source_text)])
                
                # Handle single float or array return
                logit = float(scores[0]) if isinstance(scores, (list, tuple, np.ndarray)) else float(scores)
                
                # Normalize via Sigmoid: 1 / (1 + exp(-x))
                score = 1 / (1 + math.exp(-logit))
                return score
            except Exception as e:
                logger.error(f"Error during Cross-Encoder prediction: {e}")
                return self._heuristic_fallback(claim, source_text)
        else:
            return self._heuristic_fallback(claim, source_text)

    def verify_artifact(self, artifact: IngestedArtifact, trusted_source: str) -> IngestedArtifact:
        """
        Verifies an ingested artifact against a trusted source and updates its conviction score.
        """
        # Limit length for performance reasons
        score = self.calculate_pair_conviction(artifact.content[:500], trusted_source[:2000])
        artifact.conviction_score = score
        return artifact

    def verify_batch_consensus(self, claim: str, sources: List[str]) -> float:
        """
        Calculates conviction based on consensus across multiple sources.
        Uses the statistical utilities to aggregate individual pair scores.
        """
        scores = []
        for source in sources:
            scores.append(self.calculate_pair_conviction(claim, source))
        
        # Use the aggregated metric (likely statistical since we normalized to 0-1 above)
        return aggregate_conviction_scores(scores)

    def _heuristic_fallback(self, claim: str, source_text: str) -> float:
        """
        A robust fallback using Jaccard similarity and number matching.
        """
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

        return min(1.0, overlap_score + num_match_bonus)

if __name__ == "__main__":
    # Simple test harness
    engine = SemanticConvictionEngine()
    
    # Test 1: Pair Conviction (Heuristic or Model)
    s1 = engine.calculate_pair_conviction("Revenue is 5M", "The company revenue hit 5M last year.")
    print(f"Single Pair Score: {s1}")

    # Test 2: Aggregated Statistical Conviction
    test_scores = [0.9, 0.85, 0.95, 0.2] # High variance due to outlier
    agg_score = aggregate_conviction_scores(test_scores)
    print(f"Aggregated Score (High Variance): {agg_score}")
    
    # Test 3: Logits Entropy
    logits = np.array([10.0, 1.0, 1.0]) # High certainty
    print(f"Entropy Conviction (Logits): {aggregate_conviction_scores(logits)}")
