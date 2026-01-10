from typing import Dict, Any, List, Optional
import logging
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
except ImportError:
    TfidfVectorizer = None
    cosine_similarity = None

class ConvictionScorer:
    """
    Implements Semantic Conviction Scoring.
    verifies claims against a 'Gold Standard' source using similarity metrics.
    """

    def __init__(self):
        if TfidfVectorizer:
            self.vectorizer = TfidfVectorizer(stop_words='english')
        else:
            self.vectorizer = None

    def calculate_conviction(self, claim: str, source_text: str) -> float:
        """
        Calculates a conviction score (0.0 - 1.0) indicating how well
        the source text supports the claim.
        """
        if not claim or not source_text:
            return 0.0

        if not self.vectorizer:
            # Fallback heuristic: word overlap
            claim_words = set(claim.lower().split())
            source_words = set(source_text.lower().split())
            overlap = len(claim_words.intersection(source_words))
            return min(1.0, overlap / max(1, len(claim_words)))

        try:
            # TF-IDF Cosine Similarity as a proxy for Entailment
            # (In production, use Cross-Encoders)
            tfidf_matrix = self.vectorizer.fit_transform([claim, source_text])
            score = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            return float(score)
        except Exception as e:
            logger.error(f"Scoring failed: {e}")
            return 0.5 # Neutral fallback

    def verify_artifact(self, artifact: Dict[str, Any], known_facts: List[str]) -> Dict[str, Any]:
        """
        Verifies an artifact content against a list of known facts (Gold Standard).
        """
        content = artifact.get("content", "")
        max_score = 0.0

        # Check against all known facts (e.g., from 10-K)
        # This is expensive O(N*M), in prod use Vector Index
        if known_facts:
            # Concatenate facts for context or check individually
            # Here we check max similarity with any fact chunk
            for fact in known_facts:
                score = self.calculate_conviction(content, fact)
                if score > max_score:
                    max_score = score
        else:
            # Self-consistency check or default trust if no ground truth
            max_score = 0.8 # Assume innocent until proven guilty

        artifact["conviction_score"] = max_score
        return artifact
