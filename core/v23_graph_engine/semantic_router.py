import logging
import numpy as np
from typing import Dict, List, Optional, Callable, Any
from enum import Enum

# Standardizing the Intent Categories
class IntentCategory(Enum):
    MARKET_DATA = "market_data"
    FINANCIAL_ANALYSIS = "financial_analysis"
    RISK_ALERT = "risk_alert"
    COMPLIANCE_CHECK = "compliance_check"
    GENERAL_KNOWLEDGE = "general_knowledge"
    UNKNOWN = "unknown"

logger = logging.getLogger(__name__)

# Logic for Vector Similarity Support
try:
    from sentence_transformers import SentenceTransformer, util
    ST_AVAILABLE = True
except ImportError:
    ST_AVAILABLE = False

class SemanticRouter:
    """
    A hybrid router that maps user queries to executable handlers using 
    Semantic Similarity (Vector Embeddings) and Keyword Heuristics.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", threshold: float = 0.65):
        self.threshold = threshold
        self.handlers: Dict[IntentCategory, Callable] = {}
        self.route_embeddings: Dict[IntentCategory, Any] = {}
        self.model = None

        # Seed phrases for semantic mapping
        self.intent_definitions = {
            IntentCategory.MARKET_DATA: ["stock price", "ticker symbol", "trading volume", "market update"],
            IntentCategory.FINANCIAL_ANALYSIS: ["q3 earnings", "balance sheet", "fundamental analysis", "financial report"],
            IntentCategory.RISK_ALERT: ["liquidity risk", "volatility warning", "credit exposure", "default risk"],
            IntentCategory.COMPLIANCE_CHECK: ["kyc verification", "aml check", "regulatory audit", "sanctions"],
            IntentCategory.GENERAL_KNOWLEDGE: ["what is", "how does", "explain the concept of"]
        }

        if ST_AVAILABLE:
            try:
                self.model = SentenceTransformer(model_name)
                self._precompute_embeddings()
            except Exception as e:
                logger.warning(f"Embedding model failed: {e}. Falling back to heuristics.")
        else:
            logger.warning("sentence-transformers not found. Using keyword heuristics.")

    def _precompute_embeddings(self):
        """Encodes seed phrases into vectors for fast similarity lookup."""
        if not self.model: return
        for category, phrases in self.intent_definitions.items():
            self.route_embeddings[category] = self.model.encode(phrases)

    def register_handler(self, category: IntentCategory, handler: Callable):
        """Binds a specific function to an intent category."""
        self.handlers[category] = handler

    def _default_handler(self, query: str):
        return f"Intent could not be determined for: '{query}'"

    def get_handler(self, query: str) -> Callable:
        """
        Determines intent and returns the bound handler.
        """
        category, confidence = self._determine_intent(query)
        logger.info(f"Routed to {category} (Confidence: {confidence:.2f})")
        
        return self.handlers.get(category, self._default_handler)

    def _determine_intent(self, query: str) -> (IntentCategory, float):
        """Hybrid logic: Embeddings first, Heuristics second."""
        # 1. Try Semantic Vector Match
        if self.model:
            try:
                query_vec = self.model.encode(query)
                best_cat, max_score = IntentCategory.UNKNOWN, 0.0

                for category, embeddings in self.route_embeddings.items():
                    score = float(util.cos_sim(query_vec, embeddings).max())
                    if score > max_score:
                        max_score, best_cat = score, category

                if max_score >= self.threshold:
                    return best_cat, max_score
            except Exception as e:
                logger.error(f"Semantic match error: {e}")

        # 2. Fallback: Keyword Heuristics
        query_lower = query.lower()
        for category, phrases in self.intent_definitions.items():
            if any(p in query_lower for p in phrases):
                return category, 0.5  # Fixed confidence for keywords
        
        return IntentCategory.UNKNOWN, 0.0

# --- Example Usage ---
if __name__ == "__main__":
    router = SemanticRouter()

    # Define simple handlers
    router.register_handler(IntentCategory.MARKET_DATA, lambda q: f"Fetching live ticker data for {q}")
    router.register_handler(IntentCategory.FINANCIAL_ANALYSIS, lambda q: f"Running deep dive analysis on {q}")

    # Test routing
    user_query = "Can you check the current trading volume for NVDA?"
    handler = router.get_handler(user_query)
    print(handler(user_query))
