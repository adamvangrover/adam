import logging
import numpy as np
from typing import Dict, List, Optional
from core.schemas.v23_5_schema import IntentCategory, RoutingResult

logger = logging.getLogger(__name__)

# Try to import sentence_transformers
try:
    from sentence_transformers import SentenceTransformer, util
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

class SemanticRouter:
    """
    Routes user queries to specific Intent Categories using Vector Embeddings.
    Replaces brittle keyword matching with semantic similarity.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", threshold: float = 0.65):
        self.threshold = threshold
        self.model = None
        self.routes: Dict[IntentCategory, Any] = {}

        # Define seed phrases for each category
        self.intent_definitions = {
            IntentCategory.DEEP_DIVE: [
                "comprehensive analysis", "full report", "deep dive", "strategic review",
                "detailed breakdown", "extensive research", "thorough investigation"
            ],
            IntentCategory.RISK_ALERT: [
                "risk exposure", "liquidity crisis", "credit check", "compliance alert",
                "volatility warning", "default probability", "stress test"
            ],
            IntentCategory.MARKET_UPDATE: [
                "stock price", "market update", "latest news", "ticker symbol",
                "current value", "trading volume", "price check"
            ],
            IntentCategory.COMPLIANCE_CHECK: [
                "regulatory compliance", "money laundering check", "KYC verification",
                "sanctions list", "legal audit"
            ]
        }

        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                self.model = SentenceTransformer(model_name)
                self._initialize_routes()
            except Exception as e:
                logger.warning(f"Failed to load SentenceTransformer: {e}. Using fallback.")
                self.model = None
        else:
            logger.warning("sentence-transformers not available. Using fallback.")

    def _initialize_routes(self):
        """
        Pre-computes embeddings for the route definitions.
        """
        if not self.model:
            return

        for category, phrases in self.intent_definitions.items():
            embeddings = self.model.encode(phrases)
            self.routes[category] = embeddings

    def route(self, query: str) -> RoutingResult:
        """
        Classifies the query into an IntentCategory.
        """
        if not self.model:
            return self._heuristic_route(query)

        try:
            query_vec = self.model.encode(query)
            best_category = IntentCategory.UNCERTAIN
            max_score = 0.0

            # Calculate cosine similarity against all categories
            for category, embeddings in self.routes.items():
                scores = util.cos_sim(query_vec, embeddings)
                category_max = float(scores.max())

                if category_max > max_score:
                    max_score = category_max
                    best_category = category

            if max_score < self.threshold:
                return RoutingResult(
                    intent=IntentCategory.UNCERTAIN,
                    confidence_score=max_score,
                    routing_metadata={"reason": "Below confidence threshold"}
                )

            return RoutingResult(
                intent=best_category,
                confidence_score=max_score,
                routing_metadata={"model": "SemanticRouter"}
            )

        except Exception as e:
            logger.error(f"Error in semantic routing: {e}")
            return self._heuristic_route(query)

    def _heuristic_route(self, query: str) -> RoutingResult:
        """
        Fallback logic using keyword matching (Legacy Logic).
        """
        query_lower = query.lower()

        for category, phrases in self.intent_definitions.items():
            for phrase in phrases:
                if phrase in query_lower:
                    return RoutingResult(
                        intent=category,
                        confidence_score=0.5, # Low confidence for heuristic
                        routing_metadata={"reason": "Keyword Match Fallback"}
                    )

        return RoutingResult(
            intent=IntentCategory.UNCERTAIN,
            confidence_score=0.0,
            routing_metadata={"reason": "No keyword match"}
        )
