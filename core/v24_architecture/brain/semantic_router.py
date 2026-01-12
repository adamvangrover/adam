import logging
import numpy as np
from typing import Dict, List, Optional
try:
    from sentence_transformers import SentenceTransformer, util
except ImportError:
    SentenceTransformer = None
    util = None

logger = logging.getLogger(__name__)

class SemanticRouter:
    """
    v24 Production Semantic Router.
    Uses dense vector embeddings (MiniLM) for intent classification.
    """

    ROUTES = {
        "DEEP_DIVE": [
            "comprehensive analysis", "full report", "deep dive", "strategic review",
            "detailed breakdown", "complete due diligence"
        ],
        "RISK_ALERT": [
            "risk exposure", "liquidity crisis", "credit check", "compliance alert",
            "stress test", "volatility warning", "downside protection"
        ],
        "MARKET_UPDATE": [
            "stock price", "market news", "ticker check", "latest quote",
            "volume data", "s&p 500 status"
        ],
        "UNCERTAIN": [] # Fallback
    }

    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        self.model = None
        self.intent_embeddings = {}

        if SentenceTransformer:
            try:
                # self.model = SentenceTransformer(model_name)
                # self._precompute_embeddings()
                pass
            except Exception as e:
                logger.warning(f"Failed to load SentenceTransformer: {e}")
        else:
            logger.warning("sentence-transformers not installed. SemanticRouter running in fallback mode.")

    def load_model(self):
        if not self.model and SentenceTransformer:
             logger.info("Loading Embedding Model for Router...")
             self.model = SentenceTransformer('all-MiniLM-L6-v2')
             self._precompute_embeddings()

    def _precompute_embeddings(self):
        logger.info("Pre-computing intent embeddings...")
        for route, phrases in self.ROUTES.items():
            if phrases:
                self.intent_embeddings[route] = self.model.encode(phrases)

    def route(self, query: str, threshold: float = 0.60) -> str:
        """
        Routes the query based on cosine similarity to intent clusters.
        """
        if not self.model:
            if SentenceTransformer:
                self.load_model()
            else:
                return self._fallback_route(query)

        query_vec = self.model.encode(query)
        results = {}

        for route, vectors in self.intent_embeddings.items():
            # Calculate similarity with all examples in the route
            if vectors is not None and len(vectors) > 0:
                # util.cos_sim returns a tensor, we want the max score
                scores = util.cos_sim(query_vec, vectors)
                max_score = float(scores.max())
                results[route] = max_score
            else:
                results[route] = 0.0

        # Find best match
        if not results:
            return "UNCERTAIN"

        best_route, score = max(results.items(), key=lambda x: x[1])

        logger.info(f"Semantic Route: {query} -> {best_route} (Score: {score:.3f})")

        if score < threshold:
            return "UNCERTAIN"

        return best_route

    def _fallback_route(self, query: str) -> str:
        """
        Simple keyword fallback if ML is unavailable.
        """
        q = query.lower()
        if "deep" in q or "analysis" in q or "report" in q:
            return "DEEP_DIVE"
        if "risk" in q or "alert" in q or "crash" in q:
            return "RISK_ALERT"
        if "price" in q or "market" in q or "quote" in q:
            return "MARKET_UPDATE"
        return "UNCERTAIN"
