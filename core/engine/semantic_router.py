import logging
import numpy as np
from typing import Dict, List, Tuple
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
except ImportError:
    TfidfVectorizer = None
    cosine_similarity = None

from core.utils.repo_context import RepoContextManager

logger = logging.getLogger(__name__)

class SemanticRouter:
    """
    Implements semantic intent classification using Vector Space Models.
    Replaces naive keyword matching with Cosine Similarity over TF-IDF vectors
    (or Dense Embeddings if sentence-transformers were available).

    Updated for v24: Integrates with RepoContextManager to allow dynamic
    customization of routes based on documentation.
    """

    INTENT_EXAMPLES = {
        "DEEP_DIVE": [
            "perform a comprehensive analysis of Apple",
            "deep dive into NVDA fundamentals",
            "full strategic review of Tesla",
            "detailed valuation report",
            "generate a complete investment memo",
            "360 degree view of the company",
            "assess management and competitive moat"
        ],
        "RISK_ALERT": [
            "check for liquidity risks",
            "monitor credit default swaps",
            "is the portfolio exposed to geopolitical tension?",
            "analyze downside scenarios",
            "stress test for interest rate hikes",
            "compliance violation alert",
            "risk exposure report"
        ],
        "MARKET_UPDATE": [
            "what is the price of AAPL?",
            "latest market news",
            "check stock quote",
            "how is the s&p 500 doing?",
            "market summary for today",
            "ticker symbol lookup"
        ],
        "CODE_GEN": [
            "write a python script to fetch data",
            "refactor this code",
            "generate a function for black scholes",
            "implement a trading algorithm",
            "create a new agent class"
        ],
        "SWARM": [
            "activate swarm intelligence",
            "run parallel scan",
            "hive mind analysis",
            "dispatch swarm scouts",
            "concurrent market scan"
        ],
        "CRISIS": [
            "simulate a market crash",
            "what happens if china invades taiwan?",
            "run a crisis simulation",
            "financial contagion stress test",
            "black swan event modeling"
        ]
    }

    def __init__(self):
        self.vectorizer = None
        self.intent_vectors = {}
        self.routes = list(self.INTENT_EXAMPLES.keys())

        # Load Context to enhance examples
        self._load_dynamic_examples()

        if TfidfVectorizer:
            self._train_vectorizer()
        else:
            logger.warning("Scikit-learn not found. SemanticRouter running in fallback mode (Keyword Match).")

    def _load_dynamic_examples(self):
        """
        Enhances intent examples by reading the AGENTS.md documentation.
        If the docs mention new agents or capabilities, we can (in theory)
        add them to the routing logic.
        """
        try:
            ctx = RepoContextManager()
            agent_defs = ctx.get_agent_definitions()

            # Simple heuristic: If "Crisis Simulation" is emphasized in docs, add more crisis keywords
            if "Crisis Simulation" in agent_defs:
                self.INTENT_EXAMPLES["CRISIS"].extend(["stress test portfolio", "macro shock analysis"])

            # If "Swarm" is present (v24), ensure Swarm keywords
            if "Swarm" in agent_defs:
                self.INTENT_EXAMPLES["SWARM"].extend(["hive mind", "distributed analysis"])

        except Exception as e:
            logger.warning(f"Failed to load dynamic routing examples: {e}")

    def _train_vectorizer(self):
        """Pre-computes vectors for intent prototypes."""
        corpus = []
        self.intent_map = [] # specific index to intent mapping

        for intent, examples in self.INTENT_EXAMPLES.items():
            for ex in examples:
                corpus.append(ex)
                self.intent_map.append(intent)

        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.vector_matrix = self.vectorizer.fit_transform(corpus)
        logger.info(f"SemanticRouter trained on {len(corpus)} examples.")

    def route(self, query: str, threshold: float = 0.3) -> str:
        """
        Classifies the query into an intent category.
        """
        if not self.vectorizer:
            return self._fallback_keyword_route(query)

        query_vec = self.vectorizer.transform([query])

        # Calculate similarity against all examples
        similarities = cosine_similarity(query_vec, self.vector_matrix).flatten()

        # Find the best match
        best_idx = np.argmax(similarities)
        best_score = similarities[best_idx]
        best_intent = self.intent_map[best_idx]

        logger.info(f"Semantic Routing: '{query}' -> {best_intent} (Score: {best_score:.4f})")

        if best_score < threshold:
            logger.info("Confidence below threshold. Defaulting to MEDIUM.")
            return "MEDIUM"

        return best_intent

    def _fallback_keyword_route(self, query: str) -> str:
        """Legacy keyword matching for environments without sklearn."""
        query_lower = query.lower()
        if "deep" in query_lower or "analysis" in query_lower:
            return "DEEP_DIVE"
        if "risk" in query_lower or "stress" in query_lower:
            return "RISK_ALERT"
        if "code" in query_lower:
            return "CODE_GEN"
        if "simulation" in query_lower or "crash" in query_lower:
            return "CRISIS"
        return "MEDIUM"
