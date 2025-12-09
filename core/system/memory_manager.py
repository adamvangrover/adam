import json
import os
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.warning("scikit-learn not installed. Vector search capabilities disabled.")

class MemoryManager:
    """
    Manages the long-term memory of the system using a local JSON store.
    Stores analysis history to allow agents to recall past insights.
    """
    def __init__(self, storage_file: str = "data/memory/analysis_history.json"):
        self.storage_file = storage_file
        self.ensure_storage_exists()

    def ensure_storage_exists(self):
        os.makedirs(os.path.dirname(self.storage_file), exist_ok=True)
        if not os.path.exists(self.storage_file):
            with open(self.storage_file, 'w') as f:
                json.dump([], f)

    def load_history(self) -> List[Dict[str, Any]]:
        try:
            with open(self.storage_file, 'r') as f:
                try:
                    return json.load(f)
                except json.JSONDecodeError:
                    return []
        except FileNotFoundError:
            return []

    def save_analysis(self, company_id: str, analysis_summary: str, metrics: Dict[str, Any] = None):
        """
        Saves an analysis record.
        """
        entry = {
            "timestamp": datetime.now().isoformat(),
            "company_id": company_id,
            "analysis_summary": analysis_summary,
            "metrics": metrics or {}
        }

        try:
            history = self.load_history()
            history.append(entry)

            with open(self.storage_file, 'w') as f:
                json.dump(history, f, indent=2)

            logger.info(f"Saved analysis for {company_id} to memory.")
        except Exception as e:
            logger.error(f"Failed to save analysis for {company_id}: {e}")

    def query_history(self, company_id: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieves past analysis for a specific company.
        """
        try:
            history = self.load_history()

            # Filter by company_id
            company_history = [h for h in history if h.get("company_id") == company_id]

            # Sort by timestamp descending
            company_history.sort(key=lambda x: x["timestamp"], reverse=True)

            return company_history[:limit]
        except Exception as e:
            logger.error(f"Failed to query history for {company_id}: {e}")
            return []

    def get_last_analysis(self, company_id: str) -> Optional[Dict[str, Any]]:
        history = self.query_history(company_id, limit=1)
        return history[0] if history else None


class VectorMemoryManager(MemoryManager):
    """
    Enhanced Memory Manager with Vector Search capabilities.
    """
    def __init__(self, storage_file: str = "data/memory/analysis_history.json"):
        super().__init__(storage_file)
        self.vectorizer = None
        self.tfidf_matrix = None
        self.history_cache = []

        if SKLEARN_AVAILABLE:
            self.vectorizer = TfidfVectorizer(stop_words='english')
            self._refresh_vectors()

    def _refresh_vectors(self):
        if not SKLEARN_AVAILABLE: return

        self.history_cache = self.load_history()
        corpus = [h.get("analysis_summary", "") for h in self.history_cache]

        # Filter out empty docs
        valid_corpus = [doc for doc in corpus if doc.strip()]

        if valid_corpus and len(valid_corpus) > 0:
            try:
                self.tfidf_matrix = self.vectorizer.fit_transform(corpus)
            except ValueError:
                self.tfidf_matrix = None
        else:
            self.tfidf_matrix = None

    def save_analysis(self, company_id: str, analysis_summary: str, metrics: Dict[str, Any] = None):
        super().save_analysis(company_id, analysis_summary, metrics)
        if SKLEARN_AVAILABLE:
            self._refresh_vectors()

    def search_similar(self, query: str, limit: int = 3) -> List[Dict[str, Any]]:
        """
        Finds analysis entries similar to the query string.
        """
        if not SKLEARN_AVAILABLE or self.tfidf_matrix is None or not self.history_cache:
            return []

        try:
            query_vec = self.vectorizer.transform([query])
            cosine_similarities = cosine_similarity(query_vec, self.tfidf_matrix).flatten()

            # Get top indices
            # argsort returns indices of elements in sorted order (ascending), so we take last 'limit' elements
            related_docs_indices = cosine_similarities.argsort()[:-limit-1:-1]

            results = []
            for i in related_docs_indices:
                if i < len(self.history_cache):
                    score = float(cosine_similarities[i])
                    if score > 0.1: # Minimum similarity threshold
                        entry = self.history_cache[i].copy()
                        entry['similarity_score'] = score
                        results.append(entry)
            return results
        except Exception as e:
            logger.error(f"Error during vector search: {e}")
            return []
