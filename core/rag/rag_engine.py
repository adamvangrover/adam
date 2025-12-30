import json
import logging
import numpy as np
from typing import List, Dict, Any, Optional
from core.llm_plugin import LLMPlugin
from core.rag.document_handling import chunk_text

logger = logging.getLogger(__name__)

class RAGEngine:
    """
    A simple, lightweight RAG engine for financial documents.
    Uses LLMPlugin for embeddings and generation.
    """

    def __init__(self, llm_plugin: Optional[LLMPlugin] = None):
        self.llm = llm_plugin or LLMPlugin()
        self.documents: List[Dict[str, Any]] = [] # List of {id, content, embedding, metadata}
        self.chunk_size = 1000
        self.chunk_overlap = 200

    def add_document(self, text: str, metadata: Optional[Dict[str, Any]] = None):
        """
        Chunks text, computes embeddings, and stores them in memory.
        """
        chunks = chunk_text(text, self.chunk_size, self.chunk_overlap)

        for i, chunk in enumerate(chunks):
            try:
                embedding = self.llm.get_embedding(chunk)
                doc_entry = {
                    "id": f"{metadata.get('id', 'doc')}_chunk_{i}" if metadata else f"doc_chunk_{i}",
                    "content": chunk,
                    "embedding": embedding,
                    "metadata": metadata or {}
                }
                self.documents.append(doc_entry)
            except Exception as e:
                logger.error(f"Failed to embed chunk {i}: {e}")

    def retrieve(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Retrieves the top_k most relevant chunks for the query using cosine similarity.
        """
        if not self.documents:
            return []

        try:
            query_embedding = self.llm.get_embedding(query)
        except Exception as e:
            logger.error(f"Failed to embed query: {e}")
            return []

        # Calculate cosine similarity
        results = []
        q_vec = np.array(query_embedding)
        norm_q = np.linalg.norm(q_vec)

        for doc in self.documents:
            d_vec = np.array(doc["embedding"])
            norm_d = np.linalg.norm(d_vec)

            if norm_q == 0 or norm_d == 0:
                similarity = 0
            else:
                similarity = np.dot(q_vec, d_vec) / (norm_q * norm_d)

            results.append({
                "doc": doc,
                "score": similarity
            })

        # Sort by score descending
        results.sort(key=lambda x: x["score"], reverse=True)
        return [r["doc"] for r in results[:top_k]]

    def query(self, query_text: str, context_documents: Optional[List[str]] = None) -> str:
        """
        Performs a RAG query: Retrieve -> Augment -> Generate.
        """
        if context_documents:
            # If documents are provided directly, use them (transient RAG)
            # This handles the case where we just want to query specific docs without adding to store
            # But for this simple engine, we assume we use the store + retrieve
            pass

        relevant_docs = self.retrieve(query_text)
        context_str = "\n\n".join([d["content"] for d in relevant_docs])

        prompt = (
            f"You are a financial analyst assistant. Answer the question based ONLY on the following context.\n\n"
            f"Context:\n{context_str}\n\n"
            f"Question: {query_text}\n\n"
            f"Answer:"
        )

        return self.llm.generate_text(prompt)
