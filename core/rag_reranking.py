from sentence_transformers import CrossEncoder

class RAGReRanker:
    """
    RAG to Prompt Bridge (Context Collapse).
    Uses a lightweight cross-encoder to re-rank chunks.
    """
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.encoder = CrossEncoder(model_name)

    def rerank(self, query: str, chunks: list, top_k: int = 5) -> list:
        """
        Re-ranks chunks based on relevance to the query.
        Returns the top_k chunks.
        """
        if not chunks:
            return []

        pairs = [[query, chunk] for chunk in chunks]
        scores = self.encoder.predict(pairs)

        # Sort chunks by score descending
        chunk_scores = list(zip(chunks, scores))
        chunk_scores.sort(key=lambda x: x[1], reverse=True)

        # Return top_k chunks
        return [chunk for chunk, score in chunk_scores[:top_k]]
