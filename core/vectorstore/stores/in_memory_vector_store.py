import logging
from typing import List, Tuple

import numpy as np

from core.vectorstore.base_vector_store import BaseVectorStore


class InMemoryVectorStore(BaseVectorStore):
    """
    A simple in-memory vector store using Python lists and NumPy for dot product similarity.
    Suitable for testing and small-scale use. Not optimized for performance or large datasets.
    """
    def __init__(self, embedding_dim: int):
        self.embedding_dim = embedding_dim
        self.documents: List[Tuple[str, np.ndarray]] = [] # Stores (text, embedding_vector)
        logging.info(f"InMemoryVectorStore initialized with embedding_dim: {self.embedding_dim}")

    async def add_documents(self, documents: List[Tuple[str, List[float]]]):
        """
        Adds documents and their embeddings to the store.
        Each document is a tuple of (text, embedding_list).
        """
        if not documents:
            return

        logging.debug(f"InMemoryVectorStore: Adding {len(documents)} documents.")
        for text, embedding_list in documents:
            if len(embedding_list) != self.embedding_dim:
                logging.error(f"InMemoryVectorStore: Embedding dimension mismatch. Expected {self.embedding_dim}, got {len(embedding_list)} for text: '{text[:50]}...'")
                # Decide: skip this doc, or raise error? For now, skip.
                continue

            embedding_vector = np.array(embedding_list, dtype=np.float32)
            self.documents.append((text, embedding_vector))
        logging.info(f"InMemoryVectorStore: Total documents in store: {len(self.documents)}")

    async def search(self, query_embedding: List[float], top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Searches the vector store for similar documents based on the query embedding.
        Returns a list of (document_text, similarity_score) tuples.
        Similarity is calculated using cosine similarity (dot product of normalized vectors).
        """
        if not self.documents:
            logging.warning("InMemoryVectorStore: Search called on empty store.")
            return []

        if len(query_embedding) != self.embedding_dim:
            logging.error(f"InMemoryVectorStore: Query embedding dimension mismatch. Expected {self.embedding_dim}, got {len(query_embedding)}.")
            return []

        query_vector = np.array(query_embedding, dtype=np.float32)

        # Normalize query vector
        query_norm = np.linalg.norm(query_vector)
        if query_norm == 0: # Avoid division by zero for zero vectors
            logging.warning("InMemoryVectorStore: Query vector is zero. Cannot compute similarity.")
            return []
        normalized_query_vector = query_vector / query_norm

        results = []
        for text, doc_vector in self.documents:
            # Normalize document vector
            doc_norm = np.linalg.norm(doc_vector)
            if doc_norm == 0: # Avoid division by zero for zero vectors
                similarity = 0.0
            else:
                normalized_doc_vector = doc_vector / doc_norm
                similarity = np.dot(normalized_query_vector, normalized_doc_vector)
            results.append((text, float(similarity))) # Ensure similarity is float

        # Sort by similarity in descending order
        results.sort(key=lambda item: item[1], reverse=True)

        return results[:top_k]

if __name__ == "__main__":
    async def main():
        # Example Usage
        dim = 4 # Small dimension for easy example
        store = InMemoryVectorStore(embedding_dim=dim)

        docs_to_add = [
            ("apple is a fruit", [0.1, 0.2, 0.3, 0.4]),
            ("banana is yellow", [0.5, 0.4, 0.1, 0.0]),
            ("apples and oranges", [0.1, 0.3, 0.2, 0.4]),
            ("i like bananas",   [0.4, 0.5, 0.1, 0.05]),
            ("cars are fast",    [0.0, 0.0, -0.5, -0.8])
        ]
        await store.add_documents(docs_to_add)
        print(f"Store content count: {len(store.documents)}")

        # Test search
        query_emb_apple = [0.1, 0.25, 0.2, 0.35] # Similar to apple docs
        search_results_apple = await store.search(query_emb_apple, top_k=3)
        print(f"\nSearch results for 'apple-like' query ({query_emb_apple}):")
        for doc, score in search_results_apple:
            print(f"  Score: {score:.4f} - Doc: '{doc}'")

        query_emb_car = [0.05, -0.05, -0.4, -0.9] # Similar to car doc
        search_results_car = await store.search(query_emb_car, top_k=2)
        print(f"\nSearch results for 'car-like' query ({query_emb_car}):")
        for doc, score in search_results_car:
            print(f"  Score: {score:.4f} - Doc: '{doc}'")

        # Test search with empty store
        empty_store = InMemoryVectorStore(embedding_dim=dim)
        empty_results = await empty_store.search(query_emb_apple)
        print(f"\nSearch results from empty store: {empty_results}")

        # Test adding doc with wrong dimension
        await store.add_documents([("wrong dim doc", [0.1, 0.2])]) # Should log error and skip

        # Test search with query of wrong dimension
        await store.search([0.1,0.2], top_k=2) # Should log error and return empty

    # import asyncio
    # asyncio.run(main())
    print("InMemoryVectorStore class defined. Run with an async event loop to test main().")
