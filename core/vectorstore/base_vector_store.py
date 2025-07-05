from abc import ABC, abstractmethod
from typing import List, Tuple

class BaseVectorStore(ABC):
    @abstractmethod
    async def add_documents(self, documents: List[Tuple[str, List[float]]]):
        """
        Adds documents and their embeddings to the vector store.
        Each document is a tuple of (text, embedding).
        """
        pass

    @abstractmethod
    async def search(self, query_embedding: List[float], top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Searches the vector store for similar documents based on the query embedding.
        Returns a list of (document_text, similarity_score) tuples.
        """
        pass
