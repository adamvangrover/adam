import logging
from typing import List
from core.embeddings.base_embedding_model import BaseEmbeddingModel


class DummyEmbeddingModel(BaseEmbeddingModel):
    """
    A dummy embedding model for testing and demonstration.
    Generates non-semantic, deterministic embeddings.
    """

    def __init__(self, model_name: str = "dummy-embedding-v1", embedding_dim: int = 128):
        self.model_name = model_name
        self.embedding_dim = embedding_dim
        logging.info(f"DummyEmbeddingModel initialized with model: {self.model_name}, dim: {self.embedding_dim}")

    async def generate_embedding(self, text: str) -> List[float]:
        logging.debug(f"DummyEmbeddingModel.generate_embedding called for text: '{text[:50]}...'")
        if not text:
            return [0.0] * self.embedding_dim

        # Simple deterministic embedding based on char codes, scaled
        # Use more chars for variability
        embedding = [(ord(char) % 128) / 128.0 - 0.5 for char in text[:self.embedding_dim*2]]

        # Ensure fixed dimension
        if len(embedding) < self.embedding_dim:
            embedding.extend([0.0] * (self.embedding_dim - len(embedding)))
        else:
            embedding = embedding[:self.embedding_dim]

        return embedding


if __name__ == "__main__":
    async def main():
        dummy_embedder = DummyEmbeddingModel(embedding_dim=10)

        text1 = "Hello world"
        emb1 = await dummy_embedder.generate_embedding(text1)
        print(f"Embedding for '{text1}': {emb1}")

        text2 = "Another example text"
        emb2 = await dummy_embedder.generate_embedding(text2)
        print(f"Embedding for '{text2}': {emb2}")

        emb_empty = await dummy_embedder.generate_embedding("")
        print(f"Embedding for empty string: {emb_empty}")

    # import asyncio
    # asyncio.run(main())
    print("DummyEmbeddingModel class defined. Run with an async event loop to test main().")
