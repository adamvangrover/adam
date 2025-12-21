from abc import ABC, abstractmethod


class BaseEmbeddingModel(ABC):
    @abstractmethod
    async def generate_embedding(self, text: str) -> list[float]:
        """
        Generates an embedding for the given text.
        """
        pass
