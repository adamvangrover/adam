from abc import ABC, abstractmethod

class BaseLLMEngine(ABC):
    @abstractmethod
    async def generate_response(self, prompt: str, context: str = None) -> str:
        """
        Generates a response from the LLM based on the given prompt and optional context.
        """
        pass

    @abstractmethod
    async def generate_embedding(self, text: str) -> list[float]:
        """
        Generates an embedding for the given text.
        Note: Some LLM providers offer embedding capabilities.
        If the chosen LLM does not, a separate embedding model should be used.
        """
        pass
