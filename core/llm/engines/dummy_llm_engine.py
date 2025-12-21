import logging
from typing import List

from core.llm.base_llm_engine import BaseLLMEngine


class DummyLLMEngine(BaseLLMEngine):
    """
    A dummy LLM engine for testing and demonstration.
    It echoes the prompt and context.
    """
    def __init__(self, model_name: str = "dummy-model-v1"):
        self.model_name = model_name
        logging.info(f"DummyLLMEngine initialized with model: {self.model_name}")

    async def generate_response(self, prompt: str, context: str = None) -> str:
        logging.debug(f"DummyLLMEngine.generate_response called with prompt: '{prompt[:50]}...', context: '{str(context)[:50]}...'")
        response = f"DummyResponse: [Prompt: {prompt}]"
        if context:
            response += f" [Context: {context}]"

        # Simulate some processing time if needed
        # await asyncio.sleep(0.1)
        return response

    async def generate_embedding(self, text: str) -> List[float]:
        """
        Generates a dummy embedding.
        The embedding is a list of floats derived from character codes, not semantically meaningful.
        """
        logging.debug(f"DummyLLMEngine.generate_embedding called for text: '{text[:50]}...'")
        if not text:
            return []
        # Simple deterministic embedding based on char codes, scaled to be between -1 and 1
        # Max ASCII value is around 127. Let's normalize by 128.
        embedding = [(ord(char) % 128) / 128.0 - 0.5 for char in text[:100]] # Limit length for simplicity
        # Ensure a fixed dimension if required by some vector stores, e.g., pad with zeros
        fixed_dim = 128 # Arbitrary fixed dimension
        if len(embedding) < fixed_dim:
            embedding.extend([0.0] * (fixed_dim - len(embedding)))
        else:
            embedding = embedding[:fixed_dim]
        return embedding

# Example Usage (conceptual)
if __name__ == "__main__":
    async def main():
        dummy_engine = DummyLLMEngine()

        # Test response generation
        prompt = "What is the capital of France?"
        context = "France is a country in Europe."
        response = await dummy_engine.generate_response(prompt, context)
        print(f"Response from dummy engine: {response}")

        prompt_no_context = "Tell me a joke."
        response_no_context = await dummy_engine.generate_response(prompt_no_context)
        print(f"Response (no context) from dummy engine: {response_no_context}")

        # Test embedding generation
        text_to_embed = "Hello, world!"
        embedding = await dummy_engine.generate_embedding(text_to_embed)
        print(f"Embedding for '{text_to_embed}': {embedding[:10]}... (length: {len(embedding)})")

        empty_text_embedding = await dummy_engine.generate_embedding("")
        print(f"Embedding for empty text: {empty_text_embedding}")

    # import asyncio # Requires asyncio to run if main uses await
    # asyncio.run(main()) # This won't run directly in the sandbox environment without special handling.
    # For testing this file, you'd typically run it in a local Python environment.
    print("DummyLLMEngine class defined. Run with an async event loop to test main().")
