import logging
from core.utils.repo_compiler.models import Chunk
from core.utils.repo_compiler.formatter import PromptFormatter

logger = logging.getLogger(__name__)

class ChunkSummarizer:
    """
    Optional component to summarize chunks using litellm.
    Fails gracefully if litellm or API keys are unavailable.
    """
    def __init__(self, model_name: str = "gpt-4o-mini"):
        self.model_name = model_name
        self.formatter = PromptFormatter()

        try:
            import litellm
            self.litellm = litellm
            self._available = True
        except ImportError:
            logger.warning("litellm not installed. Summarization will be disabled.")
            self._available = False

    def summarize(self, chunk: Chunk) -> Chunk:
        """
        Attempts to summarize the chunk. Updates the chunk.summary field.
        If unavailable or error occurs, returns the chunk unmodified.
        """
        if not self._available:
            return chunk

        if not chunk.documents:
            return chunk

        prompt_text = self.formatter.format_chunk(chunk)

        # Prevent massive chunks from blowing up the summarizer context
        # Rough estimate: 1 char ~ 0.25 tokens. 120,000 chars ~ 30,000 tokens
        if len(prompt_text) > 200000:
            logger.warning(f"Chunk {chunk.chunk_id} is too large to summarize efficiently. Skipping.")
            chunk.summary = "Chunk too large for automated summary."
            return chunk

        messages = [
            {"role": "system", "content": "You are an expert AI software architect. Analyze the provided codebase files and provide a concise, high-level summary of their purpose, the primary classes/functions, and how they relate to each other."},
            {"role": "user", "content": f"Please summarize this code chunk:\n\n{prompt_text}"}
        ]

        try:
            response = self.litellm.completion(
                model=self.model_name,
                messages=messages,
                max_tokens=500
            )
            summary = response.choices[0].message.content
            chunk.summary = summary
        except Exception as e:
            logger.error(f"Failed to summarize chunk {chunk.chunk_id}: {e}")
            chunk.summary = f"Summary generation failed: {e}"

        return chunk
