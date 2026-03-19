"""
core/data_processing/chunking_engine.py

Provides scalable text chunking strategies for Large Language Model (LLM) RAG pipelines.
Supports:
- Recursive Character Splitting (Context-aware)
- Token-based Splitting (Model-specific)
- Semantic Chunking (Placeholder for future embedding-based segmentation)

Designed for processing datasets ranging from 10KB (Single Doc) to 100GB (Corpus).
"""

import logging
import re
from typing import List, Dict, Any, Optional

try:
    import sentence_transformers
    import numpy as np
    from sklearn.metrics.pairwise import cosine_similarity
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

logger = logging.getLogger(__name__)

class ChunkingEngine:
    """
    Scalable engine for splitting large text documents into manageable chunks
    for vector embedding and knowledge graph ingestion.
    """

    def __init__(self, strategy: str = "recursive", chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        Initialize the Chunking Engine.

        Args:
            strategy: "recursive", "token", or "semantic" (future).
            chunk_size: Target size of each chunk (characters or tokens).
            chunk_overlap: Number of overlapping units between chunks to preserve context.
        """
        self.strategy = strategy
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        if strategy not in ["recursive", "token", "semantic"]:
            raise ValueError(f"Unknown strategy: {strategy}")

        self.model = None
        if self.strategy == "semantic":
            if SENTENCE_TRANSFORMERS_AVAILABLE:
                try:
                    self.model = sentence_transformers.SentenceTransformer('all-MiniLM-L6-v2')
                    logger.info("Initialized semantic chunking model: all-MiniLM-L6-v2")
                except Exception as e:
                    logger.error(f"Failed to load sentence_transformers model: {e}")
            else:
                logger.warning("sentence_transformers not available. Semantic chunking will fallback to recursive.")

    def chunk(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Splits text into chunks based on the configured strategy.

        Args:
            text: The raw text content.
            metadata: Optional metadata to attach to each chunk (e.g., source, page).

        Returns:
            List of dicts: [{"text": "...", "metadata": {...}}, ...]
        """
        if not text:
            return []

        metadata = metadata or {}
        chunks = []

        if self.strategy == "recursive":
            raw_chunks = self._recursive_character_split(text)
        elif self.strategy == "token":
            raw_chunks = self._token_split(text)
        elif self.strategy == "semantic":
            raw_chunks = self._semantic_split(text)
        else:
            raw_chunks = [text]

        # Wrap chunks with metadata
        for i, chunk_text in enumerate(raw_chunks):
            chunk_meta = metadata.copy()
            chunk_meta.update({
                "chunk_index": i,
                "chunk_strategy": self.strategy,
                "total_chunks": len(raw_chunks)
            })
            chunks.append({
                "text": chunk_text,
                "metadata": chunk_meta
            })

        return chunks

    def _recursive_character_split(self, text: str) -> List[str]:
        """
        Splits text recursively by separators (paragraphs, newlines, spaces)
        to keep related text together.
        """
        separators = ["\n\n", "\n", " ", ""]
        final_chunks = []

        # Initial split
        self._split_text(text, separators, final_chunks)

        return final_chunks

    def _semantic_split(self, text: str) -> List[str]:
        """
        Splits text into chunks using sentence embeddings to detect topic boundaries.
        """
        if not self.model:
            return self._recursive_character_split(text)

        # Basic sentence splitting (naive regex for speed and simplicity)
        sentences = re.split(r'(?<=[.!?])\s+', text)
        sentences = [s.strip() for s in sentences if s.strip()]

        if not sentences:
            return []

        if len(sentences) == 1:
            return self._recursive_character_split(text)

        try:
            # Embed all sentences
            embeddings = self.model.encode(sentences)

            # Calculate cosine similarities between adjacent sentences
            similarities = []
            for i in range(len(embeddings) - 1):
                sim = cosine_similarity([embeddings[i]], [embeddings[i+1]])[0][0]
                similarities.append(sim)

            chunks = []
            current_chunk = []
            current_length = 0

            # Threshold for splitting (can be dynamic, but static is fine for now)
            threshold = 0.4

            for i, sentence in enumerate(sentences):
                sentence_len = len(sentence) + 1 # +1 for space

                # Check length limit first
                if current_length + sentence_len > self.chunk_size and current_chunk:
                    chunks.append(" ".join(current_chunk))
                    current_chunk = [sentence]
                    current_length = sentence_len
                else:
                    current_chunk.append(sentence)
                    current_length += sentence_len

                # If similarity drops, it's a boundary
                if i < len(similarities) and similarities[i] < threshold and current_chunk:
                    # Flush chunk
                    chunks.append(" ".join(current_chunk))
                    current_chunk = []
                    current_length = 0

            if current_chunk:
                chunks.append(" ".join(current_chunk))

            # If any chunk is still over the limit, recursively split it
            final_chunks = []
            for chunk in chunks:
                if len(chunk) > self.chunk_size:
                    final_chunks.extend(self._recursive_character_split(chunk))
                else:
                    final_chunks.append(chunk)

            return final_chunks

        except Exception as e:
            logger.error(f"Semantic split failed: {e}. Falling back to recursive.")
            return self._recursive_character_split(text)

    def _split_text(self, text: str, separators: List[str], final_chunks: List[str]):
        """
        Recursive helper for text splitting.
        """
        if len(text) <= self.chunk_size:
            final_chunks.append(text)
            return

        # Find the best separator
        separator = separators[-1]
        for sep in separators:
            if sep == "":
                separator = ""
                break
            if sep in text:
                separator = sep
                break

        # Split
        if separator:
            splits = text.split(separator)
        else:
            # Fallback: Character split
            splits = list(text)

        # Re-merge small splits into chunks
        current_chunk = []
        current_length = 0

        for split in splits:
            split_len = len(split) + len(separator)

            if current_length + split_len > self.chunk_size:
                # Flush current chunk
                if current_chunk:
                    chunk_text = separator.join(current_chunk)
                    final_chunks.append(chunk_text)

                    # Handle overlap (simple approach: keep last item)
                    # For a robust implementation, we'd keep enough previous items to meet chunk_overlap
                    overlap_item = current_chunk[-1] if self.chunk_overlap > 0 else None
                    current_chunk = [overlap_item] if overlap_item else []
                    current_length = len(overlap_item) + len(separator) if overlap_item else 0
                else:
                    # Single split is too big, recurse if possible
                    if separator:
                        # Find next separator index
                        next_sep_idx = separators.index(separator) + 1
                        if next_sep_idx < len(separators):
                            self._split_text(split, separators[next_sep_idx:], final_chunks)
                        else:
                            # Hard cut
                            final_chunks.append(split[:self.chunk_size])
                    else:
                         final_chunks.append(split[:self.chunk_size])

            current_chunk.append(split)
            current_length += split_len

        # Flush remaining
        if current_chunk:
            final_chunks.append(separator.join(current_chunk))

    def _token_split(self, text: str) -> List[str]:
        """
        Approximate token splitting (1 token ~= 4 chars).
        Real implementation would use tiktoken/transformers.
        """
        # Simple whitespace tokenization for simulation
        words = text.split()
        chunks = []
        current_chunk = []
        current_count = 0

        target_tokens = self.chunk_size  # Treating chunk_size as token count here

        for word in words:
            # Rough approximation: word is 1 token
            if current_count + 1 > target_tokens:
                chunks.append(" ".join(current_chunk))

                # Overlap
                overlap_count = int(self.chunk_overlap)
                current_chunk = current_chunk[-overlap_count:] if overlap_count > 0 else []
                current_count = len(current_chunk)

            current_chunk.append(word)
            current_count += 1

        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return chunks
