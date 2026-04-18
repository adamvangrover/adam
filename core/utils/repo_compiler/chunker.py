import os
from typing import List, Dict
import tiktoken
from core.utils.repo_compiler.models import FileDocument, Chunk

class RepoChunker:
    """
    Groups FileDocuments into manageable Chunks.
    """
    def __init__(self, model_name: str = "gpt-4"):
        self.encoder = tiktoken.encoding_for_model(model_name)

    def _estimate_tokens(self, text: str) -> int:
        """Estimates the token count for a given text."""
        try:
            return len(self.encoder.encode(text))
        except Exception:
            # Fallback naive estimation
            return len(text) // 4

    def chunk_by_token_limit(self, documents: List[FileDocument], max_tokens: int = 100000) -> List[Chunk]:
        """
        Groups documents into chunks such that each chunk stays under the token limit.
        If a single file exceeds the limit, it will be placed in its own chunk.
        """
        chunks = []
        current_chunk = Chunk(chunk_id=f"chunk_{len(chunks)+1}")
        current_tokens = 0

        for doc in documents:
            doc_tokens = self._estimate_tokens(doc.content)

            # If the single document exceeds the limit, and the current chunk is empty, add it
            if doc_tokens > max_tokens:
                if len(current_chunk.documents) > 0:
                    chunks.append(current_chunk)
                    current_chunk = Chunk(chunk_id=f"chunk_{len(chunks)+1}")
                    current_tokens = 0

                single_file_chunk = Chunk(chunk_id=f"chunk_{len(chunks)+1}")
                single_file_chunk.add_document(doc)
                chunks.append(single_file_chunk)
                current_chunk = Chunk(chunk_id=f"chunk_{len(chunks)+1}")
                continue

            if current_tokens + doc_tokens > max_tokens:
                chunks.append(current_chunk)
                current_chunk = Chunk(chunk_id=f"chunk_{len(chunks)+1}")
                current_tokens = 0

            current_chunk.add_document(doc)
            current_tokens += doc_tokens

        if len(current_chunk.documents) > 0:
            chunks.append(current_chunk)

        return chunks

    def chunk_by_directory(self, documents: List[FileDocument]) -> List[Chunk]:
        """
        Groups documents by their top-level directory.
        Files in the root are grouped into a 'root' chunk.
        """
        dir_map: Dict[str, Chunk] = {}

        for doc in documents:
            parts = doc.path.split(os.sep)

            # Determine top-level dir
            if len(parts) == 1:
                top_dir = "root"
            else:
                top_dir = parts[0]

            if top_dir not in dir_map:
                dir_map[top_dir] = Chunk(chunk_id=top_dir)

            dir_map[top_dir].add_document(doc)

        return list(dir_map.values())
