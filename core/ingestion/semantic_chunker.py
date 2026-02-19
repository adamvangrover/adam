from typing import List

class SemanticChunker:
    """
    Splits text using heuristic boundaries (paragraphs, newlines) rather than fixed character counts.
    """
    def __init__(self):
        pass

    def split_text(self, text: str) -> List[str]:
        if not text:
            return []

        # Split by double newlines to simulate paragraph chunking
        # Also handle potential variations like \r\n\r\n
        chunks = [chunk.strip() for chunk in text.replace('\r\n', '\n').split('\n\n') if chunk.strip()]

        if not chunks:
            # Fallback for single block of text or text without double newlines
            return [text.strip()]

        return chunks
