import logging
from typing import List, Optional

class Document:
    """
    Represents a document to be processed and ingested into the RAG system.
    """
    def __init__(self, content: str, id: Optional[str] = None, metadata: Optional[dict] = None):
        self.id = id or str(hash(content)) # Simple ID generation if not provided
        self.content = content
        self.metadata = metadata or {}

    def __repr__(self) -> str:
        return f"Document(id='{self.id}', metadata={self.metadata}, content_length={len(self.content)})"

def chunk_text(text: str, chunk_size: int, chunk_overlap: int = 0) -> List[str]:
    """
    Simple text chunking function.
    Splits text into chunks of specified size with optional overlap.
    """
    if chunk_size <= 0:
        raise ValueError("Chunk size must be positive.")
    if chunk_overlap < 0:
        raise ValueError("Chunk overlap cannot be negative.")
    if chunk_overlap >= chunk_size:
        raise ValueError("Chunk overlap should be less than chunk size.")

    if not text:
        return []

    chunks = []
    idx = 0
    text_length = len(text)

    while idx < text_length:
        end_idx = idx + chunk_size
        chunks.append(text[idx:end_idx])
        idx += (chunk_size - chunk_overlap)
        if idx >= text_length and end_idx < text_length: # Ensure last piece is captured if overlap causes step over
             # This condition might be tricky. The goal is to avoid missing the tail end of the text.
             # If the step `chunk_size - chunk_overlap` is small, this might not be an issue.
             # A simpler way is to just ensure the last chunk goes to text_length if it's the final iteration.
             # The loop condition `idx < text_length` and `text[idx:end_idx]` should handle it mostly.
             pass # Current loop structure should handle this by taking text[idx:text_length] in the last iteration.

    # Final check: if the last chunk taken was short and there's more text,
    # or if the loop terminates slightly early due to overlap math.
    # This is more of a safeguard for specific overlap/size math.
    # A well-behaved `idx += (chunk_size - chunk_overlap)` should make the `while idx < text_length` sufficient.

    # Let's simplify the loop to be more standard:
    chunks = []
    current_position = 0
    while current_position < len(text):
        end_position = current_position + chunk_size
        chunk = text[current_position:end_position]
        chunks.append(chunk)

        next_start_position = current_position + chunk_size - chunk_overlap
        if next_start_position <= current_position : # Avoid infinite loop if step is not positive
            logging.warning(f"Chunking step is not positive ({chunk_size - chunk_overlap}). Advancing by 1 to prevent loop.")
            current_position += 1
        elif next_start_position >= len(text): # If next step is beyond text, we are done
            break
        else:
            current_position = next_start_position

    return [c for c in chunks if c] # Ensure no empty string chunks from edge cases

# Example Usage:
if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    sample_text_short = "This is a test text."
    sample_text_long = "This is a very long piece of text that is intended to be split into multiple chunks for processing by a large language model. We need to ensure that the chunking mechanism works correctly, including the overlap between consecutive chunks. This helps maintain context across boundaries." * 3

    chunk_size = 50
    chunk_overlap = 10

    print(f"\n--- Short Text (size: {chunk_size}, overlap: {chunk_overlap}) ---")
    chunks_short_test = chunk_text(sample_text_short, chunk_size, chunk_overlap)
    for i, chunk in enumerate(chunks_short_test):
        print(f"Chunk {i+1} (len {len(chunk)}): '{chunk}'")

    print(f"\n--- Long Text (size: {chunk_size}, overlap: {chunk_overlap}) ---")
    chunks_long_test = chunk_text(sample_text_long, chunk_size, chunk_overlap)
    for i, chunk in enumerate(chunks_long_test):
        print(f"Chunk {i+1} (len {len(chunk)}): '{chunk}'")

    print(f"\n--- Edge Case: No Overlap (size: {chunk_size}, overlap: 0) ---")
    chunks_no_overlap = chunk_text(sample_text_long, chunk_size, 0)
    for i, chunk in enumerate(chunks_no_overlap):
        print(f"Chunk {i+1} (len {len(chunk)}): '{chunk}'")

    print(f"\n--- Edge Case: Large Overlap (size: {chunk_size}, overlap: 40) ---")
    chunks_large_overlap = chunk_text(sample_text_long, chunk_size, 40)
    for i, chunk in enumerate(chunks_large_overlap):
        print(f"Chunk {i+1} (len {len(chunk)}): '{chunk}'")

    print(f"\n--- Edge Case: Chunk size equals overlap (should be prevented by checks) ---")
    try:
        chunk_text(sample_text_long, 50, 50)
    except ValueError as e:
        print(f"Caught expected error: {e}")

    print(f"\n--- Edge Case: Empty Text ---")
    chunks_empty = chunk_text("", chunk_size, chunk_overlap)
    print(f"Chunks from empty: {chunks_empty}")

    doc = Document(content=sample_text_long, id="sample_doc_123", metadata={"source": "example_long"})
    print(f"\nDocument object: {doc}")
