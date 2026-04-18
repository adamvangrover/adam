from typing import List
from core.utils.repo_compiler.models import FileDocument, Chunk

class PromptFormatter:
    """
    Formats parsed chunks and documents into text formats suitable for LLMs.
    """

    @staticmethod
    def format_document_markdown(doc: FileDocument) -> str:
        """Formats a single document with markdown code blocks."""
        ext = doc.extension.lstrip('.')
        # Optional: map extensions to standard markdown syntax highlighting names
        lang = ext if ext else "text"

        return f"### File: `{doc.path}`\n```{lang}\n{doc.content}\n```\n"

    @staticmethod
    def format_document_xml(doc: FileDocument) -> str:
        """Formats a single document using XML tags."""
        return f'<file path="{doc.path}">\n{doc.content}\n</file>\n'

    def format_chunk(self, chunk: Chunk, format_type: str = "markdown") -> str:
        """Formats all documents in a chunk."""
        output = [f"## Chunk: {chunk.chunk_id}"]

        if chunk.summary:
            output.append(f"### Summary\n{chunk.summary}\n")

        for doc in chunk.documents:
            if format_type == "xml":
                output.append(self.format_document_xml(doc))
            else:
                output.append(self.format_document_markdown(doc))

        return "\n".join(output)

    def format_monolith(self, documents: List[FileDocument], format_type: str = "markdown", system_prompt: str = "") -> str:
        """Formats a list of documents into a single massive string."""
        output = []
        if system_prompt:
            output.append(system_prompt)
            output.append("\n---\n")

        output.append("# Repository Source Code\n")

        for doc in documents:
            if format_type == "xml":
                output.append(self.format_document_xml(doc))
            else:
                output.append(self.format_document_markdown(doc))

        return "\n".join(output)
