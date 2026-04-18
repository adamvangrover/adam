from typing import List, Optional
from pydantic import BaseModel, Field

class FileDocument(BaseModel):
    """
    Represents a single file in the repository.
    """
    path: str = Field(..., description="The path of the file relative to the repo root.")
    content: str = Field(..., description="The content of the file.")
    size: int = Field(..., description="The size of the file in bytes.")
    lines: int = Field(..., description="The number of lines in the file.")
    extension: str = Field(..., description="The file extension.")

class Chunk(BaseModel):
    """
    Represents a chunk of documents, typically grouped by directory or size.
    """
    chunk_id: str = Field(..., description="A unique identifier for the chunk.")
    documents: List[FileDocument] = Field(default_factory=list, description="The files contained in this chunk.")
    total_size: int = Field(0, description="The total size of the chunk in bytes.")
    summary: Optional[str] = Field(None, description="An optional summary of the chunk's contents.")

    def add_document(self, doc: FileDocument):
        self.documents.append(doc)
        self.total_size += doc.size
