from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field
from uuid import UUID, uuid4

class MemoryDocument(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    payload: Dict[str, Any] = Field(..., description="Document content and metadata")
    vector: Optional[List[float]] = Field(default=None, description="Embedding vector")
    tenant_id: str = Field(default="system", description="Isolation context")

class MemoryQuery(BaseModel):
    query: str = Field(..., description="Search query string")
    context_keys: List[str] = Field(default_factory=list, description="Keys to filter search")
    top_k: int = Field(default=5)
    tenant_id: str = Field(default="system", description="Isolation context")

class SearchResult(BaseModel):
    id: UUID
    score: float = Field(..., description="Similarity score")
    payload: Dict[str, Any]
