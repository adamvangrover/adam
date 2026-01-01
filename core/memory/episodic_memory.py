import uuid
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
from pydantic import BaseModel, Field

from core.memory.vector_ops import VectorStore, VertexVectorStore
from core.llm_plugin import LLMPlugin

logger = logging.getLogger(__name__)

class Episode(BaseModel):
    """
    Represents a discrete unit of memory (e.g., a specific analysis session).
    """
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = Field(default_factory=datetime.now)
    content: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    embedding: Optional[List[float]] = None

class EpisodicMemory:
    """
    Manages long-term "Episodic" memory for agents.
    Uses an LLM to generate embeddings and a VectorStore to persist/search them.
    """

    def __init__(self, vector_store: VectorStore, llm_plugin: LLMPlugin):
        self.vector_store = vector_store
        self.llm = llm_plugin

    def remember(self, content: str, metadata: Dict[str, Any] = None):
        """
        Stores a new memory.
        """
        metadata = metadata or {}

        # 1. Generate Embedding
        try:
            embedding = self.llm.get_embedding(content)
        except Exception as e:
            logger.error(f"Failed to generate embedding for memory: {e}")
            return

        # 2. Create Episode
        episode = Episode(content=content, metadata=metadata, embedding=embedding)

        # 3. Store in Vector DB
        self.vector_store.add(
            vectors=[embedding],
            metadata=[{**metadata, "content": content, "timestamp": episode.timestamp.isoformat()}],
            ids=[episode.id]
        )
        logger.info(f"Stored memory episode: {episode.id}")
        return episode.id

    def recall(self, query: str, k: int = 3) -> List[Episode]:
        """
        Retrieves relevant past memories based on a query.
        """
        # 1. Embed query
        try:
            query_vec = self.llm.get_embedding(query)
        except Exception as e:
            logger.error(f"Failed to embed query: {e}")
            return []

        # 2. Search Vector DB
        results = self.vector_store.search(query_vec, k=k)

        # 3. Reconstruct Episodes
        episodes = []
        for res in results:
            meta = res.get("metadata", {})
            ep = Episode(
                id=res.get("id"),
                content=meta.get("content", ""),
                metadata=meta,
                timestamp=datetime.fromisoformat(meta.get("timestamp", datetime.now().isoformat()))
            )
            episodes.append(ep)

        return episodes

# --- Factory ---
def get_episodic_memory() -> EpisodicMemory:
    """
    Factory to create a Vertex-backed episodic memory system.
    """
    # In a real app, these would come from config
    store = VertexVectorStore(
        project_id="adam-v24",
        index_endpoint_name="memory-endpoint",
        deployed_index_id="main-memory-index"
    )
    llm = LLMPlugin()
    return EpisodicMemory(store, llm)
