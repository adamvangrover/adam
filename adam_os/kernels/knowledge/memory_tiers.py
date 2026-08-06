from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from pydantic import BaseModel
from datetime import datetime

class TransactionalRecord(BaseModel):
    """Represents a fact in PostgreSQL: What is true?"""
    entity_id: str
    fact_key: str
    fact_value: Any
    updated_at: datetime

class SemanticVector(BaseModel):
    """Represents an embedding in Qdrant: What is similar?"""
    vector_id: str
    embedding: List[float]
    metadata: Dict[str, Any]

class RelationalEdge(BaseModel):
    """Represents a connection in Knowledge Graph: What is connected?"""
    source_node_id: str
    target_node_id: str
    relationship_type: str
    properties: Dict[str, Any]

class TemporalEvent(BaseModel):
    """Represents an event in the Event Store: What happened?"""
    event_id: str
    aggregate_id: str
    event_type: str
    timestamp: datetime
    payload: Dict[str, Any]

class TransactionalMemoryInterface(ABC):
    @abstractmethod
    async def get_fact(self, entity_id: str, fact_key: str) -> Optional[TransactionalRecord]:
        pass

    @abstractmethod
    async def set_fact(self, record: TransactionalRecord) -> None:
        pass

class SemanticMemoryInterface(ABC):
    @abstractmethod
    async def search_similar(self, embedding: List[float], limit: int = 5) -> List[SemanticVector]:
        pass

    @abstractmethod
    async def upsert_vector(self, vector: SemanticVector) -> None:
        pass

class RelationalMemoryInterface(ABC):
    @abstractmethod
    async def get_connections(self, node_id: str, relationship_type: Optional[str] = None) -> List[RelationalEdge]:
        pass

    @abstractmethod
    async def add_connection(self, edge: RelationalEdge) -> None:
        pass

class TemporalMemoryInterface(ABC):
    @abstractmethod
    async def get_history(self, aggregate_id: str) -> List[TemporalEvent]:
        pass

    @abstractmethod
    async def append_event(self, event: TemporalEvent) -> None:
        pass
