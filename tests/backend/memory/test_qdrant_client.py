import pytest
from src.backend.memory.qdrant_client import JitMemoryClient
from src.shared.models.memory_models import MemoryDocument, MemoryQuery

@pytest.mark.asyncio
async def test_upsert_document():
    client = JitMemoryClient()
    doc = MemoryDocument(payload={"text": "test document"})
    result = await client.upsert_document(doc)
    assert result is True

@pytest.mark.asyncio
async def test_search_context():
    client = JitMemoryClient()
    query = MemoryQuery(query="test", context_keys=["doc1"])
    results = await client.search_context(query)
    assert len(results) == 1
    assert results[0].score == 0.95
    assert results[0].payload == {"retrieved_info": "data for doc1"}
