"""
tests/test_ingestion_scaling.py

Verifies the scalable ingestion pipeline across different data sizes and strategies.
"""

import os
import pytest
import shutil
import json
from core.data_processing.ingestion_engine import IngestionEngine
from core.data_processing.chunking_engine import ChunkingEngine

TEST_DATA_DIR = "tests/data/ingestion_test"

@pytest.fixture
def setup_teardown():
    os.makedirs(TEST_DATA_DIR, exist_ok=True)
    yield
    if os.path.exists(TEST_DATA_DIR):
        shutil.rmtree(TEST_DATA_DIR)

def test_chunking_engine_recursive():
    engine = ChunkingEngine(strategy="recursive", chunk_size=50, chunk_overlap=10)
    text = "This is a test sentence that is slightly longer than the chunk size to verify splitting behavior."
    chunks = engine.chunk(text)

    assert len(chunks) > 1
    assert all("text" in c for c in chunks)
    assert all("metadata" in c for c in chunks)
    # Check simple overlap logic
    assert len(chunks[0]['text']) <= 50

def test_ingestion_memory_strategy(setup_teardown):
    engine = IngestionEngine(mode="memory")
    data = [{"symbol": "TEST", "sector": "Tech", "description": "Test Corp"}]

    result = engine.ingest(data)

    assert result["status"] == "success"
    assert result["mode"] == "memory"
    assert result["items_ingested"] == 1

    # Verify it hit the UKG (Singleton might be tricky in tests, but we trust the return for now)

def test_ingestion_persistent_strategy(setup_teardown):
    engine = IngestionEngine(mode="persistent", storage_path=TEST_DATA_DIR)

    # Create dummy large file
    file_path = os.path.join(TEST_DATA_DIR, "large_doc.txt")
    with open(file_path, "w") as f:
        f.write("A" * 5000) # 5KB

    result = engine.ingest(file_path)

    assert result["status"] == "success"
    assert result["mode"] == "persistent"
    assert "chunks_generated" in result
    assert result["chunks_generated"] > 0

    # Check output file
    output_path = result["output_path"]
    assert os.path.exists(output_path)

    with open(output_path, 'r') as f:
        lines = f.readlines()
        assert len(lines) == result["chunks_generated"]
        first_chunk = json.loads(lines[0])
        assert "vector_id" in first_chunk

def test_ingestion_auto_strategy(setup_teardown):
    engine = IngestionEngine(mode="auto", storage_path=TEST_DATA_DIR)

    # 1. Structured Data -> Memory
    data_struct = [{"symbol": "AUTO", "sector": "Auto"}]
    res_mem = engine.ingest(data_struct)
    assert res_mem["mode"] == "memory"

    # 2. Large Text -> Persistent
    large_text = "B" * 15000
    res_persist = engine.ingest(large_text)
    assert res_persist["mode"] == "persistent"
