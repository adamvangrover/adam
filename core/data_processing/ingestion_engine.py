"""
core/data_processing/ingestion_engine.py

Provides a scalable ingestion pipeline for processing diverse data sizes:
- Small (10KB - 10MB): In-Memory Graph
- Medium (100MB - 1GB): Local File Storage + Vector Index
- Large (100GB+): Distributed / Batch Processing (Simulated)

Integration:
- Uses `ChunkingEngine` for RAG preparation.
- Connects to `UnifiedKnowledgeGraph` for final knowledge representation.
"""

import logging
import os
import json
import uuid
from typing import List, Dict, Any, Optional, Protocol
from abc import abstractmethod

from core.data_processing.chunking_engine import ChunkingEngine
from core.engine.unified_knowledge_graph import UnifiedKnowledgeGraph

logger = logging.getLogger(__name__)

class IngestionStrategy(Protocol):
    @abstractmethod
    def ingest(self, data_source: Any, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Ingest data and return a status report."""
        pass

class MemoryIngestionStrategy:
    """
    Ingests data directly into the in-memory UnifiedKnowledgeGraph.
    Best for small, high-value datasets (e.g., 10-Ks, Memos).
    """
    def __init__(self, ukg: UnifiedKnowledgeGraph):
        self.ukg = ukg

    def ingest(self, data_source: Any, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Expects data_source to be a dict or list of dicts compatible with UKG.
        """
        count = 0
        if isinstance(data_source, list):
            # Assume list of company dicts for now
            self.ukg.ingest_financial_data(data_source)
            count = len(data_source)
        elif isinstance(data_source, dict):
            # Assume single entity or graph update
            self.ukg.ingest_financial_data([data_source])
            count = 1

        return {"status": "success", "mode": "memory", "items_ingested": count}

class PersistentIngestionStrategy:
    """
    Ingests data into durable storage (Disk/Vector DB) and updates UKG with references.
    Best for large corpora (e.g., News Archives, SEC Filings).
    """
    def __init__(self, storage_path: str = "data/vector_store", chunking_strategy: str = "recursive"):
        self.storage_path = storage_path
        self.chunker = ChunkingEngine(strategy=chunking_strategy)
        os.makedirs(storage_path, exist_ok=True)

    def ingest(self, data_source: Any, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Expects data_source to be a raw text string or file path.
        """
        text_content = ""
        source_id = str(uuid.uuid4())

        # 1. Load Data
        if isinstance(data_source, str) and os.path.exists(data_source):
            try:
                with open(data_source, 'r', encoding='utf-8') as f:
                    text_content = f.read()
                metadata["source_path"] = data_source
                source_id = os.path.basename(data_source)
            except Exception as e:
                return {"status": "error", "message": f"File read failed: {e}"}
        elif isinstance(data_source, str):
            text_content = data_source
        else:
            return {"status": "error", "message": "Unsupported data format for Persistent Ingestion"}

        # 2. Chunk
        chunks = self.chunker.chunk(text_content, metadata)

        # 3. Embed & Store (Mock Vector Store)
        # In a real system, we would generate embeddings here.
        # For now, we save chunks to JSONL.
        output_file = os.path.join(self.storage_path, f"{source_id}.jsonl")

        with open(output_file, 'w', encoding='utf-8') as f:
            for chunk in chunks:
                # Simulate embedding ID
                chunk["vector_id"] = str(uuid.uuid4())
                f.write(json.dumps(chunk) + "\n")

        return {
            "status": "success",
            "mode": "persistent",
            "chunks_generated": len(chunks),
            "output_path": output_file
        }

class IngestionEngine:
    """
    Facade for data ingestion. auto-selects strategy based on configuration.
    """
    def __init__(self, mode: str = "auto", storage_path: str = "data/ingestion_cache"):
        self.mode = mode
        self.storage_path = storage_path
        self.ukg = UnifiedKnowledgeGraph() # Singleton access

        self.memory_strategy = MemoryIngestionStrategy(self.ukg)
        self.persistent_strategy = PersistentIngestionStrategy(storage_path)

    def ingest(self, data: Any, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Main entry point.
        """
        metadata = metadata or {}

        # Auto-detect strategy
        strategy = self.memory_strategy

        if self.mode == "persistent":
            strategy = self.persistent_strategy
        elif self.mode == "auto":
            # Heuristic: If it's a file path or long text, use persistent.
            # If it's structured data (dict/list), use memory.
            is_file = False
            try:
                if len(data) < 255 and os.path.exists(data):
                    is_file = True
            except:
                pass

            if isinstance(data, str) and (len(data) > 10000 or is_file):
                strategy = self.persistent_strategy
            elif isinstance(data, (dict, list)):
                strategy = self.memory_strategy

        logger.info(f"Ingesting data using {strategy.__class__.__name__}...")
        return strategy.ingest(data, metadata)
