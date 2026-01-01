# core/v23_graph_engine/unified_knowledge_graph.py

"""
DEPRECATED: This module is a legacy wrapper.
It now aliases the active UnifiedKnowledgeGraph in core/engine/unified_knowledge_graph.py
to ensure a shared singleton instance across the application.

Optimization: ⚡ Bolt (2026-01-01) - Unified Singleton Identity
"""

import logging
from core.engine.unified_knowledge_graph import UnifiedKnowledgeGraph as EngineUKG

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class UnifiedKnowledgeGraph(EngineUKG):
    """
    Deprecated: Use core.engine.unified_knowledge_graph.UnifiedKnowledgeGraph.

    This class inherits from the core engine's UnifiedKnowledgeGraph to ensure
    that the same underlying graph singleton (_SHARED_GRAPH_INSTANCE in core.engine)
    is used regardless of which import path is chosen.
    """
    def __init__(self):
        # The parent __init__ uses the _SHARED_GRAPH_INSTANCE defined in
        # core.engine.unified_knowledge_graph, ensuring both classes share
        # the same graph memory.
        super().__init__()
