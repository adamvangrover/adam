# core/v23_graph_engine/unified_knowledge_graph.py

"""
Manages the integration of the FIBO domain ontology and the W3C PROV-O provenance ontology.

Legacy wrapper for core.engine.unified_knowledge_graph.

**DEPRECATED**: Use `core.engine.unified_knowledge_graph` instead.
This file is preserved for backward compatibility and now aliases the new engine implementation
to share the same Singleton graph instance.
"""

import logging
from core.engine.unified_knowledge_graph import UnifiedKnowledgeGraph as CoreUnifiedKnowledgeGraph

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Alias the class to point to the core engine implementation
UnifiedKnowledgeGraph = CoreUnifiedKnowledgeGraph

# Deprecation warning on import
logger.warning("core.v23_graph_engine.unified_knowledge_graph is deprecated. "
               "Using core.engine.unified_knowledge_graph under the hood.")
