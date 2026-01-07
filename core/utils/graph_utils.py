
"""
Graph Utilities for Adam v23.5

This module provides a unified interface for LangGraph components,
handling fallback logic for environments where langgraph is not installed.
"""

import logging

logger = logging.getLogger(__name__)

try:
    from langgraph.graph import StateGraph, END, START
    from langgraph.checkpoint.memory import MemorySaver
    HAS_LANGGRAPH = True
except ImportError:
    HAS_LANGGRAPH = False
    logger.warning("LangGraph not installed. Graph features will be disabled or mocked.")

    class StateGraph:
        """Mock StateGraph for environments without langgraph."""
        def __init__(self, state_schema, *args, **kwargs):
            self.state_schema = state_schema

        def add_node(self, node_name, action):
            pass

        def add_edge(self, start_node, end_node):
            pass

        def set_entry_point(self, node_name):
            pass

        def add_conditional_edges(self, source, path, path_map=None):
            pass

        def compile(self, checkpointer=None):
            return CompiledGraphMock()

    class CompiledGraphMock:
        """Mock for a compiled graph."""
        def invoke(self, inputs, config=None):
            logger.info(f"Mock graph invoked with inputs: {inputs}")
            return inputs

    class MemorySaver:
        """Mock MemorySaver."""
        pass

    END = "END"
    START = "START"
