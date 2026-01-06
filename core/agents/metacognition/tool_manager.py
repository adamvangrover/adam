# core/agents/metacognition/tool_manager.py
from typing import List, Dict, Any, Optional
import logging

class ToolRAGMixin:
    """
    Implements Hierarchical Discovery and Tool RAG.
    Mitigates 'Context Saturation' by only loading relevant tools.
    Section 7.2 of the Protocol Paradox.
    """

    def __init__(self):
        # Master registry of all available tools (Metadata only)
        self._tool_registry_index: List[Dict[str, str]] = []
        self._active_tools: Dict[str, Any] = {}

    def register_tool_metadata(self, name: str, description: str, tool_callable: Any):
        """
        Registers a tool in the index without loading its full schema into context.
        """
        self._tool_registry_index.append({
            "name": name,
            "description": description,
            "ref": tool_callable
        })

    def retrieve_relevant_tools(self, query: str, limit: int = 3) -> List[Any]:
        """
        Retrieves the top-K relevant tools for a given query.
        Current implementation: Simple Keyword Search (Simulating Vector Search).
        """
        scored_tools = []

        query_terms = set(query.lower().split())

        for tool in self._tool_registry_index:
            score = 0
            desc_terms = set(tool["description"].lower().split())
            name_terms = set(tool["name"].lower().split())

            # Intersection count
            score += len(query_terms.intersection(desc_terms))
            score += len(query_terms.intersection(name_terms)) * 2 # Names weight more

            scored_tools.append((score, tool))

        # Sort desc
        scored_tools.sort(key=lambda x: x[0], reverse=True)

        # Select top K
        selected = [t[1] for t in scored_tools[:limit] if t[0] > 0]

        # Fallback: if no matches, return a default set or empty
        if not selected and self._tool_registry_index:
            # Maybe return a generic "search" tool if available
            pass

        logging.info(f"Tool RAG: Retrieved {len(selected)} tools for query '{query[:20]}...'")
        return selected

    def load_tools_jit(self, query: str):
        """
        Just-In-Time loading of tools into the active context/kernel.
        """
        relevant = self.retrieve_relevant_tools(query)

        # Clear current active tools to free context (Simulated)
        self._active_tools = {}

        for tool_meta in relevant:
            name = tool_meta["name"]
            func = tool_meta["ref"]
            self._active_tools[name] = func

            # If using Semantic Kernel, we would import the skill here
            if hasattr(self, 'kernel') and self.kernel:
                # self.kernel.import_skill(func, name)
                pass

        return self._active_tools
