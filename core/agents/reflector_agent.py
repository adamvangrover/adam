from __future__ import annotations
from typing import Dict, Any, List, Union
import logging
from core.agents.agent_base import AgentBase
from core.engine.states import init_reflector_state

logger = logging.getLogger(__name__)

# Try to import v23 Graph
try:
    from core.engine.reflector_graph import reflector_app
    GRAPH_AVAILABLE = True
except ImportError:
    GRAPH_AVAILABLE = False
    reflector_app = None


class ReflectorAgent(AgentBase):
    """
    The Reflector Agent performs meta-cognition.
    It analyzes the output of other agents or the system's own reasoning traces
    to identify logical fallacies, hallucination risks, or missing context.

    v23 Update: Wraps `ReflectorGraph` for iterative self-correction.
    """

    async def execute(self, content_to_analyze: Union[str, Dict[str, Any]], context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Analyzes the provided content (reasoning trace, report, etc.) and provides a critique.
        """
        if isinstance(content_to_analyze, dict):
             content_to_analyze = content_to_analyze.get("content") or content_to_analyze.get("payload") or str(content_to_analyze)

        logger.info("ReflectorAgent: Analyzing content for self-correction...")

        # --- v23 Path: Reflector Graph ---
        if GRAPH_AVAILABLE and reflector_app:
            logging.info("ReflectorAgent: Delegating to v23 ReflectorGraph.")

            initial_state = init_reflector_state(content_to_analyze, context)
            config = {"configurable": {"thread_id": "1"}}

            try:
                if hasattr(reflector_app, 'ainvoke'):
                    result = await reflector_app.ainvoke(initial_state, config=config)
                else:
                    result = reflector_app.invoke(initial_state, config=config)

                return {
                    "original_content_snippet": content_to_analyze[:100] + "...",
                    "critique_notes": result.get("critique_notes", []),
                    "quality_score": result.get("score", 0.0) * 10,  # Scale 0-1 to 0-10
                    "verification_status": "PASS" if result.get("is_valid") else "NEEDS_REVISION",
                    "refined_content": result.get("refined_content")
                }
            except Exception as e:
                logging.error(f"ReflectorGraph execution failed: {e}. Falling back to heuristics.")

        # --- v21 Path: Heuristics ---
        critique = []
        score = 10.0

        # Heuristic 1: Absolutism Check
        if "always" in content_to_analyze.lower() or "never" in content_to_analyze.lower():
            critique.append("Detected potential absolutism (always/never). Verify if exceptions exist.")
            score -= 1.5

        # Heuristic 2: Depth Check
        if len(content_to_analyze) < 100:
            critique.append("Content seems too brief for a detailed analysis.")
            score -= 2.0

        # Heuristic 3: Source Citation (Mock)
        if "source" not in content_to_analyze.lower() and "according to" not in content_to_analyze.lower():
            critique.append("No explicit sources cited. Risk of hallucination.")
            score -= 1.5

        return {
            "original_content_snippet": content_to_analyze[:100] + "...",
            "critique_notes": critique,
            "quality_score": score,
            "verification_status": "PASS" if score > 7.0 else "NEEDS_REVISION"
        }
