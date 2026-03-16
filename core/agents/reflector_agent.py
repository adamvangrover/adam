from __future__ import annotations

import logging
from typing import Any, Dict

from core.agents.agent_base import AgentBase
from core.engine.states import init_reflector_state
from core.schemas.agent_schema import AgentInput, AgentOutput

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

    async def execute(self, content_to_analyze: AgentInput, context: Dict[str, Any] = None) -> AgentOutput:
        """
        Analyzes the provided content (reasoning trace, report, etc.) and provides a critique.
        """
        text = content_to_analyze.query
        ctx = content_to_analyze.context

        # Backward compatibility for content_to_analyze name within function
        content_to_analyze = text

        logger.info("ReflectorAgent: Analyzing content for self-correction...")

        # --- v23 Path: Reflector Graph ---
        if GRAPH_AVAILABLE and reflector_app:
            logging.info("ReflectorAgent: Delegating to v23 ReflectorGraph.")

            initial_state = init_reflector_state(text, ctx)
            config = {"configurable": {"thread_id": "1"}}

            try:
                if hasattr(reflector_app, 'ainvoke'):
                    result = await reflector_app.ainvoke(initial_state, config=config)
                else:
                    result = reflector_app.invoke(initial_state, config=config)

                return AgentOutput(
                    answer=result.get("refined_content", ""),
                    sources=[],
                    confidence=float(result.get("score", 0.0)),
                    metadata={
                        "original_content_snippet": text[:100] + "...",
                        "critique_notes": result.get("critique_notes", []),
                        "verification_status": "PASS" if result.get("is_valid") else "NEEDS_REVISION"
                    }
                )
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

        return AgentOutput(
            answer="\n".join(critique) if critique else "No issues found.",
            sources=[],
            confidence=float(score) / 10.0,
            metadata={
                "original_content_snippet": text[:100] + "...",
                "critique_notes": critique,
                "verification_status": "PASS" if score > 7.0 else "NEEDS_REVISION"
            }
        )
