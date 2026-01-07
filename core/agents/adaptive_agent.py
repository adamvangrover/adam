from core.agents.agent_base import AgentBase
from core.agents.metacognition import AdaptiveConvictionMixin, StateAnchorMixin, ToolRAGMixin
from typing import Dict, Any, Optional
import logging

class AdaptiveAgent(AgentBase, AdaptiveConvictionMixin, StateAnchorMixin, ToolRAGMixin):
    """
    An agent implementation that fully embodies the 'Protocol Paradox' resolutions:
    1. Adaptive Conviction (Switching between Direct/MCP)
    2. State Anchors (Async Drift protection)
    3. Tool RAG (Context Saturation mitigation)
    """

    def __init__(self, config: Dict[str, Any], constitution: Optional[Dict[str, Any]] = None, kernel: Any = None):
        # Initialize Base
        super().__init__(config, constitution, kernel)

        # Initialize Mixins
        AdaptiveConvictionMixin.__init__(self)
        StateAnchorMixin.__init__(self)
        ToolRAGMixin.__init__(self)

    async def execute(self, *args: Any, **kwargs: Any) -> Any:
        """
        Overridden execute method that applies the Adaptive Logic.
        """
        # 1. Extract Task Description
        task_description = kwargs.get("input_text") or (args[0] if args else "")
        if not task_description:
            return {"error": "No input provided"}

        # 2. Tool RAG: Load relevant tools only
        self.load_tools_jit(str(task_description))

        # 3. Conviction Check & Mode Switching
        mode, meta = self.get_interaction_strategy(str(task_description))
        logging.info(f"AdaptiveAgent Strategy: {mode} (Conviction: {meta['conviction']:.2f}, Complexity: {meta['complexity']:.2f})")

        if mode == "DIRECT":
            return self._execute_direct_prompt(task_description)
        else:
            return await self._execute_mcp_workflow(task_description)

    def _execute_direct_prompt(self, task: str) -> Dict[str, Any]:
        """
        Simulated 'System 1' Fast execution path.
        """
        # In a real system, this would just call the LLM with a simple prompt
        return self.generate_metacognitive_response(
            content=f"[Direct Prompt Result] Processed '{task}' via natural language path.",
            conviction=1.0 # Direct path assumes high confidence in simplicity
        )

    async def _execute_mcp_workflow(self, task: str) -> Dict[str, Any]:
        """
        Simulated 'System 2' Structured execution path.
        """
        # 1. Create State Anchor before async/complex work
        anchor_id = self.create_anchor(label=f"pre_task_{task[:10]}")

        # 2. Simulate Async Work (e.g., calling tools)
        # In reality: await self.call_mcp_tool(...)
        import asyncio
        await asyncio.sleep(0.1)

        # 3. Verify Drift
        if not self.verify_anchor(anchor_id):
            logging.warning("State Drift detected during MCP workflow! Re-evaluating...")
            # Self-correction logic would go here
            pass

        return self.generate_metacognitive_response(
            content=f"[MCP Result] Processed '{task}' via structured schema path.",
            conviction=self._conviction_history[-1] if self._conviction_history else 0.9
        )
