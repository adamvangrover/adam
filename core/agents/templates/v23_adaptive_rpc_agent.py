import logging
import json
from typing import Dict, Any, List, Optional, Union
from core.system.v22_async.async_agent_base import AsyncAgentBase
from core.system.v22_async.async_task import AsyncTask

logger = logging.getLogger(__name__)

class AdaptiveRPCAgent(AsyncAgentBase):
    """
    V23.5 'Apex' Agent with Metacognitive Gating.

    Implements the 'Protocol Paradox' resolution:
    1. JSON-RPC 2.0 Native: Speaks standard MCP.
    2. Heuristic 1 (Ambiguity Guardrail): Reverts to text if conviction is low.
    3. Heuristic 2 (Context Budgeting): Just-in-Time tool loading.
    """

    def __init__(self, agent_id: str, config: Dict[str, Any], system_context: Any):
        super().__init__(agent_id, config, system_context)
        self.tools = config.get("tools", [])
        self.conviction_threshold = config.get("conviction_threshold", 0.85)
        self.max_tool_context_tokens = 2000  # Budget for Heuristic 2

    async def execute_task(self, task: AsyncTask) -> Dict[str, Any]:
        logger.info(f"Adaptive Agent {self.agent_id} engaging task: {task.task_id}")

        try:
            # --- HEURISTIC 2: Context Budgeting ---
            # Don't load all tools. Filter schemas based on task semantic relevance.
            # (In production, use vector similarity here).
            active_schemas = self._filter_tools_by_budget(task.input_data)

            # Construct Prompt with strict Protocol instructions
            prompt = self._construct_protocol_prompt(task.input_data, active_schemas)

            # --- THINKING PHASE (Metacognition) ---
            # We ask for a 'thought' trace before the JSON to measure conviction.
            response = await self.llm_engine.generate(
                prompt=prompt,
                model=self.config.get("model", "gpt-4-turbo"),
                temperature=0.2, # Low temp for protocol adherence
                response_format={"type": "json_object"}
            )

            # --- HEURISTIC 1: The Ambiguity Guardrail ---
            result = self._metacognitive_gate(response.content)

            return {
                "status": "success",
                "agent_id": self.agent_id,
                "output": result,
                "meta": {
                    "protocol": "json-rpc-2.0",
                    "mode": result.get("mode"), # 'tool_execution' or 'elicitation'
                    "conviction_score": result.get("conviction_score")
                }
            }

        except Exception as e:
            logger.error(f"Protocol Error in {self.agent_id}: {str(e)}", exc_info=True)
            return {"status": "error", "error": str(e)}

    def _filter_tools_by_budget(self, query: str) -> List[Dict]:
        """
        Implementation of Heuristic 2: Entropy Check.
        Prevents 'Context Saturation' by loading only high-relevance schemas.
        """
        # (Placeholder logic: In a real system, use embeddings to select top-k tools)
        # If the prompt mentions "risk", load risk tools. If "sql", load SQL tools.
        relevant_tools = []
        current_tokens = 0

        for tool in self.tools:
            # Estimate token cost of schema
            cost = len(json.dumps(tool)) / 4
            if current_tokens + cost > self.max_tool_context_tokens:
                break
            relevant_tools.append(tool)
            current_tokens += cost

        return relevant_tools

    def _construct_protocol_prompt(self, task: str, schemas: List[Dict]) -> str:
        return f"""
        SYSTEM: You are an autonomous agent utilizing the Model Context Protocol (MCP).

        PROTOCOL: JSON-RPC 2.0

        AVAILABLE TOOLS:
        {json.dumps(schemas, indent=2)}

        CRITICAL INSTRUCTION - AMBIGUITY GUARDRAIL:
        1. Assess your CONVICTION. Do you have all necessary parameters to call a tool?
        2. If YES (Conviction > 85%): Output a valid JSON-RPC Request.
        3. If NO (Conviction < 85%): Output a JSON containing a "clarification_request".

        FORMAT:
        {{
          "thought_trace": "Reasoning about parameter certainty...",
          "conviction_score": 0.0-1.0,
          "action": {{ ... JSON-RPC or Clarification ... }}
        }}

        TASK: {task}
        """

    def _metacognitive_gate(self, content: str) -> Dict[str, Any]:
        """
        Implementation of Heuristic 1: The 'Ask, Don't Guess' Rule.
        Parses the LLM output and enforces the conviction threshold.
        """
        try:
            payload = json.loads(content)
            score = payload.get("conviction_score", 0.0)
            action = payload.get("action", {})

            # --- The Gate ---
            if score < self.conviction_threshold:
                # REJECTION: The agent is guessing parameters.
                # Force a mode switch from 'Protocol' to 'Natural Language Elicitation'.
                logger.warning(f"Conviction {score} < Threshold. Triggering Elicitation.")

                if "method" in action:
                    # The model tried to call a tool despite low confidence. Intercept it.
                    return {
                        "mode": "elicitation",
                        "response": f"I am not confident enough to execute '{action['method']}'. "
                                    f"Please clarify the specific parameters: {action.get('params', {}).keys()}",
                        "conviction_score": score
                    }

                # If it already asked for clarification, pass it through.
                return {"mode": "elicitation", "response": action.get("clarification_request"), "conviction_score": score}

            # PASS: High conviction. Execute the tool.
            if "method" in action:
                return {
                    "mode": "tool_execution",
                    "jsonrpc_payload": {
                        "jsonrpc": "2.0",
                        "method": action["method"],
                        "params": action.get("params", {}),
                        "id": 1
                    },
                    "conviction_score": score
                }

            return {"mode": "direct_answer", "response": action.get("result"), "conviction_score": score}

        except json.JSONDecodeError:
            return {"error": "Invalid JSON output from model."}
