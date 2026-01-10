from __future__ import annotations
from typing import Any, Dict, List, Optional, Literal
import logging
import random
import asyncio

from core.agents.agent_base import AgentBase
from langgraph.graph import StateGraph, END
from core.engine.states import RedTeamState, GraphState
from core.agents.skills.counterfactual_reasoning_skill import CounterfactualReasoningSkill

logger = logging.getLogger(__name__)


class RedTeamAgent(AgentBase):
    """
    The Red Team Agent acts as an internal adversary to the system.

    ### Functionality:
    It generates novel and challenging scenarios (stress tests) to validate risk models before
    strategies are deployed. This is a critical component of the "Sovereign Financial Intelligence"
    architecture (v23.5), ensuring that the system is robust against "Black Swan" events.

    ### Architecture:
    In v23.5, this agent implements an internal **Adversarial Self-Correction Loop** using LangGraph.
    Instead of a single-shot generation, it iteratively refines its attack scenarios until they
    meet a severity threshold.

    ### Workflow:
    1.  **Generate Attack**: Uses `CounterfactualReasoningSkill` to invert assumptions in a credit memo.
    2.  **Simulate Impact**: Estimates the financial damage (e.g., VaR spike) of the scenario.
    3.  **Critique**: Checks if the scenario is severe enough (Severity > Threshold).
    4.  **Escalate**: If too mild, it loops back to Generate Attack with instructions to "Escalate".
    """

    def __init__(self, config: Dict[str, Any], kernel=None):
        super().__init__(config, kernel=kernel)
        self._name = config.get("name", "RedTeamAgent") # Use private attribute to avoid conflict
        self.skill = CounterfactualReasoningSkill(llm_client=kernel)
        # Compile the internal graph once during initialization
        self.graph_app = self._build_red_team_graph()

    @property
    def name(self) -> str:
        return self._name

    @name.setter
    def name(self, value: str):
        self._name = value

    async def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main entry point. compatible with AdaptiveSystemGraph.

        Args:
            state (dict): Can be a legacy dict or a GraphState. Must contain target info.

        Returns:
            dict: The final critique and impact assessment.
        """
        logger.info(f"[{self.name}] Executing Adversarial Stress Test...")

        # Determine target from input state
        target_entity = "General Portfolio"
        credit_memo = {}

        if isinstance(state, dict):
            # Handle GraphState input (from AdaptiveSystemGraph)
            if "request" in state:
                target_entity = state.get("request", "Unknown Target")
            # Handle direct dictionary input (legacy)
            elif "target_portfolio_id" in state:
                target_entity = state["target_portfolio_id"]

            # Extract memo if available
            credit_memo = state.get("credit_memo", {"assumptions": {"revenue_growth": 0.05, "interest_rate": 0.04}})

        # Initialize RedTeamState
        initial_state: RedTeamState = {
            "target_entity": target_entity,
            "scenario_type": "Macro",
            "current_scenario_description": "",
            "simulated_impact_score": 0.0,
            "severity_threshold": 7.5,  # Target high severity (0-10 scale)
            "critique_notes": [],
            "iteration_count": 0,
            "is_sufficiently_severe": False,
            "human_readable_status": "Initiating Red Team Loop...",
            # Inject data for skill use
            "data_context": {"credit_memo": credit_memo}
        }

        # Invoke the internal graph (Adversarial Self-Correction)
        final_state = await self.graph_app.ainvoke(initial_state)

        # Format output for the parent graph (AdaptiveSystemGraph)
        return {
            "critique": {
                "feedback": final_state.get("current_scenario_description"),
                "meets_standards": final_state.get("is_sufficiently_severe"),
                "impact_score": final_state.get("simulated_impact_score"),
                "skill_output": final_state.get("skill_output", {})
            },
            "human_readable_status": final_state.get("human_readable_status")
        }

    # --- Internal Graph Nodes ---

    async def _generate_attack_node(self, state: RedTeamState) -> Dict[str, Any]:
        """
        Node: Generates or refines an adversarial scenario using CounterfactualReasoningSkill.
        """
        target = state["target_entity"]
        iteration = state["iteration_count"]
        current_desc = state.get("current_scenario_description", "")

        credit_memo = state.get("data_context", {}).get("credit_memo", {})
        if not credit_memo:
             credit_memo = {"assumptions": {"revenue_growth": 0.05, "interest_rate": 0.04}}

        logger.info(f"[{self.name}] Generating scenario (Iter {iteration})...")

        # Use Skill to generate the base bear case
        bear_case = self.skill.generate_bear_case(credit_memo)
        new_desc = f"{bear_case['scenario']}: {bear_case['failure_catalyst']} triggered by {bear_case['inverted_assumptions']}"

        # Escalation Logic: If previous iterations failed, make it worse.
        if iteration > 0 and not state["is_sufficiently_severe"]:
             new_desc = f"ESCALATED (Iter {iteration}): {new_desc} AND Secondary Liquidity Crisis."

        return {
            "current_scenario_description": new_desc,
            "iteration_count": iteration + 1,
            "human_readable_status": f"Drafting scenario (Iter {iteration})",
            "skill_output": bear_case
        }

    async def _simulate_impact_node(self, state: RedTeamState) -> Dict[str, Any]:
        """
        Node: Simulates impact.
        In a full implementation, this would call the QuantumRiskEngine.
        Here we use the skill's heuristic score and apply multipliers for escalation.
        """
        desc = state["current_scenario_description"]
        skill_output = state.get("skill_output", {})

        # Use skill output if available, else fallback
        total_impact = skill_output.get("simulated_impact_score", 0.0) / 10.0 # Scale to 0-10

        # Adjust for escalation (manual boost if we added "AND Secondary Liquidity Crisis")
        if "ESCALATED" in desc:
            total_impact = min(10.0, total_impact * 1.5)

        logger.info(f"[{self.name}] Simulated Impact: {total_impact:.2f}")

        return {
            "simulated_impact_score": total_impact,
            "human_readable_status": f"Simulated Impact: {total_impact:.1f}"
        }

    async def _critique_node(self, state: RedTeamState) -> Dict[str, Any]:
        """
        Node: Checks severity threshold.
        Acts as the 'Reflector' in the cyclical process.
        """
        impact = state["simulated_impact_score"]
        threshold = state["severity_threshold"]

        is_severe = impact >= threshold

        return {
            "is_sufficiently_severe": is_severe,
            "human_readable_status": f"Critique: {'Severe enough' if is_severe else 'Too mild'}"
        }

    def _should_continue(self, state: RedTeamState) -> Literal["escalate", "finalize"]:
        """
        Conditional Edge: Decides whether to loop back or finish.
        """
        if state["is_sufficiently_severe"]:
            return "finalize"
        if state["iteration_count"] >= 3:  # Max retries to prevent infinite loops
            return "finalize"
        return "escalate"

    # --- Graph Construction ---

    def _build_red_team_graph(self):
        workflow = StateGraph(RedTeamState)

        # Add nodes (bound to self methods)
        workflow.add_node("generate_attack", self._generate_attack_node)
        workflow.add_node("simulate_impact", self._simulate_impact_node)
        workflow.add_node("critique", self._critique_node)

        workflow.set_entry_point("generate_attack")

        workflow.add_edge("generate_attack", "simulate_impact")
        workflow.add_edge("simulate_impact", "critique")

        workflow.add_conditional_edges(
            "critique",
            self._should_continue,
            {
                "escalate": "generate_attack",
                "finalize": END
            }
        )

        return workflow.compile()
