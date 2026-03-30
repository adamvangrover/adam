import logging
import asyncio
from typing import Dict, Any, Optional, List
# Import mocks first to patch environment if needed
import core.intelligence.nssf.mocks
from core.engine.neuro_symbolic_planner import NeuroSymbolicPlanner
try:
    from core.v24_architecture.brain.rag_planner import RAGPlanner
    V24_AVAILABLE = True
except ImportError:
    V24_AVAILABLE = False

from core.risk_engine.quantum_model import QuantumRiskEngine
from core.intelligence.nssf.liquid import LiquidSemanticEncoder
from core.intelligence.nssf.framer import FrameCollapser, RefinedExecutionPlan
from core.schemas.v23_5_schema import ExecutionPlan, PlanStep

logger = logging.getLogger(__name__)

class NSSFOrchestrator:
    """
    The main entry point for the Neuro-Symbolic Semantic Framing (NSSF) module.
    It orchestrates the flow from Symbolic Planning -> Liquid Encoding -> Deterministic Collapsing.
    """
    def __init__(self):
        # 1. Instantiate the Planners
        self.v23_planner = NeuroSymbolicPlanner()
        self.v24_planner = RAGPlanner() if V24_AVAILABLE else None

        # 2. Instantiate the Quantum Risk Engine (Backbone for Synapses)
        self.risk_engine = QuantumRiskEngine()

        # 3. Instantiate the Liquid Semantic Encoder (Agent 2)
        # Initialize LNN with default dimensions.
        # input_dim matches the mocked projection layer in liquid.py
        self.liquid_encoder = LiquidSemanticEncoder(
            input_dim=128,
            hidden_dim=64,
            backend_engine=self.risk_engine
        )

        # 4. Instantiate the Frame Collapser (Agent 3)
        self.collapser = FrameCollapser()

    async def process_query_async(self, query: str, context: Dict[str, Any] = None) -> RefinedExecutionPlan:
        """
        Full pipeline execution (Async):
        Query -> Planner (v24/v23) -> Plan -> Liquid Encoder -> State -> Collapser -> Refined Plan
        """
        if context is None:
            context = {}

        logger.info(f"NSSF: Processing query '{query}'")

        # Step 1: Generate Symbolic Plan
        if self.v24_planner:
            try:
                logger.info("NSSF: Attempting v24 RAG Planner")
                v24_result = await self.v24_planner.create_plan(query)
                plan = self._adapt_v24_plan(v24_result)
            except Exception as e:
                logger.error(f"NSSF: v24 Planner failed: {e}. Falling back to v23.")
                plan = self.v23_planner.generate_plan(query, context)
        else:
            plan = self.v23_planner.generate_plan(query, context)

        logger.info(f"NSSF: Generated plan with {len(plan.steps)} steps")

        # Step 2: Liquid Semantic Encoding
        liquid_state = self.liquid_encoder.encode_plan(plan)

        # Step 3: Deterministic Collapsing
        refined_plan = self.collapser.collapse(plan, liquid_state, context)

        logger.info(f"NSSF: Plan collapsed. Verdict: {refined_plan.collapser_verdict}")

        return refined_plan

    def process_query(self, query: str, context: Dict[str, Any] = None) -> RefinedExecutionPlan:
        """Synchronous wrapper for convenience."""
        return asyncio.run(self.process_query_async(query, context))

    def _adapt_v24_plan(self, v24_dict: Dict[str, Any]) -> ExecutionPlan:
        """
        Adapts the dictionary output from RAGPlanner to the v23 ExecutionPlan schema.
        """
        steps = []
        for s in v24_dict.get("steps", []):
            steps.append(PlanStep(
                step_id=s.get("id", "unknown"),
                action=s.get("action", "generic_action"),
                target_entity="v24_entity", # Simplified for now
                parameters=s.get("params", {}),
                rationale=s.get("description", "")
            ))

        return ExecutionPlan(
            plan_id=v24_dict.get("plan_id", "v24-adapted"),
            steps=steps,
            estimated_complexity="HIGH" # Assume high for RAG
        )

def main():
    # Example usage for verification
    logging.basicConfig(level=logging.INFO)
    nssf = NSSFOrchestrator()

    # Test query
    query = "Analyze risks for Tesla supply chain"

    # Context with some simulated market data
    context = {
        "volatility": 0.15,
        "uncertainty": 0.05
    }

    result = nssf.process_query(query, context)
    print(result.model_dump_json(indent=2))

if __name__ == "__main__":
    main()
