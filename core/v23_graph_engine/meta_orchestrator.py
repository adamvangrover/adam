# core/v23_graph_engine/meta_orchestrator.py

"""
Agent Notes (Meta-Commentary):
The MetaOrchestrator is the supreme controller of the Adam system in v23.0.
It implements the "Brain" of the architecture, deciding which cognitive path
a user query should take:
1. Fast Path (v21): Direct tool execution (e.g. "Get stock price").
2. Async Path (v22): Message-driven workflow (e.g. "Monitor news").
3. Adaptive Path (v23): Neuro-Symbolic Planner (e.g. "Analyze complex credit risk").
"""

import logging
import asyncio
from typing import Dict, Any, Optional
from core.v23_graph_engine.neuro_symbolic_planner import NeuroSymbolicPlanner
from core.v23_graph_engine.states import init_risk_state
from core.v23_graph_engine.red_team_graph import red_team_app
from core.system.agent_orchestrator import AgentOrchestrator

logger = logging.getLogger(__name__)

class MetaOrchestrator:
    def __init__(self, legacy_orchestrator: AgentOrchestrator = None):
        self.planner = NeuroSymbolicPlanner()
        self.legacy_orchestrator = legacy_orchestrator or AgentOrchestrator()
        
    async def route_request(self, query: str, context: Dict[str, Any] = None) -> Any:
        """
        Analyzes the query complexity and routes to the best engine.
        Now Async!
        """
        complexity = self._assess_complexity(query)
        logger.info(f"MetaOrchestrator: Query complexity is {complexity}")
        
        if complexity == "RED_TEAM":
            return await self._run_red_team_flow(query)
        elif complexity == "HIGH":
            return await self._run_adaptive_flow(query)
        elif complexity == "MEDIUM":
            # Route to legacy workflow (Async v22 style via AgentOrchestrator)
            # For now we use a generic workflow or analysis
            logger.info("Routing to Legacy/Async Workflow...")
            # Assuming 'general_analysis' or similar workflow exists, or just use QueryUnderstanding
            # Since execute_workflow expects a workflow name defined in workflows.yaml
            # We'll try to find a matching workflow or default to a simple one.
            return await self.legacy_orchestrator.execute_workflow("test_workflow", initial_context={"user_query": query})
        else:
            # Low complexity -> Legacy Single Agent Execution
            logger.info("Routing to Legacy Single Agent...")
            # Using execute_agent which is Fire-and-Forget (Message Broker)
            self.legacy_orchestrator.execute_agent("QueryUnderstandingAgent", context={"user_query": query})
            return {"status": "Dispatched to Message Broker", "query": query}

    def _assess_complexity(self, query: str) -> str:
        """
        Simple heuristic for routing. 
        In production, this would be a BERT classifier or Router LLM.
        """
        query_lower = query.lower()

        # Red Team / Adversarial
        if any(x in query_lower for x in ["attack", "stress test", "scenario", "adversarial", "simulate"]):
            return "RED_TEAM"

        # v23 is best for analysis, planning, and risk
        if any(x in query_lower for x in ["analyze", "risk", "plan", "strategy", "report"]):
            return "HIGH"
        # v22 is best for monitoring, alerts, background tasks
        elif any(x in query_lower for x in ["monitor", "alert", "watch", "notify"]):
            return "MEDIUM"
        # v21 is best for simple lookups
        return "LOW"

    async def _run_adaptive_flow(self, query: str):
        logger.info("Engaging v23 Neuro-Symbolic Planner...")
        
        # 1. Discover Plan
        plan = self.planner.discover_plan(query)
        if not plan:
            return {"error": "Failed to generate a plan."}
            
        # 2. Compile Graph
        app = self.planner.to_executable_graph(plan)
        if not app:
             return {"error": "Failed to compile graph."}

        # 3. Execute
        # We guess the ticker for now or extract it. 
        # In a real system, the planner would extract parameters into the state.
        ticker = "AAPL" # Default/Mock
        if "apple" in query.lower(): ticker = "AAPL"
        if "microsoft" in query.lower(): ticker = "MSFT"
        if "tesla" in query.lower(): ticker = "TSLA"
        
        initial_state = init_risk_state(ticker, query)
        
        try:
            # invoke is synchronous in LangGraph unless using ainvoke?
            # LangGraph apps are runnables. invoke is sync. ainvoke is async.
            # We should use ainvoke if possible to keep it async friendly.
            if hasattr(app, 'ainvoke'):
                result = await app.ainvoke(initial_state)
            else:
                result = app.invoke(initial_state)

            return {
                "status": "v23 Execution Complete",
                "final_state": result
            }
        except Exception as e:
            logger.error(f"v23 Execution Failed: {e}")
            return {"error": str(e)}

    async def _run_red_team_flow(self, query: str):
        logger.info("Engaging Red Team Graph...")

        # Extract params (Mock)
        target = "Apple Inc."
        scenario = "Cyber"
        if "rate" in query.lower() or "fed" in query.lower():
            scenario = "Macro"
        elif "law" in query.lower() or "compliance" in query.lower():
            scenario = "Regulatory"

        initial_state = {
            "target_entity": target,
            "scenario_type": scenario,
            "current_scenario_description": "",
            "simulated_impact_score": 0.0,
            "severity_threshold": 8.0, # High threshold to force loops
            "critique_notes": [],
            "iteration_count": 0,
            "is_sufficiently_severe": False,
            "human_readable_status": "Initiating Red Team Protocol..."
        }

        try:
             # Using ainvoke if available
            if hasattr(red_team_app, 'ainvoke'):
                result = await red_team_app.ainvoke(initial_state)
            else:
                result = red_team_app.invoke(initial_state)
            return {
                "status": "Red Team Simulation Complete",
                "final_state": result
            }
        except Exception as e:
            logger.error(f"Red Team Execution Failed: {e}")
            return {"error": str(e)}
