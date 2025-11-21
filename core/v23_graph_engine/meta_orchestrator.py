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
from typing import Dict, Any, Optional
from core.v23_graph_engine.neuro_symbolic_planner import NeuroSymbolicPlanner
from core.v23_graph_engine.states import init_risk_state

logger = logging.getLogger(__name__)

class MetaOrchestrator:
    def __init__(self):
        self.planner = NeuroSymbolicPlanner()
        # In a real implementation, we would instantiate the v22 HybridOrchestrator here
        # self.legacy_orchestrator = HybridOrchestrator() 
        
    def route_request(self, query: str, context: Dict[str, Any] = None) -> Any:
        """
        Analyzes the query complexity and routes to the best engine.
        """
        complexity = self._assess_complexity(query)
        logger.info(f"MetaOrchestrator: Query complexity is {complexity}")
        
        if complexity == "HIGH":
            return self._run_adaptive_flow(query)
        elif complexity == "MEDIUM":
            # return self.legacy_orchestrator.run_async(query)
            return {"status": "Routed to v22 Async Engine", "query": query}
        else:
            # return self.legacy_orchestrator.run_sync(query)
            return {"status": "Routed to v21 Sync Tool", "query": query}

    def _assess_complexity(self, query: str) -> str:
        """
        Simple heuristic for routing. 
        In production, this would be a BERT classifier or Router LLM.
        """
        query_lower = query.lower()
        # v23 is best for analysis, planning, and risk
        if any(x in query_lower for x in ["analyze", "risk", "plan", "strategy", "report"]):
            return "HIGH"
        # v22 is best for monitoring, alerts, background tasks
        elif any(x in query_lower for x in ["monitor", "alert", "watch", "notify"]):
            return "MEDIUM"
        # v21 is best for simple lookups
        return "LOW"

    def _run_adaptive_flow(self, query: str):
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
        if "Apple" in query: ticker = "AAPL"
        if "Microsoft" in query: ticker = "MSFT"
        
        initial_state = init_risk_state(ticker, query)
        
        try:
            result = app.invoke(initial_state)
            return {
                "status": "v23 Execution Complete",
                "final_state": result
            }
        except Exception as e:
            logger.error(f"v23 Execution Failed: {e}")
            return {"error": str(e)}
