# core/engine/meta_orchestrator.py

import logging
import asyncio
from typing import Dict, Any, List
import uuid

# Logic Layers
from core.engine.neuro_symbolic_planner import NeuroSymbolicPlanner
from core.common.mcp_executor import MCPExecutor

# Graph Engines
try:
    from core.engine.deep_dive_graph_agentic import deep_dive_app
except ImportError:
    from core.engine.deep_dive_graph import deep_dive_app

from core.engine.red_team_graph import red_team_app
from core.engine.esg_graph import esg_graph_app
from core.engine.regulatory_compliance_graph import compliance_graph_app
from core.engine.crisis_simulation_graph import crisis_simulation_app
from core.engine.states import init_esg_state, init_compliance_state, init_crisis_state

from core.system.agent_orchestrator import AgentOrchestrator
from core.mcp.registry import MCPRegistry

logger = logging.getLogger(__name__)

class MetaOrchestrator:
    """
    Adam v23.5 Hybrid MetaOrchestrator.
    Integrates v23.0 Adaptive Routing (FO Apps, Complexity Check)
    with v23.5 Swarm Logic (Plan -> Execute -> Reflect -> Synthesis).
    """

    def __init__(self, legacy_orchestrator: AgentOrchestrator = None):
        self.legacy_orchestrator = legacy_orchestrator or AgentOrchestrator()
        self.planner = NeuroSymbolicPlanner()
        self.mcp_executor = MCPExecutor()
        self.mcp_registry = MCPRegistry() # Legacy FO MCP

    async def route_request(self, query: str, context: Dict[str, Any] = None) -> Any:
        """
        Main entry point. Routes based on complexity and intent.
        """
        logger.info(f"MetaOrchestrator: Processing query '{query}'")

        # 1. Assess Complexity/Intent (v23.0 Logic)
        complexity = self._assess_complexity(query, context)
        logger.info(f"Routed Complexity/Strategy: {complexity}")

        # 2. Route
        # Family Office Super-App Routing
        if complexity.startswith("FO_"):
            if complexity == "FO_WEALTH": return await self._run_fo_wealth(query)
            elif complexity == "FO_DEAL": return await self._run_fo_deal(query)
            elif complexity == "FO_EXECUTION": return await self._run_fo_execution(query)
            elif complexity == "FO_MARKET": return await self._run_fo_market(query)
        
        # Swarm / High-Value Routing (v23.5)
        elif complexity in ["DEEP_DIVE", "RED_TEAM", "ESG", "COMPLIANCE", "CRISIS", "HIGH"]:
            return await self._run_swarm_logic(query, strategy=complexity, context=context)

        # Legacy / Medium Routing
        elif complexity == "MEDIUM":
             logger.info("Routing to Legacy/Async Workflow...")
             return await self.legacy_orchestrator.execute_workflow("test_workflow", initial_context={"user_query": query})

        # Fallback / Low
        else:
             logger.info("Routing to Legacy Single Agent...")
             self.legacy_orchestrator.execute_agent("QueryUnderstandingAgent", context={"user_query": query})
             return {"status": "Dispatched to Message Broker", "query": query}

    # --- v23.5 Swarm Logic ---

    async def _run_swarm_logic(self, query: str, strategy: str, context: Dict[str, Any] = None) -> Any:
        """
        Implements the State Machine: Plan -> Execute -> Reflect -> Synthesis.
        """
        # STATE 1: PLAN (Refinement)
        plan = self._plan_phase(query, strategy, context)
        
        # STATE 2 & 3: EXECUTE & REFLECT LOOP
        max_retries = 3
        verified_data = None
        
        for i in range(max_retries):
            logger.info(f"Swarm Cycle {i+1} for {strategy}")

            # Execute
            raw_result = await self._execute_phase(plan)

            # Reflect
            reflection = await self._reflect_phase(raw_result, query)

            if reflection["status"] == "PASSED":
                verified_data = raw_result
                break
            else:
                logger.warning(f"Reflection Failed: {reflection.get('reason')}. Replanning...")
                plan["retry_context"] = reflection.get("reason")

        if not verified_data:
            return {"error": "Workflow failed validation after max retries.", "last_result": raw_result if 'raw_result' in locals() else None}

        # STATE 4: SYNTHESIS
        return self._synthesis_phase(verified_data)

    def _plan_phase(self, query: str, strategy: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        return {
            "strategy": strategy,
            "query": query,
            "context": context or {},
            "tools": ["azure_ai_search", "microsoft_fabric_run_sql"]
        }

    async def _execute_phase(self, plan: Dict[str, Any]) -> Any:
        strategy = plan["strategy"]
        query = plan["query"]
        config = {"configurable": {"thread_id": str(uuid.uuid4())}}

        try:
            if strategy == "DEEP_DIVE" or strategy == "HIGH":
                ticker = "AAPL"
                if "apple" in query.lower(): ticker = "AAPL"
                initial_state = {"ticker": ticker, "v23_knowledge_graph": {"meta": {}, "nodes": {}}}

                if deep_dive_app:
                    if hasattr(deep_dive_app, 'ainvoke'): return await deep_dive_app.ainvoke(initial_state, config=config)
                    return deep_dive_app.invoke(initial_state, config=config)
                return {"error": "Deep Dive App not available"}

            elif strategy == "RED_TEAM":
                initial_state = {
                    "target_entity": "Apple Inc.",
                    "scenario_type": "Cyber",
                    "current_scenario_description": query,
                    "simulated_impact_score": 0.0,
                    "severity_threshold": 5.0,
                    "critique_notes": [],
                    "iteration_count": 0,
                    "is_sufficiently_severe": False,
                    "human_readable_status": "Initiating Red Team..."
                }
                if red_team_app:
                    if hasattr(red_team_app, 'ainvoke'): return await red_team_app.ainvoke(initial_state, config=config)
                    return red_team_app.invoke(initial_state, config=config)

            elif strategy == "ESG":
                initial_state = init_esg_state("Apple Inc.", "Technology")
                if esg_graph_app:
                    if hasattr(esg_graph_app, 'ainvoke'): return await esg_graph_app.ainvoke(initial_state, config=config)
                    return esg_graph_app.invoke(initial_state, config=config)

            elif strategy == "COMPLIANCE":
                initial_state = init_compliance_state("Generic Bank", "US")
                if compliance_graph_app:
                    if hasattr(compliance_graph_app, 'ainvoke'): return await compliance_graph_app.ainvoke(initial_state, config=config)
                    return compliance_graph_app.invoke(initial_state, config=config)

            elif strategy == "CRISIS":
                initial_state = init_crisis_state(query, {"aum": 1000000})
                if crisis_simulation_app:
                     if hasattr(crisis_simulation_app, 'ainvoke'): return await crisis_simulation_app.ainvoke(initial_state, config=config)
                     return crisis_simulation_app.invoke(initial_state, config=config)

            # Fallback within Swarm
            return {"error": f"Strategy {strategy} not implemented in Swarm"}

        except Exception as e:
            logger.error(f"Execution failed for {strategy}: {e}", exc_info=True)
            return {"error": str(e)}

    async def _reflect_phase(self, result: Any, query: str) -> Dict[str, Any]:
        if isinstance(result, dict):
            if "error" in result:
                return {"status": "FAILED", "reason": result["error"]}
            valid_keys = ["v23_knowledge_graph", "final_state", "tool_output", "final_report"]
            if any(k in result for k in valid_keys):
                 return {"status": "PASSED"}
        return {"status": "PASSED"}

    def _synthesis_phase(self, verified_data: Any) -> Any:
        return {
            "final_report": verified_data,
            "status": "Synthesized and Verified"
        }

    # --- v23.0 Logic (Restored & Enhanced) ---

    def _assess_complexity(self, query: str, context: Dict[str, Any] = None) -> str:
        query_lower = query.lower()
        context = context or {}

        if "deep dive" in query_lower or context.get("simulation_depth") == "Deep": return "DEEP_DIVE"
        if any(x in query_lower for x in ["full analysis", "partner", "valuation", "covenant"]): return "DEEP_DIVE"
        if any(x in query_lower for x in ["attack", "adversarial", "red team"]): return "RED_TEAM"
        if any(x in query_lower for x in ["simulation", "simulate", "crisis", "shock", "stress test"]): return "CRISIS"
        if any(x in query_lower for x in ["esg", "environmental", "sustainability"]): return "ESG"
        if any(x in query_lower for x in ["compliance", "kyc", "aml", "regulation"]): return "COMPLIANCE"

        if any(x in query_lower for x in ["ips", "trust", "wealth", "goal", "governance"]): return "FO_WEALTH"
        if any(x in query_lower for x in ["deal", "private equity", "venture", "screening"]): return "FO_DEAL"
        if any(x in query_lower for x in ["buy", "sell", "execute", "order"]): return "FO_EXECUTION"
        if any(x in query_lower for x in ["price", "quote", "ticker", "market data"]): return "FO_MARKET"

        if any(x in query_lower for x in ["analyze", "risk", "plan", "strategy"]): return "HIGH"
        if any(x in query_lower for x in ["monitor", "alert", "watch"]): return "MEDIUM"
        return "LOW"

    async def _run_fo_market(self, query: str):
        logger.info("Engaging FO Super-App Market Module...")
        words = query.split()
        symbol = "AAPL"
        for w in words:
            if w.isupper() and len(w) <= 5 and w.isalpha(): symbol = w

        if "price" in query.lower() or "quote" in query.lower():
            result = self.mcp_registry.invoke("price_asset", symbol=symbol, side="mid")
        else:
            result = self.mcp_registry.invoke("retrieve_market_data", symbol=symbol)
        return {"status": "FO Market Data Retrieved", "data": result}

    async def _run_fo_execution(self, query: str):
        logger.info("Engaging FO Super-App Execution Module...")
        words = query.split()
        symbol = "AAPL"
        side = "buy"
        qty = 100
        if "sell" in query.lower(): side = "sell"
        for w in words:
            if w.isupper() and len(w) <= 5 and w.isalpha(): symbol = w
            if w.isdigit(): qty = float(w)
        result = self.mcp_registry.invoke("execute_order", order={"symbol": symbol, "side": side, "qty": qty})
        return {"status": "FO Execution Submitted", "report": result}

    async def _run_fo_wealth(self, query: str):
        logger.info("Engaging FO Super-App Wealth Module...")
        if "plan" in query.lower() and "goal" in query.lower():
            goal_name = "General Wealth Goal"
            target = 1000000.0
            horizon = 10
            result = self.mcp_registry.invoke("plan_wealth_goal", goal_name=goal_name, target_amount=target, horizon_years=horizon, current_savings=target*0.1)
            return {"status": "Wealth Plan Generated", "plan": result}
        elif "ips" in query.lower() or "governance" in query.lower():
            result = self.mcp_registry.invoke("generate_ips", family_name="Smith", risk_profile="Growth", goals=["Preserve Capital", "Growth"], constraints=["ESG"])
            return {"status": "IPS Generated", "ips": result}
        return {"status": "Wealth Module: Unknown Action"}

    async def _run_fo_deal(self, query: str):
        logger.info("Engaging FO Super-App Deal Flow Module...")
        deal_name = "Project Alpha"
        sector = "Technology"
        val = 100.0
        ebitda = 10.0
        result = self.mcp_registry.invoke("screen_deal", deal_name=deal_name, sector=sector, valuation=val, ebitda=ebitda)
        return {"status": "Deal Screened", "result": result}
