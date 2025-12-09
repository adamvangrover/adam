# core/engine/meta_orchestrator.py

"""
Agent Notes (Meta-Commentary):
The MetaOrchestrator is the supreme controller of the Adam system in v23.0.
It implements the "Brain" of the architecture, deciding which cognitive path
a user query should take:
1. Fast Path (v21): Direct tool execution (e.g. "Get stock price").
2. Async Path (v22): Message-driven workflow (e.g. "Monitor news").
3. Adaptive Path (v23): Neuro-Symbolic Planner (e.g. "Analyze complex credit risk").
4. Specialized Paths: ESG, Compliance, Red Team.
5. Deep Dive Path (v23.5): 5-Phase autonomous financial analysis.
"""

import logging
import asyncio
from typing import Dict, Any, Optional

from core.engine.neuro_symbolic_planner import NeuroSymbolicPlanner
from core.engine.states import init_risk_state, init_esg_state, init_compliance_state, init_crisis_state, init_omniscient_state
from core.engine.red_team_graph import red_team_app
from core.engine.esg_graph import esg_graph_app
from core.engine.regulatory_compliance_graph import compliance_graph_app
from core.engine.crisis_simulation_graph import crisis_simulation_app
from core.engine.deep_dive_graph import deep_dive_app
from core.engine.reflector_graph import reflector_app
from core.engine.states import init_reflector_state
from core.system.agent_orchestrator import AgentOrchestrator
from core.mcp.registry import MCPRegistry

# v23.5 Deep Dive Agents (for Fallback)
from core.agents.specialized.management_assessment_agent import ManagementAssessmentAgent
from core.agents.fundamental_analyst_agent import FundamentalAnalystAgent
from core.agents.specialized.peer_comparison_agent import PeerComparisonAgent
from core.agents.specialized.snc_rating_agent import SNCRatingAgent
from core.agents.specialized.covenant_analyst_agent import CovenantAnalystAgent
from core.agents.specialized.monte_carlo_risk_agent import MonteCarloRiskAgent
from core.agents.specialized.quantum_scenario_agent import QuantumScenarioAgent
from core.agents.specialized.portfolio_manager_agent import PortfolioManagerAgent
from core.schemas.v23_5_schema import V23KnowledgeGraph, Meta, Nodes, HyperDimensionalKnowledgeGraph, EquityAnalysis, Fundamentals, ValuationEngine, DCFModel, PriceTargets

logger = logging.getLogger(__name__)

class MetaOrchestrator:
    def __init__(self, legacy_orchestrator: AgentOrchestrator = None):
        self.planner = NeuroSymbolicPlanner()
        self.legacy_orchestrator = legacy_orchestrator or AgentOrchestrator()
        self.mcp_registry = MCPRegistry() # FO Super-App Integration
        
    async def route_request(self, query: str, context: Dict[str, Any] = None) -> Any:
        """
        Analyzes the query complexity and routes to the best engine.
        Now Async!
        """
        complexity = self._assess_complexity(query, context)
        logger.info(f"MetaOrchestrator: Query complexity is {complexity}")
        
        result = None
        if complexity == "DEEP_DIVE":
            result = await self._run_deep_dive_flow(query)
        elif complexity == "RED_TEAM":
            result = await self._run_red_team_flow(query)
        elif complexity == "CRISIS":
            result = await self._run_crisis_flow(query)
        elif complexity == "ESG":
            result = await self._run_esg_flow(query)
        elif complexity == "COMPLIANCE":
            result = await self._run_compliance_flow(query)
        elif complexity == "FO_WEALTH":
            result = await self._run_fo_wealth(query)
        elif complexity == "FO_DEAL":
            result = await self._run_fo_deal(query)
        elif complexity == "FO_EXECUTION":
            result = await self._run_fo_execution(query)
        elif complexity == "FO_MARKET":
            result = await self._run_fo_market(query)
        elif complexity == "HIGH":
            result = await self._run_adaptive_flow(query)
        elif complexity == "MEDIUM":
            # Route to legacy workflow (Async v22 style via AgentOrchestrator)
            logger.info("Routing to Legacy/Async Workflow...")
            result = await self.legacy_orchestrator.execute_workflow("test_workflow", initial_context={"user_query": query})
        else:
            # Low complexity -> Legacy Single Agent Execution
            logger.info("Routing to Legacy Single Agent...")
            self.legacy_orchestrator.execute_agent("QueryUnderstandingAgent", context={"user_query": query})
            result = {"status": "Dispatched to Message Broker", "query": query}

        # Optional: Reflection Step for high-value outputs
        if complexity in ["DEEP_DIVE", "HIGH", "CRISIS", "RED_TEAM"]:
             result = await self._reflect_on_result(result, query)

        return result

    async def _reflect_on_result(self, result: Any, query: str) -> Any:
        """
        Runs the ReflectorGraph to critique and potentially refine the result.
        """
        logger.info("Engaging Reflector (Meta-Cognition)...")

        # Extract content to reflect on
        content_to_reflect = str(result)
        # Try to find a human readable status or summary
        if isinstance(result, dict):
            if "human_readable_status" in result:
                content_to_reflect = result["human_readable_status"]
            elif "final_state" in result:
                # drill down
                fs = result["final_state"]
                if isinstance(fs, dict):
                    content_to_reflect = fs.get("human_readable_status", str(fs))

        # Initialize Reflector
        initial_state = init_reflector_state(content_to_reflect, context={"original_query": query})

        try:
            config = {"configurable": {"thread_id": "reflector_1"}}
            if hasattr(reflector_app, 'ainvoke'):
                reflection = await reflector_app.ainvoke(initial_state, config=config)
            else:
                reflection = reflector_app.invoke(initial_state, config=config)

            # Attach reflection to result
            if isinstance(result, dict):
                result["meta_reflection"] = reflection

            logger.info("Reflection Complete.")
            return result

        except Exception as e:
            logger.warning(f"Reflection failed: {e}. Returning original result.")
            return result

    def _assess_complexity(self, query: str, context: Dict[str, Any] = None) -> str:
        """
        Simple heuristic for routing. 
        In production, this would be a BERT classifier or Router LLM.
        """
        query_lower = query.lower()
        context = context or {}

        # v23.5 Deep Dive Trigger
        if "deep dive" in query_lower or context.get("simulation_depth") == "Deep":
            return "DEEP_DIVE"

        # Deep Dive / v23.5
        if any(x in query_lower for x in ["deep dive", "full analysis", "partner", "valuation", "covenant"]):
            return "DEEP_DIVE"

        # Red Team / Adversarial
        if any(x in query_lower for x in ["attack", "adversarial"]):
            return "RED_TEAM"

        # Crisis / Simulation
        if any(x in query_lower for x in ["simulation", "simulate", "crisis", "shock", "stress test", "scenario"]):
            return "CRISIS"

        # ESG
        if any(x in query_lower for x in ["esg", "environmental", "sustainability", "carbon", "green", "social impact"]):
            return "ESG"

        # Compliance
        if any(x in query_lower for x in ["compliance", "kyc", "aml", "regulation", "violation", "dodd-frank", "basel"]):
            return "COMPLIANCE"

        # FO Super-App: Wealth & Governance
        if any(x in query_lower for x in ["ips", "trust", "estate", "wealth", "goal", "governance"]):
            return "FO_WEALTH"

        # FO Super-App: Deal Flow
        if any(x in query_lower for x in ["deal", "private equity", "venture", "screening", "multiple"]):
            return "FO_DEAL"

        # FO Super-App: Execution
        if any(x in query_lower for x in ["buy", "sell", "execute trade", "place order"]):
            return "FO_EXECUTION"

        # FO Super-App: Market Data
        if any(x in query_lower for x in ["price", "quote", "ticker", "market data"]):
            return "FO_MARKET"

        # v23 is best for analysis, planning, and risk
        if any(x in query_lower for x in ["analyze", "risk", "plan", "strategy", "report"]):
            return "HIGH"
        # v22 is best for monitoring, alerts, background tasks
        elif any(x in query_lower for x in ["monitor", "alert", "watch", "notify"]):
            return "MEDIUM"
        # v21 is best for simple lookups
        return "LOW"

    async def _run_deep_dive_flow(self, query: str):
        """
        Primary execution method for Deep Dive.
        Tries the v23 Graph first, falls back to Manual Orchestration.
        """
        logger.info("Engaging v23.5 Deep Dive Protocol...")

        # Mock Ticker Extraction
        ticker = "AAPL"
        if "apple" in query.lower(): ticker = "AAPL"
        elif "tesla" in query.lower(): ticker = "TSLA"
        elif "microsoft" in query.lower(): ticker = "MSFT"

        initial_state = init_omniscient_state(ticker)

        # 1. Try Graph Execution
        try:
            logger.info("Attempting Deep Dive Graph Execution...")
            config = {"configurable": {"thread_id": "1"}}

            if deep_dive_app:
                if hasattr(deep_dive_app, 'ainvoke'):
                    result = await deep_dive_app.ainvoke(initial_state, config=config)
                else:
                    result = deep_dive_app.invoke(initial_state, config=config)

                logger.info("Deep Dive Graph Execution Successful.")
                # We normalize the output to be the OmniscientState directly (which contains v23_knowledge_graph)
                return result
            else:
                logger.warning("Deep Dive App is not initialized.")
                raise Exception("Graph App Missing")

        except Exception as e:
            logger.error(f"Deep Dive Graph Failed: {e}. Falling back to Manual Orchestration.", exc_info=True)
            return await self._run_deep_dive_manual_fallback(query)

    async def _run_deep_dive_manual_fallback(self, query: str):
        """
        Fallback method that manually calls agents if the graph fails.
        """
        logger.info("Engaging Deep Dive Manual Fallback...")

        try:
            # Initialize Agents (Empty config for now as they use defaults/mocks)
            mgmt_agent = ManagementAssessmentAgent(config={})
            fund_agent = FundamentalAnalystAgent(config={})
            peer_agent = PeerComparisonAgent(config={})
            snc_agent = SNCRatingAgent(config={})
            mc_agent = MonteCarloRiskAgent(config={})
            quant_agent = QuantumScenarioAgent(config={})
            pm_agent = PortfolioManagerAgent(config={})

            # Phase 1: Entity & Management
            logger.info("Fallback Phase 1: Entity & Management")
            company_name_extraction = query.replace("Perform a deep dive analysis on", "").strip() or query
            entity_eco = await mgmt_agent.execute({"company_name": company_name_extraction})

            # Phase 2: Deep Fundamental
            logger.info("Fallback Phase 2: Deep Fundamental")
            company_id = company_name_extraction
            fund_data = await fund_agent.execute(company_id)

            # Map fund_data
            dcf_val = 0.0
            if isinstance(fund_data, dict):
                 dcf_val = fund_data.get("dcf_valuation", 0.0)

            # Safe Fallback for Peer Agent
            multiples = {}
            try:
                multiples = await peer_agent.execute({"company_id": company_id})
            except:
                multiples = {"error": "Peer Agent Failed"}

            equity_analysis = EquityAnalysis(
                fundamentals=Fundamentals(
                    revenue_cagr_3yr="5%",
                    ebitda_margin_trend="Expanding"
                ),
                valuation_engine=ValuationEngine(
                    dcf_model=DCFModel(
                        wacc=0.08,
                        terminal_growth=0.02,
                        intrinsic_value=float(dcf_val)
                    ),
                    multiples_analysis=multiples,
                    price_targets=PriceTargets(bear_case=80.0, base_case=100.0, bull_case=120.0)
                )
            )

            # Phase 3: Credit
            logger.info("Fallback Phase 3: Credit")
            credit_input = {
                "facilities": [{"id": "TLB", "amount": "500M", "ltv": 0.5}],
                "current_leverage": 3.0
            }
            credit_analysis = await snc_agent.execute(credit_input)

            # Phase 4: Risk
            logger.info("Fallback Phase 4: Risk")
            ebitda = 100.0
            debt = 300.0
            sim_engine = await mc_agent.execute({"ebitda": ebitda, "debt": debt, "volatility": 0.2})

            try:
                scenarios = await quant_agent.execute({})
                sim_engine.quantum_scenarios = scenarios
            except:
                pass

            # Phase 5: Synthesis
            logger.info("Fallback Phase 5: Synthesis")
            synthesis = await pm_agent.execute({
                "phase1": entity_eco,
                "phase2": equity_analysis,
                "phase3": credit_analysis,
                "phase4": sim_engine
            })

            # Construct Knowledge Graph
            nodes = Nodes(
                entity_ecosystem=entity_eco,
                equity_analysis=equity_analysis,
                credit_analysis=credit_analysis,
                simulation_engine=sim_engine,
                strategic_synthesis=synthesis
            )

            kg = V23KnowledgeGraph(
                meta=Meta(target=query, generated_at="2023-10-27", model_version="Adam-v23.5-Fallback"),
                nodes=nodes
            )

            result = HyperDimensionalKnowledgeGraph(v23_knowledge_graph=kg)
            return result.model_dump(by_alias=True)

        except Exception as e:
            logger.error(f"Deep Dive Fallback Failed: {e}", exc_info=True)
            return {"error": f"All Deep Dive methods failed. Last error: {str(e)}"}

    async def _run_adaptive_flow(self, query: str):
        logger.info("Engaging v23 Neuro-Symbolic Planner...")
        
        # 0. Entity Extraction (Simple Heuristic)
        start_node = "Apple Inc."
        if "tesla" in query.lower(): start_node = "Tesla Inc."
        elif "microsoft" in query.lower(): start_node = "Microsoft Corp"

        target_node = "CreditRating"
        if "esg" in query.lower(): target_node = "ESGScore"
        elif "risk" in query.lower(): target_node = "RiskScore"
        elif "sentiment" in query.lower(): target_node = "Sentiment"

        # 1. Discover Plan
        try:
            plan_data = self.planner.discover_plan(start_node, target_node)
            if not plan_data:
                return {"error": "Failed to generate a plan."}
        except Exception as e:
            logger.error(f"Planner failed: {e}", exc_info=True)
            return {"error": "Planner Exception"}
            
        # 2. Compile Graph
        try:
            app = self.planner.to_executable_graph(plan_data)
            if not app:
                 return {"error": "Failed to compile graph."}
        except Exception as e:
            logger.error(f"Graph compilation failed: {e}", exc_info=True)
            return {"error": "Compilation Exception"}

        # 3. Execute
        # Transform plan_data into the expected PlanOnGraph structure for the state
        plan_struct = {
            "id": "plan-001",
            "steps": plan_data.get("steps", []),
            "is_complete": False,
            "cypher_query": plan_data.get("symbolic_plan")
        }
        
        initial_state = {
            "request": query,
            "plan": plan_struct,
            "current_task_index": 0,
            "assessment": {"content": ""},
            "critique": None,
            "human_feedback": None,
            "iteration": 0,
            "max_iterations": 5
        }
        
        try:
            config = {"configurable": {"thread_id": "1"}}
            if hasattr(app, 'ainvoke'):
                result = await app.ainvoke(initial_state, config=config)
            else:
                result = app.invoke(initial_state, config=config)

            return {
                "status": "v23 Execution Complete",
                "final_state": result
            }
        except Exception as e:
            logger.error(f"v23 Execution Failed: {e}", exc_info=True)
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
            config = {"configurable": {"thread_id": "1"}}
            if hasattr(red_team_app, 'ainvoke'):
                result = await red_team_app.ainvoke(initial_state, config=config)
            else:
                result = red_team_app.invoke(initial_state, config=config)
            return {
                "status": "Red Team Simulation Complete",
                "final_state": result
            }
        except Exception as e:
            logger.error(f"Red Team Execution Failed: {e}")
            return {"error": str(e)}

    async def _run_esg_flow(self, query: str):
        logger.info("Engaging ESG Graph...")

        # Mock Extraction
        company = "Acme Corp"
        if "apple" in query.lower(): company = "Apple Inc."
        elif "exxon" in query.lower(): company = "Exxon Mobil"

        sector = "Technology"
        if "oil" in query.lower() or "exxon" in query.lower(): sector = "Energy"

        initial_state = init_esg_state(company, sector)

        try:
            config = {"configurable": {"thread_id": "1"}}
            if hasattr(esg_graph_app, 'ainvoke'):
                result = await esg_graph_app.ainvoke(initial_state, config=config)
            else:
                result = esg_graph_app.invoke(initial_state, config=config)
            return {
                "status": "ESG Analysis Complete",
                "final_state": result
            }
        except Exception as e:
             logger.error(f"ESG Execution Failed: {e}")
             return {"error": str(e)}

    async def _run_compliance_flow(self, query: str):
        logger.info("Engaging Compliance Graph...")

        # Mock Extraction
        entity = "Generic Bank"
        jurisdiction = "US"
        if "eu" in query.lower() or "gdpr" in query.lower(): jurisdiction = "EU"

        initial_state = init_compliance_state(entity, jurisdiction)

        try:
            config = {"configurable": {"thread_id": "1"}}
            if hasattr(compliance_graph_app, 'ainvoke'):
                result = await compliance_graph_app.ainvoke(initial_state, config=config)
            else:
                result = compliance_graph_app.invoke(initial_state, config=config)
            return {
                "status": "Compliance Check Complete",
                "final_state": result
            }
        except Exception as e:
            logger.error(f"Compliance Execution Failed: {e}")
            return {"error": str(e)}

    async def _run_crisis_flow(self, query: str):
        logger.info("Engaging Crisis Simulation Graph...")

        # Mock Portfolio
        portfolio = {"id": "Main Fund", "aum": 100_000_000}

        initial_state = init_crisis_state(query, portfolio)

        try:
            config = {"configurable": {"thread_id": "1"}}
            if hasattr(crisis_simulation_app, 'ainvoke'):
                result = await crisis_simulation_app.ainvoke(initial_state, config=config)
            else:
                result = crisis_simulation_app.invoke(initial_state, config=config)
            return {
                "status": "Crisis Simulation Complete",
                "final_state": result
            }
        except Exception as e:
            logger.error(f"Crisis Simulation Failed: {e}")
            return {"error": str(e)}

    async def _run_fo_market(self, query: str):
        logger.info("Engaging FO Super-App Market Module...")
        # Simple extraction logic
        words = query.split()
        symbol = "AAPL" # Default
        for w in words:
            if w.isupper() and len(w) <= 5 and w.isalpha():
                symbol = w

        if "price" in query.lower() or "quote" in query.lower():
            result = self.mcp_registry.invoke("price_asset", symbol=symbol, side="mid")
        else:
            result = self.mcp_registry.invoke("retrieve_market_data", symbol=symbol)

        return {
            "status": "FO Market Data Retrieved",
            "data": result
        }

    async def _run_fo_execution(self, query: str):
        logger.info("Engaging FO Super-App Execution Module...")
        # Simple extraction
        words = query.split()
        symbol = "AAPL"
        side = "buy"
        qty = 100

        if "sell" in query.lower(): side = "sell"

        for w in words:
            if w.isupper() and len(w) <= 5 and w.isalpha():
                symbol = w
            if w.isdigit():
                qty = float(w)

        result = self.mcp_registry.invoke("execute_order", order={"symbol": symbol, "side": side, "qty": qty})
        return {
            "status": "FO Execution Submitted",
            "report": result
        }

    async def _run_fo_wealth(self, query: str):
        logger.info("Engaging FO Super-App Wealth Module...")

        # Simple heuristic extraction
        # "Plan goal for Education target 500000 horizon 10"
        if "plan" in query.lower() and "goal" in query.lower():
            # Mock extraction
            goal_name = "General Wealth Goal"
            target = 1000000.0
            horizon = 10

            words = query.split()
            for i, w in enumerate(words):
                if w.isdigit():
                    if target == 1000000.0: target = float(w)
                    else: horizon = int(w)

            result = self.mcp_registry.invoke("plan_wealth_goal", goal_name=goal_name, target_amount=target, horizon_years=horizon, current_savings=target*0.1)
            return {"status": "Wealth Plan Generated", "plan": result}

        elif "ips" in query.lower() or "governance" in query.lower():
            result = self.mcp_registry.invoke("generate_ips", family_name="Smith", risk_profile="Growth", goals=["Preserve Capital", "Growth"], constraints=["ESG"])
            return {"status": "IPS Generated", "ips": result}

        return {"status": "Wealth Module: Unknown Action"}

    async def _run_fo_deal(self, query: str):
        logger.info("Engaging FO Super-App Deal Flow Module...")

        # "Screen deal Project X sector Tech val 50 ebitda 5"
        deal_name = "Project Alpha"
        sector = "Technology"
        val = 100.0
        ebitda = 10.0

        words = query.split()
        nums = [float(s) for s in words if s.replace('.', '', 1).isdigit()]
        if len(nums) >= 1: val = nums[0]
        if len(nums) >= 2: ebitda = nums[1]

        result = self.mcp_registry.invoke("screen_deal", deal_name=deal_name, sector=sector, valuation=val, ebitda=ebitda)
        return {"status": "Deal Screened", "result": result}
