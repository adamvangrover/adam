# core/v23_graph_engine/meta_orchestrator.py

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

from core.v23_graph_engine.neuro_symbolic_planner import NeuroSymbolicPlanner
from core.v23_graph_engine.states import init_risk_state, init_esg_state, init_compliance_state, init_crisis_state
from core.v23_graph_engine.red_team_graph import red_team_app
from core.v23_graph_engine.esg_graph import esg_graph_app
from core.v23_graph_engine.regulatory_compliance_graph import compliance_graph_app
from core.v23_graph_engine.crisis_simulation_graph import crisis_simulation_app
from core.system.agent_orchestrator import AgentOrchestrator

# v23.5 Deep Dive Agents
from core.agents.specialized.ManagementAssessmentAgent import ManagementAssessmentAgent
from core.agents.fundamental_analyst_agent import FundamentalAnalystAgent
from core.agents.specialized.PeerComparisonAgent import PeerComparisonAgent
from core.agents.specialized.SNCRatingAgent import SNCRatingAgent
from core.agents.specialized.CovenantAnalystAgent import CovenantAnalystAgent
from core.agents.specialized.MonteCarloRiskAgent import MonteCarloRiskAgent
from core.agents.specialized.QuantumScenarioAgent import QuantumScenarioAgent
from core.agents.specialized.PortfolioManagerAgent import PortfolioManagerAgent
from core.schemas.v23_5_schema import V23KnowledgeGraph, Meta, Nodes, HyperDimensionalKnowledgeGraph, EquityAnalysis, Fundamentals, ValuationEngine, DCFModel, PriceTargets

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
        complexity = self._assess_complexity(query, context)
        logger.info(f"MetaOrchestrator: Query complexity is {complexity}")
        
        if complexity == "DEEP_DIVE":
            return await self._run_deep_dive_flow(query)
        elif complexity == "RED_TEAM":
            return await self._run_red_team_flow(query)
        elif complexity == "CRISIS":
            return await self._run_crisis_flow(query)
        elif complexity == "ESG":
            return await self._run_esg_flow(query)
        elif complexity == "COMPLIANCE":
            return await self._run_compliance_flow(query)
        elif complexity == "HIGH":
            return await self._run_adaptive_flow(query)
        elif complexity == "MEDIUM":
            # Route to legacy workflow (Async v22 style via AgentOrchestrator)
            logger.info("Routing to Legacy/Async Workflow...")
            return await self.legacy_orchestrator.execute_workflow("test_workflow", initial_context={"user_query": query})
        else:
            # Low complexity -> Legacy Single Agent Execution
            logger.info("Routing to Legacy Single Agent...")
            self.legacy_orchestrator.execute_agent("QueryUnderstandingAgent", context={"user_query": query})
            return {"status": "Dispatched to Message Broker", "query": query}

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

        # v23 is best for analysis, planning, and risk
        if any(x in query_lower for x in ["analyze", "risk", "plan", "strategy", "report"]):
            return "HIGH"
        # v22 is best for monitoring, alerts, background tasks
        elif any(x in query_lower for x in ["monitor", "alert", "watch", "notify"]):
            return "MEDIUM"
        # v21 is best for simple lookups
        return "LOW"

    async def _run_deep_dive_flow(self, query: str):
        logger.info("Engaging Deep Dive Protocol (v23.5)...")

        # Initialize Agents
        mgmt_agent = ManagementAssessmentAgent(config={})
        fund_agent = FundamentalAnalystAgent(config={})
        peer_agent = PeerComparisonAgent(config={})
        snc_agent = SNCRatingAgent(config={})
        # cov_agent = CovenantAnalystAgent(config={})
        mc_agent = MonteCarloRiskAgent(config={})
        quant_agent = QuantumScenarioAgent(config={})
        pm_agent = PortfolioManagerAgent(config={})

        try:
            # Phase 1: Entity & Management
            logger.info("Phase 1: Entity & Management")
            # In a real system, we'd extract the company name precisely using an LLM or Named Entity Recognition
            # Here we assume the query might contain it or we use the query as is.
            # A simplistic heuristic: use the query as the company name.
            company_name_extraction = query.replace("Perform a deep dive analysis on", "").strip() or query

            entity_eco = await mgmt_agent.execute({"company_name": company_name_extraction})

            # Phase 2: Deep Fundamental
            logger.info("Phase 2: Deep Fundamental")
            company_id = company_name_extraction # Using name as ID for now
            fund_data = await fund_agent.execute(company_id)

            # Map fund_data to EquityAnalysis
            dcf_val = 0.0
            if isinstance(fund_data, dict):
                 # Safely extract with fallback
                 dcf_val = fund_data.get("dcf_valuation", 0.0)
                 if dcf_val is None: dcf_val = 0.0

            equity_analysis = EquityAnalysis(
                fundamentals=Fundamentals(
                    revenue_cagr_3yr="5%", # Mock extraction
                    ebitda_margin_trend="Expanding"
                ),
                valuation_engine=ValuationEngine(
                    dcf_model=DCFModel(
                        wacc=0.08,
                        terminal_growth=0.02,
                        intrinsic_value=float(dcf_val)
                    ),
                    multiples_analysis=await peer_agent.execute({"company_id": company_id}),
                    price_targets=PriceTargets(bear_case=80.0, base_case=100.0, bull_case=120.0)
                )
            )

            # Phase 3: Credit
            logger.info("Phase 3: Credit")
            credit_input = {
                "facilities": [{"id": "TLB", "amount": "500M", "ltv": 0.5}],
                "current_leverage": 3.0
            }
            credit_analysis = await snc_agent.execute(credit_input)

            # Phase 4: Risk
            logger.info("Phase 4: Risk")
            # Monte Carlo Agent needs EBITDA, Debt, Volatility.
            # We should try to get these from fund_data if available.
            ebitda = 100.0
            debt = 300.0
            if isinstance(fund_data, dict):
                 # Attempt to extract if structure matches mock data from FundamentalAnalystAgent
                 # fund_data["financial_ratios"] might have it? Or raw data?
                 pass

            sim_engine = await mc_agent.execute({"ebitda": ebitda, "debt": debt, "volatility": 0.2})
            scenarios = await quant_agent.execute({})
            sim_engine.quantum_scenarios = scenarios

            # Phase 5: Synthesis
            logger.info("Phase 5: Synthesis")
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
                meta=Meta(target=query, generated_at="2023-10-27", model_version="Adam-v23.5"),
                nodes=nodes
            )

            result = HyperDimensionalKnowledgeGraph(v23_knowledge_graph=kg)
            return result.model_dump(by_alias=True)

        except Exception as e:
            logger.error(f"Deep Dive Execution Failed: {e}", exc_info=True)
            return {"error": str(e)}

    # ... (rest of methods)
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
