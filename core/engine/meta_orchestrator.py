"""
core/engine/meta_orchestrator.py

Agent Notes (Meta-Commentary):
The MetaOrchestrator is the supreme controller of the Adam system (v24.0 Architecture).
It implements the "Brain" of the architecture, deciding which cognitive path a user query should take.

Architectural Evolution:
1. Fast Path (v21): Direct tool execution (e.g., "Get stock price").
2. Async Path (v22): Message-driven workflow (e.g., "Monitor news").
3. Adaptive Path (v23): Neuro-Symbolic Planner (e.g., "Analyze complex credit risk").
4. Specialized Paths (v23.5): ESG, Compliance, Red Team, Crisis, Surveillance.
5. Deep Dive Omni-Path (v24.0): 5-Phase autonomous financial analysis with Graph -> Native -> Legacy failovers.
6. Swarm Intelligence: Parallelized Hive Mind processing.
"""

import logging
import asyncio
import re
import uuid
import json
from typing import Dict, Any, Optional, Callable, Awaitable

# --- Core Imports ---
from core.schemas.hnasp import HNASPState, Meta as HNASPMeta, PersonaState, LogicLayer
from core.engine.neuro_symbolic_planner import NeuroSymbolicPlanner
from core.engine.states import (
    init_risk_state, init_esg_state, init_compliance_state, 
    init_crisis_state, init_omniscient_state, init_surveillance_state,
    init_reflector_state
)

# --- Graph Applications ---
from core.engine.red_team_graph import red_team_app
from core.engine.esg_graph import esg_graph_app
from core.engine.regulatory_compliance_graph import compliance_graph_app
from core.engine.crisis_simulation_graph import crisis_simulation_app
from core.engine.deep_dive_graph import deep_dive_app
from core.engine.surveillance_graph import surveillance_graph_app
from core.engine.reflector_graph import reflector_app

# --- Orchestration & Tools ---
from core.system.agent_orchestrator import AgentOrchestrator
from core.mcp.registry import MCPRegistry
from core.llm_plugin import LLMPlugin
from core.engine.autonomous_self_improvement import CodeAlchemist
from core.engine.swarm.hive_mind import HiveMind
from core.engine.semantic_router import SemanticRouter
from core.utils.logging_utils import SwarmLogger
from core.utils.repo_context import RepoContextManager
from core.utils.proof_of_thought import ProofOfThoughtLogger

# --- v23.5 Deep Dive Agents (Legacy Support & Fallback) ---
from core.agents.specialized.management_assessment_agent import ManagementAssessmentAgent
from core.agents.fundamental_analyst_agent import FundamentalAnalystAgent
from core.agents.specialized.peer_comparison_agent import PeerComparisonAgent
from core.agents.specialized.credit_snc import SNCRatingAgent 
from core.agents.specialized.financial_covenant_agent import CovenantAnalystAgent 
from core.agents.specialized.monte_carlo_risk_agent import MonteCarloRiskAgent
from core.agents.specialized.quantum_scenario_agent import QuantumScenarioAgent
from core.agents.specialized.portfolio_manager_agent import PortfolioManagerAgent

# --- Schemas ---
from core.schemas.v23_5_schema import (
    V23KnowledgeGraph, Meta, Nodes, HyperDimensionalKnowledgeGraph, 
    EquityAnalysis, Fundamentals, ValuationEngine, DCFModel, PriceTargets, 
    CreditAnalysis, CovenantRiskAnalysis
)

logger = logging.getLogger(__name__)

class MetaOrchestrator:
    def __init__(self, legacy_orchestrator: AgentOrchestrator = None):
        """
        Initialize the MetaOrchestrator with all sub-engines and the routing registry.
        """
        self.swarm_logger = SwarmLogger()
        self.swarm_logger.log_event("SYSTEM_BOOT", "MetaOrchestrator", {"version": "v24.0.1-Alpha"})

        # Project OMEGA: Trust Engine
        self.pot_logger = ProofOfThoughtLogger()

        # 1. Cognitive Engines
        self.planner = NeuroSymbolicPlanner()
        self.code_alchemist = CodeAlchemist()
        self.hive_mind = HiveMind(worker_count=5)
        self.semantic_router = SemanticRouter()
        
        # 2. Tooling & Context
        self.mcp_registry = MCPRegistry()  # FO Super-App Integration
        self.repo_context = RepoContextManager()
        self.llm_plugin = LLMPlugin(config={
            "provider": "gemini", 
            "gemini_model_name": "gemini-3-pro"
        })

        # 3. Legacy Support
        try:
            self.legacy_orchestrator = legacy_orchestrator or AgentOrchestrator()
        except Exception as e:
            logger.error(f"Failed to initialize AgentOrchestrator: {e}")
            self.legacy_orchestrator = None

        # 4. Dynamic Route Registry
        # Maps complexity keys to executable async methods
        self._route_registry: Dict[str, Callable[[str, Dict], Awaitable[Any]]] = {
            "DEEP_DIVE": self._run_deep_dive_flow,
            "SWARM": self._run_swarm_flow,
            "CODE_GEN": self._run_code_gen_flow,
            "RED_TEAM": self._run_red_team_flow,
            "CRISIS": self._run_crisis_flow,
            "SURVEILLANCE": self._run_surveillance_flow,
            "ESG": self._run_esg_flow,
            "COMPLIANCE": self._run_compliance_flow,
            "FO_WEALTH": self._run_fo_wealth,
            "FO_DEAL": self._run_fo_deal,
            "FO_EXECUTION": self._run_fo_execution,
            "FO_MARKET": self._run_fo_market,
            "HIGH": self._run_adaptive_flow,
        }
        
        logger.info("MetaOrchestrator initialized with Dynamic Routing Registry.")

        # Project OMEGA Roadmap Note:
        # Phase 1 of AdamOS migration will involve replacing this Python class with a Rust-based Kernel
        # (see core/experimental/adamos_kernel/). The `route_registry` will eventually be
        # handled by the WASM plugin system.
        # Current status: PROTOTYPING.

    async def route_request(self, query: str, context: Optional[Dict[str, Any]] = None) -> Any:
        """
        The Central Nervous System: Analyzes query complexity, injects context, 
        and routes to the optimal cognitive engine.
        """
        # 1. Context Enrichment (v24 Feature)
        context = await self._enrich_context(query, context)
        self.swarm_logger.log_event("REQUEST_RECEIVED", "MetaOrchestrator", {"query": query, "context_keys": list(context.keys())})

        # 2. Complexity Assessment (Hybrid: Heuristic -> Semantic)
        complexity = self._assess_complexity(query, context)
        
        # If Heuristics are uncertain, engage Semantic Router
        if complexity == "LOW" or complexity == "UNKNOWN":
            logger.info("Heuristic check indeterminate. Engaging SemanticRouter.")
            complexity = self.semantic_router.route(query)

        logger.info(f"MetaOrchestrator: Query routed to [{complexity}]")
        self.swarm_logger.log_thought("MetaOrchestrator", f"Assessed complexity as {complexity}. Dispatching to engine.")
        self.pot_logger.log_thought("MetaOrchestrator", f"ROUTING_DECISION: {complexity}", metadata={"query": query})

        # 3. Execution via Registry
        result = None
        handler = self._route_registry.get(complexity)

        try:
            if handler:
                # Most handlers take query + context, some might just take query.
                # We inspect or just pass both if flexible, here we standardise on passing both.
                # For handlers that defined only query, we adapt.
                try:
                    result = await handler(query, context)
                except TypeError:
                    # Fallback for handlers that don't accept context
                    result = await handler(query)
            
            elif complexity == "MEDIUM":
                # Legacy Workflow Route
                result = await self._run_legacy_workflow(query, context)
            
            else:
                # Fallback / Low Complexity
                result = await self._run_legacy_single_agent(query, context)

        except Exception as e:
            logger.critical(f"Routing Execution Failed for {complexity}: {e}", exc_info=True)
            result = {"error": str(e), "status": "Routing Failed", "fallback": "Manual Intervention Required"}

        # 4. Meta-Cognition (Reflection)
        # Only reflect on complex outputs or if explicitly requested
        if complexity in ["DEEP_DIVE", "HIGH", "CRISIS", "RED_TEAM", "CODE_GEN"] or context.get("force_reflection"):
            result = await self._reflect_on_result(result, query)

        # Log final outcome hash
        self.pot_logger.log_thought("MetaOrchestrator", "EXECUTION_COMPLETE", metadata={"status": "success", "complexity": complexity})

        return result

    async def _enrich_context(self, query: str, context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """
        v24 Architecture: Injects Repo Context and System Awareness into the request before routing.
        """
        if context is None:
            context = {}
        
        # If system prompt is missing, inject high-level goals from Repo Context
        if "system_prompt" not in context:
            try:
                # Lightweight retrieval from repo context (e.g., fetch AGENTS.md or relevant docs)
                repo_knowledge = self.repo_context.get_context_for_query(query)
                if repo_knowledge:
                    context["repo_context"] = repo_knowledge
                    logger.debug("Injected Repo Context into request.")
            except Exception as e:
                logger.warning(f"Context enrichment failed: {e}")
        
        return context

    def _assess_complexity(self, query: str, context: Dict[str, Any] = None) -> str:
        """
        Fast Path Heuristic Routing.
        Determines the intent based on keywords and context flags.
        """
        query_lower = query.lower()
        context = context or {}

        # 1. Explicit Overrides from Context
        if context.get("force_route"):
            return context["force_route"]

        # 2. Code Generation / Refactoring
        if any(x in query_lower for x in ["generate code", "write a function", "implement", "refactor", "code agent", "fix bug"]):
            return "CODE_GEN"

        # 3. Swarm Trigger
        if any(x in query_lower for x in ["swarm", "scan", "parallel", "hive", "scout"]):
            return "SWARM"

        # 4. v23.5 Deep Dive Trigger
        if any(x in query_lower for x in ["deep dive", "full analysis", "partner", "valuation", "covenant", "comprehensive report"]):
            return "DEEP_DIVE"

        # 5. Keyword Mapping
        triggers = {
            "RED_TEAM": ["attack", "adversarial", "red team", "vulnerability", "exploit", "simulation", "wargame"],
            "CRISIS": ["crisis", "shock", "stress test", "scenario", "black swan", "market crash"],
            "SURVEILLANCE": ["surveillance", "watch", "zombie", "distressed", "monitor", "alert", "weakest link"],
            "ESG": ["esg", "environmental", "sustainability", "carbon", "green", "social impact", "governance"],
            "COMPLIANCE": ["compliance", "kyc", "aml", "regulation", "violation", "dodd-frank", "basel", "sec filing"],
            "FO_WEALTH": ["ips", "trust", "estate", "wealth", "goal", "retirement", "succession"],
            "FO_DEAL": ["deal", "private equity", "venture", "screening", "multiple", "cap table"],
            "FO_EXECUTION": ["buy", "sell", "execute trade", "place order", "limit order"],
            "FO_MARKET": ["price", "quote", "ticker", "market data", "volume", "chart"],
            "HIGH": ["analyze", "plan", "strategy", "report", "architect", "design"],
            "MEDIUM": ["summary", "digest", "update", "check"]
        }

        for category, keywords in triggers.items():
            if any(k in query_lower for k in keywords):
                return category

        return "LOW"

    async def _reflect_on_result(self, result: Any, query: str) -> Any:
        """
        Runs the ReflectorGraph to critique and potentially refine the result.
        """
        logger.info("Engaging Reflector (Meta-Cognition)...")
        
        # Prepare content
        content_to_reflect = str(result)
        if isinstance(result, dict):
            content_to_reflect = result.get("human_readable_status") or result.get("final_report") or str(result)

        initial_state = init_reflector_state(content_to_reflect, context={"original_query": query})

        try:
            config = {"configurable": {"thread_id": f"reflector_{uuid.uuid4()}"}}
            if hasattr(reflector_app, 'ainvoke'):
                reflection = await reflector_app.ainvoke(initial_state, config=config)
            else:
                reflection = reflector_app.invoke(initial_state, config=config)

            # Attach reflection to result
            if isinstance(result, dict):
                result["meta_reflection"] = reflection
                # Future v25: If reflection score is low, auto-trigger a retry loop here.
                if reflection.get("score", 10) < 4:
                    logger.warning("Reflection Score Low. Marking for Human Review.")
                    result["flag_for_review"] = True

            logger.info("Reflection Complete.")
            return result

        except Exception as e:
            logger.warning(f"Reflection failed: {e}. Returning original result.")
            return result

    # =========================================================================
    #  Execution Flows
    # =========================================================================

    async def _run_deep_dive_flow(self, query: str, context: Optional[Dict[str, Any]] = None):
        """
        The "Apex" Analysis Flow.
        Architecture: Waterfall Resilience.
        1. Attempt v23.5 Graph (Preferred).
        2. Fallback to Native Gemini (Path 3).
        3. Fallback to Legacy Agents (Manual Orchestration).
        """
        logger.info("Engaging v24 Deep Dive Protocol...")
        
        # 0. Swarm Reconnaissance (Parallel Data Gathering)
        # We launch this regardless of the subsequent path to warm up data caches
        if not self.hive_mind.workers:
            await self.hive_mind.initialize()
        
        # Async fire-and-forget for swarm scouts
        asyncio.create_task(self.hive_mind.disperse_task(
            "ANALYST", 
            {"target": query, "purpose": "preliminary_deep_dive"}, 
            intensity=5.0
        ))

        ticker = self._extract_ticker(query)
        initial_state = init_omniscient_state(ticker)

        # Path 1: Graph Execution
        try:
            logger.info("Attempting Deep Dive Graph Execution...")
            config = {"configurable": {"thread_id": str(uuid.uuid4())}}
            
            if deep_dive_app:
                result = await deep_dive_app.ainvoke(initial_state, config=config)
                logger.info("Deep Dive Graph Execution Successful.")
                return result
            else:
                raise Exception("Graph App not initialized")

        except Exception as e:
            logger.error(f"Deep Dive Graph Failed: {e}. Initiating Fallback 1 (Native Gemini).", exc_info=True)
            return await self._run_deep_dive_manual_fallback(query)

    async def _run_deep_dive_manual_fallback(self, query: str):
        """
        Fallback 1: Native Gemini (Structured Output Mode).
        """
        logger.info("Engaging Deep Dive Native Gemini Flow...")
        try:
            prompt = f"Perform a deep dive analysis for: {query}. Populate the full HyperDimensionalKnowledgeGraph schema."
            tools = list(self.mcp_registry.tools.values())

            result_obj, metadata = self.llm_plugin.generate_structured(
                prompt,
                response_schema=HyperDimensionalKnowledgeGraph,
                tools=tools,
                thinking_level="high"
            )
            logger.info(f"Gemini Native Execution Complete.")
            return result_obj.model_dump(by_alias=True)

        except Exception as e:
            logger.error(f"Native Gemini Flow Failed: {e}. Initiating Fallback 2 (Legacy Agents).", exc_info=True)
            return await self._run_deep_dive_legacy_agents(query)

    async def _run_deep_dive_legacy_agents(self, query: str):
        """
        Fallback 2: Legacy Sequential Agent Execution.
        Manually coordinates 7 agents to build the report.
        """
        logger.info("Engaging Legacy Agent Deep Dive...")
        try:
            trace_id = str(uuid.uuid4())
            common_config = {"trace_id": trace_id}
            
            # --- Initialize Agents ---
            agents = {
                "mgmt": ManagementAssessmentAgent(config=common_config),
                "fund": FundamentalAnalystAgent(config=common_config),
                "peer": PeerComparisonAgent(config=common_config),
                "snc": SNCRatingAgent(config=common_config),
                "mc": MonteCarloRiskAgent(config=common_config),
                "quant": QuantumScenarioAgent(config=common_config),
                "pm": PortfolioManagerAgent(config=common_config)
            }

            # --- Execution Steps ---
            # 1. Entity
            company_match = re.search(r"analysis for (.*)", query, re.IGNORECASE)
            company_name = company_match.group(1).strip(" .") if company_match else query
            entity_eco = await agents["mgmt"].execute({"company_name": company_name})
            
            company_id = entity_eco.legal_entity.name if hasattr(entity_eco, 'legal_entity') else company_name

            # 2. Fundamentals
            fund_data = await agents["fund"].execute(company_id)
            dcf_val = fund_data.get("dcf_valuation", 0.0) if isinstance(fund_data, dict) else 0.0
            raw_metrics = fund_data.get("raw_metrics", {}) if isinstance(fund_data, dict) else {}
            
            ebitda = raw_metrics.get("ebitda", 100.0)
            total_debt = raw_metrics.get("total_debt", 300.0)

            # 3. Peers (Safe Execute)
            try:
                multiples = await agents["peer"].execute({"company_id": company_id})
            except:
                multiples = {}

            # 4. Credit & Risk
            financials_input = {
                "ebitda": float(ebitda),
                "total_debt": float(total_debt),
                "interest_expense": float(ebitda / 4.0) if ebitda else 1.0
            }
            
            snc_model = await agents["snc"].execute(
                financials=financials_input,
                capital_structure=[{"name": "Term Loan", "amount": float(total_debt), "priority": 1}],
                enterprise_value=float(dcf_val)
            )

            sim_engine = await agents["mc"].execute(
                current_ebitda=ebitda,
                ebitda_volatility=0.2,
                interest_expense=financials_input["interest_expense"]
            )

            # 5. Synthesis
            # Construct partial schema objects
            equity_analysis = EquityAnalysis(
                fundamentals=Fundamentals(revenue_cagr_3yr="5%", ebitda_margin_trend="Stable"),
                valuation_engine=ValuationEngine(
                    dcf_model=DCFModel(intrinsic_value_estimate=float(dcf_val)),
                    multiples_analysis=multiples
                )
            )
            
            credit_analysis = CreditAnalysis(snc_rating_model=snc_model)

            nodes = Nodes(
                entity_ecosystem=entity_eco,
                equity_analysis=equity_analysis,
                credit_analysis=credit_analysis,
                simulation_engine=sim_engine
            )

            kg = V23KnowledgeGraph(
                meta=Meta(target=query, model_version="Adam-v24-Legacy-Fallback"),
                nodes=nodes
            )

            return HyperDimensionalKnowledgeGraph(v23_knowledge_graph=kg).model_dump(by_alias=True)

        except Exception as e:
            logger.critical(f"All Deep Dive methods failed. {e}", exc_info=True)
            return {"error": "Deep Dive Failure", "details": str(e)}

    # --- Specialized Flows ---

    async def _run_swarm_flow(self, query: str, context: Optional[Dict] = None):
        logger.info("Engaging Swarm Intelligence (Hive Mind)...")
        if not self.hive_mind.workers:
            await self.hive_mind.initialize()
        
        target = query.replace("swarm", "").strip()
        await self.hive_mind.disperse_task("ANALYST", {"target": target})
        results = await self.hive_mind.gather_results(timeout=5.0)
        
        return {
            "status": "Swarm Execution Complete",
            "worker_count": self.hive_mind.worker_count,
            "results": results
        }

    async def _run_code_gen_flow(self, query: str, context: Optional[Dict] = None):
        logger.info("Engaging Code Alchemist...")
        prompt = query.replace("generate code", "").strip()
        result = self.code_alchemist.generate_code(prompt)
        return {"status": "Code Generated", "result": result}

    async def _run_adaptive_flow(self, query: str, context: Optional[Dict] = None):
        logger.info("Engaging Neuro-Symbolic Planner...")
        # 1. Plan
        start_node = self._extract_entity(query)
        plan_data = self.planner.discover_plan(start_node, "RiskScore")
        
        # 2. Compile
        if not plan_data: return {"error": "Planning Failed"}
        app = self.planner.to_executable_graph(plan_data)
        
        # 3. Execute
        initial_state = {
            "request": query, 
            "plan": {"id": "p1", "steps": plan_data.get("steps", [])},
            "current_task_index": 0
        }
        
        config = {"configurable": {"thread_id": "adaptive_1"}}
        result = await app.ainvoke(initial_state, config=config)
        return {"status": "Adaptive Execution Complete", "final_state": result}

    # --- Graph Wrappers ---

    async def _run_red_team_flow(self, query: str, context: Optional[Dict] = None):
        return await self._generic_graph_run(red_team_app, {
            "target_entity": self._extract_entity(query),
            "scenario_type": "Cyber" if "cyber" in query else "Macro",
            "severity_threshold": 8.0
        }, "Red Team")

    async def _run_esg_flow(self, query: str, context: Optional[Dict] = None):
        return await self._generic_graph_run(esg_graph_app, 
            init_esg_state(self._extract_entity(query), "General"), "ESG")

    async def _run_compliance_flow(self, query: str, context: Optional[Dict] = None):
        return await self._generic_graph_run(compliance_graph_app, 
            init_compliance_state(self._extract_entity(query), "US"), "Compliance")

    async def _run_crisis_flow(self, query: str, context: Optional[Dict] = None):
        portfolio = {"id": "Main Fund", "aum": 100_000_000}
        return await self._generic_graph_run(crisis_simulation_app, 
            init_crisis_state(query, portfolio), "Crisis Simulation")

    async def _run_surveillance_flow(self, query: str, context: Optional[Dict] = None):
        return await self._generic_graph_run(surveillance_graph_app, 
            init_surveillance_state(sector="TMT" if "tech" in query else "General"), "Surveillance")

    async def _generic_graph_run(self, app, initial_state, name):
        """Helper to reduce boilerplate for Graph execution."""
        logger.info(f"Engaging {name} Graph...")
        try:
            config = {"configurable": {"thread_id": str(uuid.uuid4())}}
            if hasattr(app, 'ainvoke'):
                result = await app.ainvoke(initial_state, config=config)
            else:
                result = app.invoke(initial_state, config=config)
            return {"status": f"{name} Complete", "final_state": result}
        except Exception as e:
            logger.error(f"{name} Failed: {e}")
            return {"error": str(e)}

    # --- Family Office (FO) Super-App Modules ---

    async def _run_fo_market(self, query: str, context: Optional[Dict] = None):
        symbol = self._extract_ticker(query) or "AAPL"
        if "price" in query.lower():
            result = self.mcp_registry.invoke("price_asset", symbol=symbol, side="mid")
        else:
            result = self.mcp_registry.invoke("retrieve_market_data", symbol=symbol)
        return {"status": "FO Market Data", "data": result}

    async def _run_fo_execution(self, query: str, context: Optional[Dict] = None):
        symbol = self._extract_ticker(query) or "AAPL"
        side = "sell" if "sell" in query.lower() else "buy"
        # Extract qty roughly
        qty = 100
        for w in query.split():
            if w.isdigit(): qty = float(w)
            
        result = self.mcp_registry.invoke("execute_order", order={"symbol": symbol, "side": side, "qty": qty})
        return {"status": "FO Order Submitted", "report": result}

    async def _run_fo_wealth(self, query: str, context: Optional[Dict] = None):
        if "ips" in query.lower():
            result = self.mcp_registry.invoke("generate_ips", family_name="User", risk_profile="Growth")
            return {"status": "IPS Generated", "ips": result}
        
        # Default goal plan
        result = self.mcp_registry.invoke("plan_wealth_goal", goal_name="General", target_amount=1e6, horizon_years=10)
        return {"status": "Wealth Plan Generated", "plan": result}

    async def _run_fo_deal(self, query: str, context: Optional[Dict] = None):
        result = self.mcp_registry.invoke("screen_deal", deal_name="Opportunity", sector="Tech", valuation=100.0)
        return {"status": "Deal Screened", "result": result}

    # --- Legacy Helpers ---

    async def _run_legacy_workflow(self, query: str, context: Optional[Dict] = None):
        if self.legacy_orchestrator:
            return await self.legacy_orchestrator.execute_workflow("test_workflow", initial_context={"user_query": query})
        return {"status": "Legacy Orchestrator Unavailable", "query": query}

    async def _run_legacy_single_agent(self, query: str, context: Optional[Dict] = None):
        if self.legacy_orchestrator:
            self.legacy_orchestrator.execute_agent("QueryUnderstandingAgent", context={"user_query": query})
            return {"status": "Dispatched to Message Broker", "query": query}
        return {"status": "Legacy Orchestrator Unavailable", "query": query}

    # --- Utilities ---

    def _extract_ticker(self, query: str) -> str:
        """Simple Regex Ticker Extractor"""
        # Look for $TICKER or known names
        if "apple" in query.lower(): return "AAPL"
        if "tesla" in query.lower(): return "TSLA"
        if "microsoft" in query.lower(): return "MSFT"
        # Try to find uppercase word
        for w in query.split():
            if w.isupper() and 2 <= len(w) <= 5 and w.isalpha():
                return w
        return "SPY"

    def _extract_entity(self, query: str) -> str:
        """Simple Entity Extractor"""
        match = re.search(r"((?:[A-Z][a-z0-9]+\s?)+)", query)
        if match:
            candidate = match.group(1).strip()
            ignored = {"Analyze", "Perform", "The", "What", "Risk"}
            if candidate not in ignored:
                return candidate
        return self._extract_ticker(query)