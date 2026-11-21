"""
core/engine/meta_orchestrator.py

Agent Notes (Meta-Commentary):
The MetaOrchestrator is the supreme controller of the Adam system (v26.0 Architecture).
It implements the "Brain" of the architecture, deciding which cognitive path a user query should take.

Architectural Evolution:
1. Fast Path (v21): Direct tool execution (e.g., "Get stock price").
2. Async Path (v22): Message-driven workflow (e.g., "Summarize today's digest").
3. Adaptive Path (v23): Neuro-Symbolic Planner (e.g., "Analyze complex credit risk").
4. Specialized Paths (v23.5): ESG, Compliance, Red Team, Crisis, Surveillance.
5. Deep Dive Omni-Path (v24.0): 5-Phase autonomous financial analysis with Graph -> Native -> Legacy failovers.
6. Swarm Intelligence: Parallelized Hive Mind processing.
"""

import asyncio
import dataclasses
import datetime
import logging
import math
import re
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, FrozenSet, List, Optional, Pattern, Tuple

from core.agents.fundamental_analyst_agent import FundamentalAnalystAgent
from core.agents.portfolio_optimization_agent import PortfolioOptimizationAgent
from core.agents.specialized.credit_snc import SNCRatingAgent

# --- v23.5 Deep Dive Agents (Legacy Support & Fallback) ---
from core.agents.specialized.management_assessment_agent import ManagementAssessmentAgent
from core.agents.specialized.monte_carlo_risk_agent import MonteCarloRiskAgent
from core.agents.specialized.peer_comparison_agent import PeerComparisonAgent
from core.agents.specialized.portfolio_manager_agent import PortfolioManagerAgent
from core.agents.specialized.quantum_scenario_agent import QuantumScenarioAgent
from core.config.models import BLOCKED_MODELS, SR11_7_BLOCKED_MODEL_ERROR, RouterConfig
from core.engine.autonomous_self_improvement import CodeAlchemist
from core.engine.crisis_simulation_graph import crisis_simulation_app
from core.engine.deep_dive_graph import deep_dive_app
from core.engine.esg_graph import esg_graph_app
from core.engine.neuro_symbolic_planner import NeuroSymbolicPlanner

# --- Graph Applications ---
from core.engine.red_team_graph import red_team_app
from core.engine.reflector_graph import reflector_app
from core.engine.regulatory_compliance_graph import compliance_graph_app
from core.engine.semantic_router import SemanticRouter
from core.engine.states import (
    init_compliance_state,
    init_crisis_state,
    init_esg_state,
    init_omniscient_state,
    init_reflector_state,
    init_surveillance_state,
)
from core.engine.surveillance_graph import surveillance_graph_app
from core.engine.swarm.hive_mind import HiveMind
from core.engine.swarm.mirofish_engine import MiroFishSwarmEngine
from core.llm_plugin import LLMPlugin
from core.market_data.service import MarketDataService
from core.mcp.registry import MCPRegistry

# --- Schemas ---
from core.schemas.v23_5_schema import (
    CreditAnalysis,
    DCFModel,
    EquityAnalysis,
    Fundamentals,
    HyperDimensionalKnowledgeGraph,
    Meta,
    Nodes,
    V26KnowledgeGraph,
    ValuationEngine,
)

# --- Orchestration & Tools ---
from core.system.agent_orchestrator import AgentOrchestrator
from core.utils.logging_utils import SwarmLogger
from core.utils.proof_of_thought import ProofOfThoughtLogger
from core.utils.repo_context import RepoContextManager

import yaml

logger = logging.getLogger(__name__)


class Route(str, Enum):
    DEEP_DIVE = "DEEP_DIVE"
    SWARM = "SWARM"
    CODE_GEN = "CODE_GEN"
    RED_TEAM = "RED_TEAM"
    CRISIS = "CRISIS"
    SURVEILLANCE = "SURVEILLANCE"
    ESG = "ESG"
    COMPLIANCE = "COMPLIANCE"
    FO_WEALTH = "FO_WEALTH"
    FO_DEAL = "FO_DEAL"
    FO_EXECUTION = "FO_EXECUTION"
    FO_MARKET = "FO_MARKET"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"

    @classmethod
    def coerce(cls, value: Any) -> Optional["Route"]:
        if isinstance(value, cls):
            return value
        try:
            return cls(str(value).upper())
        except ValueError:
            return None


def _rx(*terms: str) -> Pattern[str]:
    return re.compile(r"\b(?:" + "|".join(terms) + r")\b", re.IGNORECASE)


# Ordered: first match wins. Ambiguous everyday words (monitor, watch, alert, report,
# simulation, scenario, scan, parallel, multiple, goal, check, plan, design) are
# deliberately excluded and left to the SemanticRouter.
ROUTE_PATTERNS: Tuple[Tuple[Route, Pattern[str]], ...] = (
    (Route.CODE_GEN, _rx(r"generate code", r"write a function", r"implement", r"refactor", r"code agent", r"fix bug")),
    (Route.SWARM, _rx(r"swarm", r"hive", r"scouts?")),
    (Route.DEEP_DIVE, _rx(r"deep dive", r"full analysis", r"partner", r"valuation", r"covenants?", r"comprehensive report")),
    (Route.RED_TEAM, _rx(r"attack", r"adversarial", r"red team", r"vulnerabilit(?:y|ies)", r"exploit", r"wargame")),
    (Route.CRISIS, _rx(r"crisis", r"shock", r"stress test", r"black swan", r"market crash")),
    (Route.SURVEILLANCE, _rx(r"surveillance", r"zombie", r"distressed", r"weakest link")),
    (Route.ESG, _rx(r"esg", r"environmental", r"sustainability", r"carbon", r"green", r"social impact", r"governance")),
    (Route.COMPLIANCE, _rx(r"compliance", r"kyc", r"aml", r"regulation", r"violation", r"dodd-frank", r"basel", r"sec filing")),
    (Route.FO_WEALTH, _rx(r"ips", r"trust", r"estate", r"wealth", r"retirement", r"succession")),
    (Route.FO_DEAL, _rx(r"deals?", r"private equity", r"venture", r"screening", r"cap table")),
    (Route.FO_EXECUTION, _rx(r"buy", r"sell", r"execute trade", r"place order", r"limit order")),
    (Route.FO_MARKET, _rx(r"prices?", r"quotes?", r"ticker", r"market data", r"volume", r"chart")),
    (Route.HIGH, _rx(r"analy[sz]e", r"strategy", r"architect")),
    (Route.MEDIUM, _rx(r"summary", r"digest", r"update")),
)

_DOLLAR_TICKER_RE = re.compile(r"\$([A-Za-z]{1,5})\b")
_UPPER_TICKER_RE = re.compile(r"\b[A-Z]{2,5}\b")
_PROPER_NOUN_RE = re.compile(r"\b[A-Z][\w&.-]*(?:\s+[A-Z][\w&.-]*)*")
_TARGET_PREP_RE = re.compile(r"\b(?:for|on|of|about|into)\s+$", re.IGNORECASE)
_QTY_RE = re.compile(
    r"\b(?:buy|sell|order(?:\s+for)?)\s+(\d+(?:\.\d+)?)(?![\w.])"
    r"|(?<![\w.$])(\d+(?:\.\d+)?)\s+(?:shares?|units?|contracts?)\b",
    re.IGNORECASE,
)

CLARIFICATION_NO_ENTITY = {"error": "no_entity_resolved", "status": "clarification_needed"}


def _default_aliases() -> Dict[str, str]:
    return {"apple": "AAPL", "tesla": "TSLA", "microsoft": "MSFT"}


def _default_non_tickers() -> FrozenSet[str]:
    return frozenset({
        "AI", "AML", "API", "CEO", "CFO", "DCF", "ESG", "ETF", "EU", "FO", "GDP", "IPO",
        "IPS", "KYC", "LBO", "PE", "SEC", "SNC", "UK", "US", "USA", "USD",
    })


def _default_entity_stopwords() -> FrozenSet[str]:
    return frozenset({
        "Analyze", "Analyse", "Assess", "Buy", "Check", "Evaluate", "Get", "How", "Is", "Perform",
        "Please", "Price", "Red", "Review", "Risk", "Run", "Sell", "Show", "The", "Team", "What",
    })


def _default_reflect_on() -> FrozenSet[Route]:
    return frozenset({Route.DEEP_DIVE, Route.HIGH, Route.CRISIS, Route.RED_TEAM, Route.CODE_GEN})


@dataclass
class OrchestratorConfig:
    llm_provider: str = "google"
    llm_model: str = "gemini-4-neon"
    routing_config_path: str = "config/llm_routing.yaml"
    hive_workers: int = 5
    mirofish_agents: int = 50
    default_portfolio_aum: float = 100_000_000
    default_universe: List[str] = field(default_factory=lambda: ["SPY", "TLT", "GLD", "HYG", "LQD"])
    reflect_on: FrozenSet[Route] = field(default_factory=_default_reflect_on)
    ticker_aliases: Dict[str, str] = field(default_factory=_default_aliases)
    non_ticker_tokens: FrozenSet[str] = field(default_factory=_default_non_tickers)
    entity_stopwords: FrozenSet[str] = field(default_factory=_default_entity_stopwords)
    price_history_days: int = 365
    universe_metadata_path: str = "config/default_universe_metadata.yaml"


def assert_model_allowed(model_id: str) -> str:
    """SR 11-7 gate: reject quarantined models before any LLM client is constructed."""
    clean = (model_id or "").strip().lower()
    if clean in BLOCKED_MODELS or "astra" in clean:
        raise SR11_7_BLOCKED_MODEL_ERROR(
            f"SR 11-7 VIOLATION: Model '{model_id}' is permanently quarantined and may not be used by MetaOrchestrator."
        )
    return model_id


# Maps OrchestratorConfig.llm_provider onto LLMPlugin provider keys.
_LLM_PLUGIN_PROVIDERS = {"google": "gemini", "gemini": "gemini", "openai": "openai", "cohere": "cohere"}


class MetaOrchestrator:
    def __init__(self, legacy_orchestrator: AgentOrchestrator = None, config: Optional[OrchestratorConfig] = None):
        """
        Initialize the MetaOrchestrator with all sub-engines and the routing registry.
        """
        self.config = self._resolve_config(config or OrchestratorConfig())

        self.swarm_logger = SwarmLogger()
        self.swarm_logger.log_event("SYSTEM_BOOT", "MetaOrchestrator", {"version": "v26.0.0-Beta"})

        # Project OMEGA: Trust Engine
        self.pot_logger = ProofOfThoughtLogger()

        # 1. Cognitive Engines
        self.planner = NeuroSymbolicPlanner()
        self.code_alchemist = CodeAlchemist()
        self.hive_mind = HiveMind(worker_count=self.config.hive_workers)
        self.mirofish_engine = MiroFishSwarmEngine(agent_count=self.config.mirofish_agents)
        self.semantic_router = SemanticRouter()
        self._background_tasks: set = set()

        # 2. Tooling & Context
        self.mcp_registry = MCPRegistry()  # FO Super-App Integration
        self.repo_context = RepoContextManager()
        plugin_provider = _LLM_PLUGIN_PROVIDERS.get(self.config.llm_provider.lower(), self.config.llm_provider.lower())
        self.llm_plugin = LLMPlugin(config={
            "provider": plugin_provider,
            f"{plugin_provider}_model_name": assert_model_allowed(self.config.llm_model),
        })

        # 3. Legacy Support
        try:
            self.legacy_orchestrator = legacy_orchestrator or AgentOrchestrator()
        except Exception as e:
            logger.error(f"Failed to initialize AgentOrchestrator: {e}")
            self.legacy_orchestrator = None

        # 4. Dynamic Route Registry: every Route dispatches through here.
        self._route_registry: Dict[Route, Callable[[str, Dict[str, Any]], Awaitable[Any]]] = {
            Route.DEEP_DIVE: self._run_deep_dive_flow,
            Route.SWARM: self._run_swarm_flow,
            Route.CODE_GEN: self._run_code_gen_flow,
            Route.RED_TEAM: self._run_red_team_flow,
            Route.CRISIS: self._run_crisis_flow,
            Route.SURVEILLANCE: self._run_surveillance_flow,
            Route.ESG: self._run_esg_flow,
            Route.COMPLIANCE: self._run_compliance_flow,
            Route.FO_WEALTH: self._run_fo_wealth,
            Route.FO_DEAL: self._run_fo_deal,
            Route.FO_EXECUTION: self._run_fo_execution,
            Route.FO_MARKET: self._run_fo_market,
            Route.HIGH: self._run_adaptive_flow,
            Route.MEDIUM: self._run_legacy_workflow,
            Route.LOW: self._run_legacy_single_agent,
        }

        logger.info("MetaOrchestrator initialized with Dynamic Routing Registry.")

        # Project OMEGA Roadmap Note:
        # Phase 1 of AdamOS migration will involve replacing this Python class with a Rust-based Kernel
        # (see core/experimental/adamos_kernel/). The `route_registry` will eventually be
        # handled by the WASM plugin system.
        # Current status: PROTOTYPING.

    @staticmethod
    def _resolve_config(config: OrchestratorConfig) -> OrchestratorConfig:
        """Applies the tier-1 System 2 model from the routing registry, if present."""
        try:
            router_config = RouterConfig.load_from_yaml(config.routing_config_path)
        except Exception as e:
            logger.warning(f"Routing config '{config.routing_config_path}' unavailable ({e}); using '{config.llm_model}'.")
            return config
        tier_1 = router_config.tier_1_primary_system2
        if tier_1 and tier_1.model_id:
            return dataclasses.replace(config, llm_model=tier_1.model_id)
        return config

    async def route_request(self, query: str, context: Optional[Dict[str, Any]] = None) -> Any:
        """
        The Central Nervous System: Analyzes query complexity, injects context,
        and routes to the optimal cognitive engine.
        """
        # 1. Context Enrichment (v24 Feature)
        context = await self._enrich_context(query, context)
        self.swarm_logger.log_event("REQUEST_RECEIVED", "MetaOrchestrator", {"query": query, "context_keys": list(context.keys())})

        # 2. Complexity Assessment (Hybrid: Heuristic -> Semantic)
        route = self._assess_complexity(query, context)

        # If Heuristics are uncertain (and the caller did not force a route), engage Semantic Router
        if route is Route.LOW and not context.get("force_route"):
            logger.info("Heuristic check indeterminate. Engaging SemanticRouter.")
            semantic = self.semantic_router.route(query)
            route = Route.coerce(semantic) or Route.LOW

        logger.info(f"MetaOrchestrator: Query routed to [{route.value}]")
        self.swarm_logger.log_thought("MetaOrchestrator", f"Assessed complexity as {route.value}. Dispatching to engine.")
        self.pot_logger.log_thought("MetaOrchestrator", f"ROUTING_DECISION: {route.value}", metadata={"query": query})

        # 3. Execution via Registry
        handler = self._route_registry[route]
        try:
            result = await handler(query, context)
        except Exception as e:
            logger.critical(f"Routing Execution Failed for {route.value}: {e}", exc_info=True)
            result = {"error": str(e), "status": "Routing Failed", "fallback": "Manual Intervention Required"}

        # 4. Meta-Cognition (Reflection)
        # Only reflect on complex outputs or if explicitly requested
        if route in self.config.reflect_on or context.get("force_reflection"):
            result = await self._reflect_on_result(result, query)

        # Log final outcome hash
        self.pot_logger.log_thought("MetaOrchestrator", "EXECUTION_COMPLETE", metadata={"status": "success", "complexity": route.value})

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

    def _assess_complexity(self, query: str, context: Optional[Dict[str, Any]] = None) -> Route:
        """
        Fast Path Heuristic Routing.
        Explicit context override first, then ordered word-boundary patterns; unmatched -> LOW.
        """
        forced = (context or {}).get("force_route")
        if forced:
            route = Route.coerce(forced)
            if route:
                return route
            logger.warning(f"Ignoring unknown force_route '{forced}'.")

        for route, pattern in ROUTE_PATTERNS:
            if pattern.search(query):
                return route
        return Route.LOW

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

    async def _run_deep_dive_flow(self, query: str, context: Dict[str, Any]):
        """
        The "Apex" Analysis Flow.
        Architecture: Waterfall Resilience.
        1. Attempt v26.0 Graph (Preferred).
        2. Fallback to Native Gemini (Path 3).
        3. Fallback to Legacy Agents (Manual Orchestration).
        """
        target = self.resolve_entity(query, context)
        if not target:
            return dict(CLARIFICATION_NO_ENTITY)

        logger.info("Engaging v26 Deep Dive Protocol...")

        # 0. Swarm Reconnaissance (Parallel Data Gathering)
        # We launch this regardless of the subsequent path to warm up data caches
        if not self.hive_mind.workers:
            await self.hive_mind.initialize()

        # Async fire-and-forget for swarm scouts
        recon = asyncio.create_task(self.hive_mind.disperse_task(
            "ANALYST",
            {"target": query, "purpose": "preliminary_deep_dive"},
            intensity=5.0
        ))
        self._background_tasks.add(recon)
        recon.add_done_callback(self._on_background_task_done)

        initial_state = init_omniscient_state(target)

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

    def _on_background_task_done(self, task: "asyncio.Task") -> None:
        self._background_tasks.discard(task)
        if task.cancelled():
            return
        exc = task.exception()
        if exc is not None:
            logger.error(f"Swarm reconnaissance task failed: {exc!r}", exc_info=exc)

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
            logger.info("Gemini Native Execution Complete.")
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
            # Non-standard PydanticAgentBase calls return AgentOutput.model_dump(); metrics live in metadata.
            fund_meta = fund_data.get("metadata", fund_data) if isinstance(fund_data, dict) else {}
            dcf_val = fund_meta.get("dcf_valuation", 0.0)
            raw_metrics = fund_meta.get("raw_metrics") or {}

            # Placeholder financials when the fundamentals agent returns nothing; flagged below.
            estimated_fields = [k for k in ("ebitda", "total_debt", "interest_expense") if raw_metrics.get(k) is None]
            ebitda = raw_metrics.get("ebitda") if raw_metrics.get("ebitda") is not None else 100.0
            total_debt = raw_metrics.get("total_debt") if raw_metrics.get("total_debt") is not None else 300.0
            interest_expense = raw_metrics.get("interest_expense")
            if interest_expense is None:
                interest_expense = ebitda / 4.0 if ebitda else 1.0
            if estimated_fields:
                logger.warning(f"Legacy deep dive using estimated financials for {company_id}: {estimated_fields}")

            # 3. Peers (Safe Execute)
            try:
                multiples = await agents["peer"].execute({"company_id": company_id})
            except Exception as e:
                logger.warning(f"Peer comparison failed for {company_id}: {e}")
                multiples = {}

            # 4. Credit & Risk
            financials_input = {
                "ebitda": float(ebitda),
                "total_debt": float(total_debt),
                "interest_expense": float(interest_expense)
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

            kg = V26KnowledgeGraph(
                meta=Meta(target=query, model_version="Adam-v26-Legacy-Fallback"),
                nodes=nodes
            )

            output = HyperDimensionalKnowledgeGraph(v26_knowledge_graph=kg).model_dump(by_alias=True)
            if estimated_fields:
                meta = output["v26_knowledge_graph"]["meta"]
                meta["_estimated"] = True
                meta["_estimated_fields"] = estimated_fields
            return output

        except Exception as e:
            logger.critical(f"All Deep Dive methods failed. {e}", exc_info=True)
            return {"error": "Deep Dive Failure", "details": str(e)}

    # --- Specialized Flows ---

    async def _run_swarm_flow(self, query: str, context: Dict[str, Any]):
        """
        Engages the massive parallelized Swarm Intelligence (MiroFish Integration).
        Replaces simple HiveMind task dispersion with a full cyclical reasoning workflow.
        """
        logger.info("Engaging Massive Parallelized Swarm Intelligence (MiroFish)...")

        target = query.replace("swarm", "").strip()

        try:
            # Phase 1: Environment Init (GraphRAG Extraction)
            await self.mirofish_engine.initialize_environment(
                seed_parameters={"target": target, "sentiment": "Bearish", "context": context}
            )

            # Phase 2: Agent Decoupling via Event Bus with dynamic wind-down
            # Allows the simulation to halt early if consensus is reached, saving tokens.
            simulation_data = await self.mirofish_engine.run_simulation_cycles(cycles=5)

            # Phase 3: Reporting Agent Tool Calling
            final_synthesis = await self.mirofish_engine.synthesize_report()

            # Phase 4: Embedded Neuro-Symbolic Validation
            # The Risk Officer (Planner/Reflector) critiques the emergent finding
            validation = await self._reflect_on_result(final_synthesis, f"Validate swarm output for: {target}")

            return {
                "status": "MiroFish Swarm Execution Complete",
                "active_personas": self.mirofish_engine.agent_count,
                "emergent_dynamics": final_synthesis,
                "raw_simulation_log": simulation_data,
                "neuro_symbolic_signoff": validation.get("meta_reflection", {}).get("score", 10) >= 8
            }

        except Exception as e:
            logger.error(f"MiroFish Swarm Failed: {e}. Gracefully falling back to Classical Crisis Engine.", exc_info=True)
            # Graceful Degradation: Fallback to the purely deterministic, linear CrisisSimulationEngine
            return await self._run_crisis_flow(query, context)

    async def _run_code_gen_flow(self, query: str, context: Dict[str, Any]):
        logger.info("Engaging Code Alchemist...")
        prompt = query.replace("generate code", "").strip()
        result = self.code_alchemist.generate_code(prompt)
        return {"status": "Code Generated", "result": result}

    async def _run_adaptive_flow(self, query: str, context: Dict[str, Any]):
        start_node = self.resolve_entity(query, context)
        if not start_node:
            return dict(CLARIFICATION_NO_ENTITY)

        logger.info("Engaging Neuro-Symbolic Planner...")
        # 1. Plan
        plan_data = self.planner.discover_plan(start_node, "RiskScore")

        # 2. Compile
        if not plan_data:
            return {"error": "Planning Failed"}
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

    async def _run_red_team_flow(self, query: str, context: Dict[str, Any]):
        entity = self.resolve_entity(query, context)
        if not entity:
            return dict(CLARIFICATION_NO_ENTITY)
        return await self._generic_graph_run(red_team_app, {
            "target_entity": entity,
            "scenario_type": "Cyber" if "cyber" in query else "Macro",
            "severity_threshold": 8.0
        }, "Red Team")

    async def _run_esg_flow(self, query: str, context: Dict[str, Any]):
        entity = self.resolve_entity(query, context)
        if not entity:
            return dict(CLARIFICATION_NO_ENTITY)
        return await self._generic_graph_run(esg_graph_app, init_esg_state(entity, "General"), "ESG")

    async def _run_compliance_flow(self, query: str, context: Dict[str, Any]):
        entity = self.resolve_entity(query, context)
        if not entity:
            return dict(CLARIFICATION_NO_ENTITY)
        return await self._generic_graph_run(compliance_graph_app, init_compliance_state(entity, "US"), "Compliance")

    async def _run_crisis_flow(self, query: str, context: Dict[str, Any]):
        portfolio = context.get("portfolio") or await self._construct_default_portfolio(context)
        if "allocation" in portfolio and "positions" not in portfolio:
            portfolio = self._positions_from_allocation(portfolio)
        return await self._generic_graph_run(crisis_simulation_app,
            init_crisis_state(query, portfolio), "Crisis Simulation")

    async def _run_surveillance_flow(self, query: str, context: Dict[str, Any]):
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

    # --- Default Portfolio Construction ---

    async def _construct_default_portfolio(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Builds a portfolio over `config.default_universe` via mean-variance optimization.
        Falls back to an equal-weight allocation flagged `_estimated` if prices or the optimizer are unavailable.
        """
        universe = list(context.get("universe") or self.config.default_universe)
        base = {"id": "Default Universe Portfolio", "aum": self.config.default_portfolio_aum, "universe": universe}

        prices, synthetic = None, False
        try:
            prices, synthetic = await asyncio.to_thread(self._fetch_universe_prices, universe)
        except Exception as e:
            logger.warning(f"Price history unavailable for default universe: {e}")

        if prices is not None:
            try:
                optimized = await PortfolioOptimizationAgent(config={"agent_id": "default_portfolio_constructor"}).execute(historical_prices=prices, method="mean_variance")
            except Exception as e:
                logger.warning(f"Portfolio optimization failed: {e}")
                optimized = {"status": "error", "message": str(e)}
            if optimized.get("status") == "success":
                portfolio = {
                    **base,
                    "allocation": optimized["allocation"],
                    "metrics": optimized["metrics"],
                    "method": optimized["method"],
                    "_constructed": True,
                }
                if synthetic:
                    portfolio["_synthetic_prices"] = True
                return portfolio
            logger.warning(f"Optimizer unavailable ({optimized.get('message')}); using equal-weight allocation.")

        weight = round(1.0 / len(universe), 4) if universe else 0.0
        return {
            **base,
            "allocation": {symbol: weight for symbol in universe},
            "metrics": {},
            "method": "equal_weight",
            "_constructed": False,
            "_estimated": True,
        }

    def _positions_from_allocation(self, portfolio: Dict[str, Any]) -> Dict[str, Any]:
        """
        Bridges an `allocation` ({symbol: weight}) into crisis-engine `positions` using
        `config.universe_metadata_path`. Symbols without metadata are skipped and listed in `_skipped_symbols`.
        """
        metadata = self._load_universe_metadata()
        aum = float(portfolio.get("aum") or self.config.default_portfolio_aum)
        positions, skipped = [], []
        for symbol, weight in portfolio["allocation"].items():
            meta = metadata.get(symbol)
            if not meta:
                skipped.append(symbol)
                continue
            positions.append({"name": symbol, "weight": float(weight), "exposure": float(weight) * aum, **meta})
        if skipped:
            logger.warning(f"No universe metadata for {skipped}; excluded from crisis positions.")
        return {**portfolio, "positions": positions, "_skipped_symbols": skipped}

    def _load_universe_metadata(self) -> Dict[str, Dict[str, Any]]:
        path = Path(self.config.universe_metadata_path)
        if not path.is_absolute() and not path.exists():
            path = Path(__file__).resolve().parents[2] / path
        try:
            return yaml.safe_load(path.read_text()) or {}
        except (OSError, yaml.YAMLError) as e:
            logger.warning(f"Universe metadata '{path}' unavailable: {e}")
            return {}

    def _fetch_universe_prices(self, universe: List[str]):
        """Returns (close-price DataFrame, used_synthetic_data) for the universe, or (None, False)."""
        import pandas as pd

        end = datetime.date.today()
        start = end - datetime.timedelta(days=self.config.price_history_days)
        service = MarketDataService()
        columns, synthetic = {}, False
        for symbol in universe:
            rows = service.get_historical_data(symbol, start.isoformat(), end.isoformat())
            if not rows:
                return None, False
            # Live yfinance rows use "Date"/"Close"; the synthetic fallback uses "date"/"close".
            live = "Close" in rows[0]
            synthetic = synthetic or not live
            dates = [r["Date" if live else "date"] for r in rows]
            closes = [r["Close" if live else "close"] for r in rows]
            series = pd.Series(closes, index=pd.to_datetime(dates, utc=True).normalize(), name=symbol)
            columns[symbol] = series[~series.index.duplicated()]
        frame = pd.DataFrame(columns).dropna()
        if len(frame) < 3:
            return None, False
        return frame, synthetic

    # --- Family Office (FO) Super-App Modules ---

    async def _run_fo_market(self, query: str, context: Dict[str, Any]):
        symbol = self.resolve_ticker(query, context)
        if not symbol:
            return dict(CLARIFICATION_NO_ENTITY)
        if "price" in query.lower():
            result = self.mcp_registry.invoke("price_asset", symbol=symbol, side="mid")
        else:
            result = self.mcp_registry.invoke("retrieve_market_data", symbol=symbol)
        return {"status": "FO Market Data", "data": result}

    async def _run_fo_execution(self, query: str, context: Dict[str, Any]):
        symbol = self.resolve_ticker(query, context)
        if not symbol:
            return dict(CLARIFICATION_NO_ENTITY)
        qty = self._resolve_quantity(query, context)
        if qty is None:
            return {"error": "no_quantity_resolved", "status": "clarification_needed", "symbol": symbol}
        side = "sell" if "sell" in query.lower() else "buy"

        result = self.mcp_registry.invoke("execute_order", order={"symbol": symbol, "side": side, "qty": qty})
        return {"status": "FO Order Submitted", "report": result}

    async def _run_fo_wealth(self, query: str, context: Dict[str, Any]):
        if "ips" in query.lower():
            result = self.mcp_registry.invoke("generate_ips", family_name="User", risk_profile="Growth")
            return {"status": "IPS Generated", "ips": result}

        # Default goal plan
        result = self.mcp_registry.invoke("plan_wealth_goal", goal_name="General", target_amount=1e6, horizon_years=10)
        return {"status": "Wealth Plan Generated", "plan": result}

    async def _run_fo_deal(self, query: str, context: Dict[str, Any]):
        result = self.mcp_registry.invoke("screen_deal", deal_name="Opportunity", sector="Tech", valuation=100.0)
        return {"status": "Deal Screened", "result": result}

    # --- Legacy Helpers ---

    async def _run_legacy_workflow(self, query: str, context: Dict[str, Any]):
        if self.legacy_orchestrator:
            return await self.legacy_orchestrator.execute_workflow("test_workflow", initial_context={"user_query": query})
        return {"status": "Legacy Orchestrator Unavailable", "query": query}

    async def _run_legacy_single_agent(self, query: str, context: Dict[str, Any]):
        if self.legacy_orchestrator:
            self.legacy_orchestrator.execute_agent("QueryUnderstandingAgent", context={"user_query": query})
            return {"status": "Dispatched to Message Broker", "query": query}
        return {"status": "Legacy Orchestrator Unavailable", "query": query}

    # --- Entity Resolution ---

    def resolve_ticker(self, query: str, context: Optional[Dict[str, Any]] = None) -> Optional[str]:
        """Context keys -> alias map -> $TICKER -> bare uppercase token. Returns None if unresolved."""
        context = context or {}
        for key in ("ticker", "symbol"):
            value = context.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip().upper()

        for alias, ticker in self.config.ticker_aliases.items():
            if re.search(rf"\b{re.escape(alias)}\b", query, re.IGNORECASE):
                return ticker

        match = _DOLLAR_TICKER_RE.search(query)
        if match:
            return match.group(1).upper()

        for token in _UPPER_TICKER_RE.findall(query):
            if token not in self.config.non_ticker_tokens:
                return token
        return None

    def resolve_entity(self, query: str, context: Optional[Dict[str, Any]] = None) -> Optional[str]:
        """Context entity keys -> ticker resolution -> capitalised proper noun. Returns None if unresolved."""
        context = context or {}
        for key in ("entity", "target_entity", "company", "company_name"):
            value = context.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()

        ticker = self.resolve_ticker(query, context)
        if ticker:
            return ticker

        # A proper noun introduced by "for/on/of/about/into" ("Summarize risks for Boeing") is the target;
        # otherwise the first proper noun wins ("Exxon credit outlook in Europe").
        candidates = []
        for match in _PROPER_NOUN_RE.finditer(query):
            words = [w for w in match.group(0).split() if w not in self.config.non_ticker_tokens]
            while words and words[0] in self.config.entity_stopwords:
                words.pop(0)
            if words:
                targeted = bool(_TARGET_PREP_RE.search(query[: match.start()]))
                candidates.append((not targeted, " ".join(words)))
        candidates.sort(key=lambda c: c[0])
        return candidates[0][1] if candidates else None

    @staticmethod
    def _resolve_quantity(query: str, context: Dict[str, Any]) -> Optional[float]:
        qty = context.get("qty", context.get("quantity"))
        if qty is None:
            match = _QTY_RE.search(query)
            qty = (match.group(1) or match.group(2)) if match else None
        try:
            value = float(qty)
        except (TypeError, ValueError):
            return None
        return value if math.isfinite(value) and value > 0 else None
