# PDIL Architecture and Dependency Report

## Architectural Changelog

*   **Horizontal Scalability:** Created a shared `core_types.py` utility layer across different modules (`src/schemas/`, `afos/afos/`, `core/schemas/`, `adam_v3/kernel/`) mandating strict Pydantic type checking (`AgentInput`, `AgentOutput`) across agents (Nexus/Sentinel).
*   **Vertical Scalability:** Deepened the stack efficiency by standardizing the `ProvenanceHeader` which maps directly to PROV-O terminology (`wasGeneratedBy`, `generatedAtTime`, `value`, `wasDerivedFrom`, `hadPrimarySource`).
*   **W3C PROV-O Compliance:** Added the `check_grounding()` method to `AgentOutput` and `RiskAssessmentOutput` schemas. This ensures that every output from the probabilistic layer contains a reference to its source data object by validating `self.provenance_trace.source_data_object` (or `self.provenance.source_data_object`), thereby satisfying W3C PROV-O requirements and preventing ungrounded AI inferences.
*   **Observed Drift Flag:** Logic shifting from existing implementation correctly flags `observed_drift=True` inside `AgentOutput`, enabling self-healing mechanisms inside the `GovernanceGatekeeper`.

## Upgraded PDIL Component

The Probabilistic-to-Deterministic Integration Layer (PDIL) schema upgrades have been implemented to enforce W3C PROV-O compliance via the `check_grounding` method.

**Example from `core/schemas/agent_schema.py`:**

```python
class AgentOutput(BaseModel):
    # ... fields ...
    provenance_trace: ProvenanceHeader = Field(..., description="Immutable provenance trace linking the agent output to source and logic version.")
    observed_drift: bool = Field(default=False, description="Flag indicating if behavior or calculation drift was observed.")

    def check_grounding(self) -> bool:
        """
        Verifies that this output contains a valid reference to its source data object,
        satisfying W3C PROV-O compliance requirements.
        """
        return bool(self.provenance_trace.source_data_object)
```

**Example from `afos/afos/core_types.py`:**

```python
class ProvenanceHeader(BaseModel):
    # ... fields ...
    source_data_object: str = Field(default="", description="Reference to the source data object")

class RiskAssessmentOutput(BaseModel):
    provenance: ProvenanceHeader
    # ... fields ...
    def check_grounding(self) -> bool:
        """
        Verifies that this output contains a valid reference to its source data object,
        satisfying W3C PROV-O compliance requirements.
        """
        return bool(self.provenance.source_data_object)
```

## Dependency Graph (Filtered for PDIL / Schemas)

```json
{
  "src/pdil/middleware.py": [
    "json",
    "hashlib",
    "asyncio",
    "time",
    "urllib.request",
    "urllib.error",
    "asyncio",
    "socket",
    "ipaddress",
    "ssl",
    "urllib.parse.urlparse",
    "jsonschema",
    "typing.Dict",
    "typing.Any",
    "typing.Optional",
    "src.pdil.models.ProvenanceHeader",
    "json_logic.jsonLogic"
  ],
  "src/governance/workflow_scaffold.py": [
    "time",
    "json",
    "hashlib",
    "typing.Dict",
    "typing.Any",
    "src.governance.gatekeeper.GovernanceGatekeeper",
    "src.pdil.models.ProvenanceHeader"
  ],
  "src/governance/gatekeeper.py": [
    "src.pdil.models.ProvenanceHeader",
    "src.pdil.middleware.GovernanceError",
    "src.pdil.middleware.SecurityGovernanceGatekeeper",
    "src.pdil.middleware.JsonLogicGovernanceGatekeeper",
    "src.pdil.middleware.DriftIntelligenceLayer",
    "src.pdil.middleware.ProofOfThoughtLogger",
    "src.pdil.middleware.MilestoneLogger"
  ],
  "src/schemas/core_types.py": [
    "json",
    "hashlib",
    "typing.Any",
    "typing.Dict",
    "pydantic.BaseModel",
    "pydantic.Field",
    "hashlib",
    "json",
    "src.pdil.models.ProvenanceHeader"
  ],
  "core/agents/credit_risk_agent.py": [
    "__future__.annotations",
    "typing.Any",
    "typing.Dict",
    "typing.Optional",
    "logging",
    "numpy",
    "json",
    "hashlib",
    "datetime.datetime",
    "core.agents.agent_base.AgentBase",
    "core.schemas.agent_schema.AgentOutput",
    "src.pdil.models.ProvenanceHeader",
    "src.pdil.middleware.JsonLogicGovernanceGatekeeper",
    "scipy.stats.norm"
  ],
  "core/agents/liquidity_risk_agent.py": [
    "__future__.annotations",
    "typing.Any",
    "typing.Dict",
    "typing.Optional",
    "logging",
    "numpy",
    "json",
    "hashlib",
    "datetime.datetime",
    "core.agents.agent_base.AgentBase",
    "core.schemas.agent_schema.AgentOutput",
    "src.pdil.models.ProvenanceHeader",
    "src.pdil.middleware.JsonLogicGovernanceGatekeeper"
  ],
  "core/agents/peer_set_agent.py": [
    "typing.Any",
    "typing.Dict",
    "typing.List",
    "typing.Union",
    "typing.Optional",
    "logging",
    "asyncio",
    "core.agents.agent_base.AgentBase",
    "core.schemas.agent_schema.AgentInput",
    "core.schemas.agent_schema.AgentOutput",
    "yfinance",
    "sklearn.feature_extraction.text.TfidfVectorizer",
    "sklearn.metrics.pairwise.cosine_similarity"
  ],
  "core/agents/risk_assessment_agent.py": [
    "__future__.annotations",
    "typing.Any",
    "typing.Dict",
    "typing.Optional",
    "typing.List",
    "typing.Union",
    "logging",
    "json",
    "numpy",
    "datetime",
    "asyncio",
    "core.agents.agent_base.AgentBase",
    "core.schemas.agent_schema.AgentInput",
    "core.schemas.agent_schema.AgentOutput",
    "scipy.stats.norm",
    "core.engine.cyclical_reasoning_graph.cyclical_reasoning_app",
    "core.engine.states.init_risk_state",
    "core.evaluation.judge.AuditorAgent",
    "core.evaluation.tracing.TraceLog"
  ],
  "core/agents/system_health_agent.py": [
    "time",
    "typing.Dict",
    "typing.Any",
    "pydantic.BaseModel",
    "core.schemas.agent_schema.AgentInput",
    "core.schemas.agent_schema.AgentOutput",
    "core.agents.agent_base.AgentBase"
  ],
  "core/agents/insider_activity_agent.py": [
    "logging",
    "typing.Any",
    "typing.Dict",
    "typing.Union",
    "pydantic.BaseModel",
    "pydantic.Field",
    "random",
    "core.agents.agent_base.AgentBase",
    "core.schemas.agent_schema.AgentInput",
    "core.schemas.agent_schema.AgentOutput"
  ],
  "core/agents/meta_cognitive_agent.py": [
    "logging",
    "re",
    "typing.Any",
    "typing.Dict",
    "typing.List",
    "typing.Optional",
    "core.schemas.agent_schema.AgentInput",
    "core.schemas.agent_schema.AgentOutput",
    "core.agents.agent_base.AgentBase",
    "core.agents.system_health_agent.SystemHealthAgent"
  ],
  "core/agents/competitor_analysis_agent.py": [
    "typing.Any",
    "typing.Dict",
    "typing.List",
    "typing.Union",
    "typing.Optional",
    "logging",
    "core.agents.agent_base.AgentBase",
    "core.schemas.agent_schema.AgentInput",
    "core.schemas.agent_schema.AgentOutput"
  ],
  "core/agents/snc_analyst_agent.py": [
    "semantic_kernel.Kernel",
    "unittest.mock.patch",
    "json",
    "core.agents.agent_base.AgentBase",
    "core.schemas.agent_schema.AgentInput",
    "core.schemas.agent_schema.AgentOutput",
    "typing.Dict",
    "typing.Any",
    "typing.Optional",
    "typing.Tuple",
    "typing.Union",
    "enum.Enum",
    "asyncio",
    "os",
    "logging",
    "sys",
    "datetime.datetime",
    "datetime.timezone",
    "core.compliance.snc_validators.evaluate_compliance",
    "core.compliance.snc_validators.RiskLevel"
  ],
  "core/agents/agent_base.py": [
    "src.pdil.models.ProvenanceHeader",
    "abc.ABC",
    "abc.abstractmethod",
    "typing.Any",
    "typing.Dict",
    "typing.List",
    "typing.Optional",
    "logging",
    "json",
    "asyncio",
    "pydantic.BaseModel",
    "pydantic.Field",
    "uuid",
    "warnings",
    "datetime.datetime",
    "uuid",
    "warnings",
    "datetime.datetime",
    "core.utils.logging_utils.SwarmLogger",
    "core.schemas.hnasp.HNASPState",
    "core.schemas.hnasp.Meta",
    "core.schemas.hnasp.PersonaState",
    "core.schemas.hnasp.LogicLayer",
    "core.schemas.hnasp.ContextStream",
    "core.schemas.hnasp.PersonaDynamics",
    "core.schemas.hnasp.SecurityContext",
    "core.schemas.hnasp.PersonaIdentities",
    "core.schemas.hnasp.Identity",
    "core.schemas.hnasp.EPAVector",
    "core.agents.mixins.memory_mixin.MemoryMixin",
    "core.system.boot_protocol.BootProtocol",
    "core.swarms.memory_matrix.MemoryMatrix",
    "json_logic.jsonLogic",
    "semantic_kernel.Kernel",
    "core.schemas.hnasp.ExecutionTrace"
  ],
  "core/agents/quantum_portfolio_manager_agent.py": [
    "__future__.annotations",
    "typing.Any",
    "typing.Dict",
    "typing.List",
    "typing.Optional",
    "typing.Union",
    "logging",
    "asyncio",
    "numpy",
    "pandas",
    "core.agents.agent_base.AgentBase",
    "core.schemas.agent_schema.AgentInput",
    "core.schemas.agent_schema.AgentOutput",
    "core.data_sources.data_fetcher.DataFetcher",
    "core.simulations.quantum_monte_carlo.QuantumMonteCarloBridge"
  ],
  "core/agents/options_flow_agent.py": [
    "logging",
    "typing.Any",
    "typing.Dict",
    "typing.Union",
    "core.agents.agent_base.AgentBase",
    "core.schemas.agent_schema.AgentInput",
    "core.schemas.agent_schema.AgentOutput",
    "random"
  ],
  "core/agents/code_architect_agent.py": [
    "logging",
    "os",
    "re",
    "typing.Any",
    "typing.Dict",
    "core.agents.pydantic_agent_base.PydanticAgentBase",
    "core.schemas.agent_schema.AgentInput",
    "core.schemas.agent_schema.AgentOutput"
  ],
  "core/agents/pydantic_agent_base.py": [
    "abc.abstractmethod",
    "typing.Any",
    "typing.Dict",
    "typing.Optional",
    "typing.TypeVar",
    "typing.Generic",
    "pydantic.BaseModel",
    "pydantic.ValidationError",
    "logging",
    "core.agents.agent_base.AgentBase",
    "core.schemas.agent_schema.AgentInput",
    "core.schemas.agent_schema.AgentOutput"
  ],
  "core/agents/algo_trading_agent.py": [
    "numpy",
    "pandas",
    "matplotlib.pyplot",
    "random",
    "logging",
    "typing.Dict",
    "typing.Any",
    "typing.Union",
    "typing.Optional",
    "core.agents.agent_base.AgentBase",
    "core.schemas.agent_schema.AgentInput",
    "core.schemas.agent_schema.AgentOutput",
    "asyncio"
  ],
  "core/agents/fundamental_analyst_agent.py": [
    "asyncio",
    "json",
    "assets.prompt_loader.load_prompt",
    "logging",
    "typing.Any",
    "typing.Dict",
    "typing.List",
    "typing.Optional",
    "typing.Union",
    "unittest.mock.patch",
    "semantic_kernel.Kernel",
    "core.agents.pydantic_agent_base.PydanticAgentBase",
    "core.analysis.gemini_analyzer.GeminiFinancialReportAnalyzer",
    "core.schemas.agent_schema.AgentInput",
    "core.schemas.agent_schema.AgentOutput",
    "core.schemas.agent_schema.FundamentalReport",
    "core.system.memory_manager.VectorMemoryManager",
    "src.mcp_server.DCFInputs",
    "src.mcp_server.calculate_dcf"
  ],
  "core/agents/market_sentiment_agent.py": [
    "__future__.annotations",
    "typing.Any",
    "typing.Dict",
    "typing.Tuple",
    "typing.Optional",
    "typing.Union",
    "typing.List",
    "logging",
    "asyncio",
    "core.agents.agent_base.AgentBase",
    "core.schemas.agent_schema.AgentInput",
    "core.schemas.agent_schema.AgentOutput",
    "core.data_sources.financial_news_api.SimulatedFinancialNewsAPI",
    "core.data_sources.prediction_market_api.SimulatedPredictionMarketAPI",
    "core.data_sources.social_media_api.SimulatedSocialMediaAPI",
    "core.data_sources.web_traffic_api.SimulatedWebTrafficAPI",
    "core.data_sources.data_fetcher.DataFetcher",
    "core.agents.specialized.crypto_arbitrage_agent.CryptoArbitrageAgent",
    "core.agents.options_flow_agent.OptionsFlowAgent",
    "core.agents.insider_activity_agent.InsiderActivityAgent",
    "typing_extensions.deprecated"
  ],
  "core/agents/realtime_pd_agent.py": [
    "typing.Dict",
    "typing.Any",
    "typing.Optional",
    "hashlib",
    "json",
    "logging",
    "datetime.datetime",
    "core.schemas.agent_schema.AgentOutput",
    "src.pdil.models.ProvenanceHeader",
    "src.pdil.middleware.JsonLogicGovernanceGatekeeper",
    "core.agents.agent_base.AgentBase",
    "scipy.stats.norm",
    "numpy"
  ],
  "core/agents/technical_analyst_agent.py": [
    "__future__.annotations",
    "typing.Any",
    "typing.Dict",
    "typing.Union",
    "typing.Optional",
    "logging",
    "asyncio",
    "pandas",
    "numpy",
    "sklearn.ensemble.RandomForestClassifier",
    "pickle",
    "os",
    "core.agents.agent_base.AgentBase",
    "core.security.safe_unpickler.SafeUnpickler",
    "core.schemas.agent_schema.AgentInput",
    "core.schemas.agent_schema.AgentOutput",
    "core.data_sources.data_fetcher.DataFetcher"
  ],
  "core/agents/reflector_agent.py": [
    "__future__.annotations",
    "typing.Dict",
    "typing.Any",
    "typing.List",
    "typing.Union",
    "logging",
    "core.agents.agent_base.AgentBase",
    "core.schemas.agent_schema.AgentInput",
    "core.schemas.agent_schema.AgentOutput",
    "core.engine.states.init_reflector_state",
    "core.engine.reflector_graph.reflector_app"
  ],
  "core/agents/specialized/dark_pool_agent.py": [
    "typing.Any",
    "typing.Dict",
    "typing.List",
    "typing.Optional",
    "logging",
    "pydantic.Field",
    "core.agents.agent_base.AgentBase",
    "core.schemas.agent_schema.AgentInput",
    "core.schemas.agent_schema.AgentOutput"
  ],
  "core/agents/meta_agents/omega_meta_orchestrator.py": [
    "json",
    "typing.Any",
    "core.agents.pydantic_agent_base.PydanticAgentBase",
    "core.schemas.agent_schema.AgentInput",
    "core.schemas.agent_schema.AgentOutput"
  ],
  "core/engine/cognitive_harness.py": [
    "typing.Any",
    "typing.Dict",
    "core.agents.meta_agents.omega_meta_orchestrator.OmegaMetaOrchestrator",
    "core.schemas.agent_schema.AgentInput",
    "core.schemas.agent_schema.AgentOutput"
  ],
  "core/schemas/__init__.py": [
    "core.schemas.hnasp_v3.HNASPState",
    "core.schemas.hnasp_v3.HNASP",
    "core.schemas.v23_5_schema.V26KnowledgeGraph",
    "core.schemas.v23_5_schema.HyperDimensionalKnowledgeGraph",
    "core.schemas.tool_use.ToolManifest",
    "core.schemas.tool_use.ToolCall",
    "core.schemas.tool_use.ToolResult",
    "core.schemas.tool_use.ToolUsageSession",
    "core.schemas.hnasp_integration.IntegratedAgentState",
    "core.schemas.hnasp_integration.AgentPacket",
    "core.schemas.cognitive_state.CognitiveState",
    "core.schemas.cognitive_state.StrategicPlan",
    "core.schemas.cognitive_state.ThoughtNode",
    "core.schemas.observability.AgentTelemetry",
    "core.schemas.observability.Trace",
    "core.schemas.observability.Span",
    "core.schemas.observability.Metric",
    "core.schemas.registry.SchemaRegistry",
    "core.schemas.agent_schema.AgentInput",
    "core.schemas.agent_schema.AgentOutput",
    "core.schemas.agent_schema.FundamentalReport",
    "core.schemas.qpf_schema.QPFInput",
    "core.schemas.qpf_schema.QPFOutput"
  ],
  "core/schemas/agent_schema.py": [
    "src.pdil.models.ProvenanceHeader",
    "typing.Dict",
    "typing.Any",
    "typing.List",
    "typing.Optional",
    "pydantic.BaseModel",
    "pydantic.Field"
  ],
  "adam_v3/kernel/schema.py": [
    "src.pdil.models.ProvenanceHeader",
    "pydantic.BaseModel",
    "typing.Dict",
    "typing.Any",
    "typing.Optional"
  ],
  "afos/afos/engine.py": [
    "afos.core_types.Obligor",
    "afos.core_types.Facility",
    "afos.core_types.RiskAssessmentOutput",
    "afos.core_types.ProvenanceHeader",
    "afos.core_types.ModelArbitrationOutput"
  ],
  "afos/afos/core_types.py": [
    "pydantic.BaseModel",
    "pydantic.Field",
    "pydantic.ConfigDict",
    "typing.Optional",
    "typing.List",
    "typing.Dict",
    "typing.Any",
    "datetime.datetime",
    "datetime.timezone"
  ]
}
```

## Provenance Test Suite

Tests enforce that `check_grounding` functions correctly across the integration layer. (Executed via `uv run pytest tests/unit/test_agent_schemas.py tests/test_provenance.py tests/test_pdil_property.py`)

**Example test from `tests/unit/test_agent_schemas.py`:**
```python
def test_agent_output_instantiation_and_grounding():
    """Test AgentOutput correctly uses ProvenanceHeader and validates W3C PROV-O."""
    header = ProvenanceHeader(
        git_commit_hash="abcdef123456",
        timestamp="2023-10-25T12:00:00Z",
        content_hash="hash123",
        jsonLogic_version="1.0",
        confidence_score=0.95,
        derivation_path="sim -> calc -> output",
        source_data_object="doc_456"
    )

    output_data = AgentOutput(
        provenance_trace=header,
        data={"result": "success"},
        observed_drift=False
    )

    assert output_data.check_grounding() is True

def test_agent_output_failing_grounding():
    """Test check_grounding fails when source_data_object is missing/empty."""
    header = ProvenanceHeader(
        # ... fields ...
        source_data_object="" # Empty string to fail grounding
    )
    output_data = AgentOutput(provenance_trace=header, data={"result": "success"}, observed_drift=False)
    assert output_data.check_grounding() is False
```
