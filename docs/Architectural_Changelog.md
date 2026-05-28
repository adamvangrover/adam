# Architectural Changelog

## Dependency Mapping
This maps the dependencies across modules, templates, schemas, and core utilities inside machine learning, documentation, orchestration, agent skills, mcp servers, and core utility files in the 'Adam' repository.

```json
{
  "src/mcp_server.py": [
    "src.core_valuation.ValuationEngine",
    "src.config.DEFAULT_ASSUMPTIONS",
    "core.risk_engine.quantum_model.calculate_quantum_var",
    "core.engine.cyclical_reasoning_graph.CyclicalReasoningGraph",
    "core.risk_engine.quantum_model.calculate_quantum_var"
  ],
  "src/llm/agent.py": [
    "src.core.config.settings"
  ],
  "src/api/routers/ingest.py": [
    "src.orchestration.tasks.process_data_pipeline",
    "src.orchestration.tasks.redis_client",
    "src.core.logging.logger"
  ],
  "src/adam/api/main.py": [
    "src.adam.api.models.OptimizationRequest",
    "src.adam.api.models.OptimizationResponse",
    "src.adam.api.auth.get_current_user",
    "src.adam.core.optimizers.AdamW",
    "src.adam.core.optimizers.Lion",
    "src.adam.core.optimizers.AdamMini",
    "src.adam.core.state_manager.StateManager"
  ],
  "src/adam/core/state_manager.py": [
    "core.security.safe_unpickler.safe_loads",
    "core.security.safe_unpickler.safe_loads"
  ],
  "src/ingestion/plugin_manager.py": [
    "src.core.logging.logger"
  ],
  "src/ingestion/plugins/pdf_parser.py": [
    "src.ingestion.base.IngestionStrategy",
    "src.core.logging.logger"
  ],
  "src/ingestion/plugins/excel_parser.py": [
    "src.core.logging.logger"
  ],
  "src/orchestration/celery_app.py": [
    "src.core.config.settings"
  ],
  "src/orchestration/tasks.py": [
    "src.core.config.settings",
    "src.core.logging.logger",
    "src.ingestion.plugin_manager.plugin_manager",
    "src.llm.agent.get_agent",
    "src.llm.schemas.SpreadsheetBatchOutput"
  ],
  "src/schemas/core_types.py": [
    "src.governance.gatekeeper.ProvenanceHeader"
  ],
  "server/mcp_server.py": [
    "core.v22_quantum_pipeline.qmc_engine.QuantumMonteCarloEngine",
    "core.vertical_risk_agent.generative_risk.GenerativeRiskEngine",
    "core.agents.specialized.snc_rating_agent.SNCRatingAgent",
    "core.agents.specialized.covenant_analyst_agent.CovenantAnalystAgent",
    "core.agents.specialized.peer_comparison_agent.PeerComparisonAgent",
    "core.engine.neuro_symbolic_planner.NeuroSymbolicPlanner",
    "core.data_processing.universal_ingestor.UniversalIngestor",
    "core.security.governance.GovernanceEnforcer",
    "core.security.governance.GovernanceError",
    "core.security.governance.ApprovalRequired",
    "core.security.eaci_middleware.EACIMiddleware",
    "core.security.sandbox.SecureSandbox"
  ],
  "server/server.py": [
    "core.security.sql_validator.SQLValidator",
    "core.vertical_risk_agent.generative_risk.GenerativeRiskEngine",
    "core.v22_quantum_pipeline.qmc_engine.QuantumMonteCarloEngine",
    "core.engine.meta_orchestrator.MetaOrchestrator",
    "core.credit_sentinel.agents.ratio_calculator.RatioCalculator",
    "core.credit_sentinel.models.distress_classifier.DistressClassifier",
    "core.credit_sentinel.agents.risk_analyst.RiskAnalyst",
    "core.credit_sentinel.agents.risk_analyst.AgentInput",
    "core.agents.risk_assessment_agent.RiskAssessmentAgent",
    "core.governance.constitution.Constitution",
    "core.security.governance.GovernanceEnforcer",
    "core.security.governance.GovernanceError",
    "core.security.governance.ApprovalRequired",
    "adam_core_rust",
    "core.security.sandbox.SecureSandbox"
  ],
  "server/sentinel_api.py": [
    "core.governance.sentinel_harness.DecisionState",
    "core.governance.sentinel_harness.run_credit_workflow"
  ]
}
```

## Enhancements
- **Horizontal & Vertical Scalability:** Created a shared `core_types.py` utility layer mandating strict Pydantic type checking (`AgentInput`, `AgentOutput`) across agents (Nexus/Sentinel). Added PROV-O compliance to `AgentOutput`.
- **Probabilistic-to-Deterministic Integration Layer (PDIL):** Upgraded `GovernanceGatekeeper` to include strict jsonschema validation, enforce confidence scoring, implement async validation pipeline, and introduce autonomous self-healing logic.
- **W3C PROV-O Compliance:** Verified groundedness by checking W3C PROV-O requirements where every output from the probabilistic layer contains a reference to its source data object. Upgraded `ProvenanceHeader` to map directly to PROV-O terminology (`wasGeneratedBy`, `generatedAtTime`, `value`, `wasDerivedFrom`, `hadPrimarySource`).
- **Observed Drift Flag:** Logic shifting from existing implementation will correctly flag `observed_drift=True` as defined in `AgentOutput`, and is now caught by `heal_drift` inside `GovernanceGatekeeper`.

## Drift Analysis
- **Observed Drift:** True (Upgraded PDIL layer, standardized PROV-O metadata, added async concurrency patterns and self-healing protocols).
