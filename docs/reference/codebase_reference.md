# Reference: Codebase Capabilities

A generated reference breakdown of the capabilities of key files in the repository via advanced AST parsing.

## Module: `core/vertical_risk_agent/agents`

### File: `core/vertical_risk_agent/agents/analyst.py`
_No general description available._

#### Component: `QuantAgent`
Specialized in extracting tabular data, running calculations, and analyzing Excel models.

- **Action: `analyze_financials(state)`**: Extracts financial data and calculates ratios.

### File: `core/vertical_risk_agent/agents/compliance.py`
_No general description available._

#### Component: `ComplianceAuditorAgent`
Verifies KYC/AML and Basel III capital constraints per the presentation (Slide 5).

- **Action: `run_compliance_audit(state)`**: Executes policy checks across constraints.

### File: `core/vertical_risk_agent/agents/document_extraction.py`
_No general description available._

#### Component: `DocumentExtractionAgent`
Pulls structured metrics from 10-K/10-Q filings, strictly enforcing Pydantic schemas.

- **Action: `extract_metrics(state)`**: Extracts document metrics into a Pydantic structure.

### File: `core/vertical_risk_agent/agents/legal.py`
_No general description available._

#### Component: `LegalAgent`
Specialized in RAG over long-context legal documents (Credit Agreements, Indentures).

- **Action: `analyze_covenants(state)`**: Extracts covenant definitions and logic.

### File: `core/vertical_risk_agent/agents/market.py`
_No general description available._

#### Component: `MarketAgent`
Specialized in web search and competitor analysis.

- **Action: `research_market(state)`**: Performs market research.

### File: `core/vertical_risk_agent/agents/supervisor.py`
_No general description available._

#### Component: `StateGraph`
A specialized component.

- **Action: `add_node(name, func)`**: Performs a specific task.
- **Action: `add_edge(start, end)`**: Performs a specific task.
- **Action: `add_conditional_edges(source, router, map)`**: Performs a specific task.
- **Action: `compile(checkpointer, interrupt_before)`**: Performs a specific task.
- **Action: `set_entry_point(name)`**: Performs a specific task.

#### Component: `CompiledGraphMock`
A specialized component.

- **Action: `invoke(state, config)`**: Performs a specific task.

#### Component: `MemorySaver`
A specialized component.


#### Function: `supervisor_node(state)`
The Supervisor decides which agent to call next or if the process is done.

#### Function: `route_supervisor(state)`
Conditional logic to determine the next node.

#### Function: `critique_node(state)`
Checks for consistency between Quant and Legal.
Now integrates the 4-Layered Evaluation & Verification Framework.

#### Function: `human_approval_node(state)`
Pauses for human sign-off.

## Module: `src/pdil`

### File: `src/pdil/fallbacks.py`
_No general description available._

#### Component: `CircuitBreaker`
Redundant fallback to prevent cascading failures in deterministic systems.

- **Action: `record_failure()`**: Performs a specific task.
- **Action: `reset()`**: Performs a specific task.

#### Component: `IndependentGatekeeperCheck`
Maintains the entire gatekeeper as a redundant, fallback modular independent check.
If primary systems fail or require an additive verification layer, this executes the full governance check.

- **Action: `perform_redundant_check(payload)`**: Executes an independent validation pass as a complementary check.
Returns True if valid, False if it violates governance (does not raise).

### File: `src/pdil/flows.py`
_No general description available._

#### Component: `ExecutionFlow`
Overlapping execution pipelines for redundant verifications.

- **Action: `execute(payload)`**: Performs a specific task.

### File: `src/pdil/lifecycle.py`
_No general description available._

#### Component: `ModelLifecycleManager`
Manages model and software lifecycle states.

- **Action: `register_model(model_id, version, deprecation_date)`**: Registers a model version in the lifecycle.
- **Action: `check_lifecycle(model_id)`**: Checks if a model is nearing deprecation or obsolete.

### File: `src/pdil/middleware.py`
_No general description available._

#### Component: `NoRedirectHandler`
A specialized component.

- **Action: `redirect_request(req, fp, code, msg, headers, newurl)`**: Performs a specific task.

#### Component: `NoRedirectHandler`
A specialized component.

- **Action: `redirect_request(req, fp, code, msg, headers, newurl)`**: Performs a specific task.

#### Component: `GovernanceError` (inherits: `Exception`)
Raised when an inference fails governance validation.


#### Component: `JsonLogicGovernanceGatekeeper`
A specialized component.

- **Action: `validate_inference(inference_output)`**: Validates LLM probabilistic inferences using jsonLogic.
- **Action: `entry_gate(inference_output)`**: Performs a specific task.
- **Action: `exit_gate(inference_output)`**: Performs a specific task.

#### Component: `SecurityGovernanceGatekeeper`
A specialized component.

- **Action: `validate_inference(inference_output)`**: Validates LLM probabilistic inferences natively using jsonschema.
- **Action: `entry_gate(inference_output)`**: Performs a specific task.
- **Action: `exit_gate(inference_output)`**: Performs a specific task.

#### Component: `DriftIntelligenceLayer`
A specialized component.

- **Action: `detect_and_heal_drift(inference_output, historical_hash)`**: Performs a specific task.
- **Action: `heal_drift(inference_output)`**: Performs a specific task.

#### Component: `ProofOfThoughtLogger`
A specialized component.

- **Action: `log_payload(payload, derivation_path, source_data_object)`**: Performs a specific task.

#### Component: `MilestoneLogger`
A specialized component.

- **Action: `add_milestone(name, details, complexity, conviction)`**: Performs a specific task.
- **Action: `get_most_efficient_process()`**: Performs a specific task.

### File: `src/pdil/models.py`
_No general description available._

#### Component: `ProvenanceHeader` (inherits: `BaseModel`)
W3C PROV-O compliant metadata header for probabilistic inferences.
This ensures that AI context windows can discern *why* a decision was made by tracing data back to its source.


### File: `src/pdil/optimization.py`
_No general description available._

#### Component: `TokenOptimizer`
Manages compute and token optimization for LLM interactions.

- **Action: `optimize_context(context)`**: Truncates or summarizes context to fit within token limits.
- **Action: `record_usage(model_name, tokens_used, cost_per_1k)`**: Tracks cost and compute utilization per model.

### File: `src/pdil/primitives.py`
_No general description available._

#### Component: `Primitive`
Base class for all PDIL primitives.


#### Component: `DataPrimitive` (inherits: `Primitive`)
Primitive for handling data inputs.


### File: `src/pdil/storage.py`
_No general description available._

#### Component: `DriftStorageBackend`
Storage backend for keeping historical drift data.

- **Action: `save_drift(execution_id, drift_info)`**: Performs a specific task.

### File: `src/pdil/system.py`
_No general description available._

#### Component: `SystemStateManager`
Manages meta-state, layer state, and context across the PDIL architecture.

- **Action: `update_layer_state(layer_name, state_delta)`**: Updates the state context of a specific architectural layer.
- **Action: `get_context(layer_name)`**: Retrieves combined context for a layer.
